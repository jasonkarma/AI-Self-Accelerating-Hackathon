import json, sqlite3
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.constants import WT_SEQUENCE
from core.features import encode_sequence, encode_sequences
from core.developability import RiskNormalizer, raw_components
from core.preregistration import freeze_protocol
from core.ledger import ImmutableLedger
from core.oracle import BindingOracle, DevelopabilityOracle
from core.schemas import PositionalSoftPrior
from core.surrogate import BindingSurrogate, link_predictions
from core.tool_arsenal import TOOL_REGISTRY, configure_candidate_lookup, profile_developability_liabilities
from data.datasets import generate_fixture, public_pool, extract_hcdr3, prepare_abbibench_frame

def test_features_are_exactly_299_and_position_aware():
    x=encode_sequence(WT_SEQUENCE); assert x.shape==(299,); assert x[:260].sum()==13
    assert not np.array_equal(x,encode_sequence("ARWGGDGFYAMDY"))

def test_risk_and_label_blind_freeze(tmp_path):
    frame=generate_fixture(); public=public_pool(frame); normalizer=RiskNormalizer.fit(public.sequence)
    score=normalizer.score(WT_SEQUENCE); assert 0<=score["risk"]<=1; assert score["y_dev"]==pytest.approx(1-score["risk"])
    protocol=freeze_protocol(public,output=tmp_path/"p.json",is_synthetic=True); assert protocol["threshold_source"] in {"wild_type","pool_median"}; assert protocol["is_synthetic"]
    assert protocol["qualified_count"] == round(protocol["qualification_pass_rate"] * len(public))
    if protocol["threshold_source"] == "pool_median":
        assert protocol["threshold_percentile"] == 0.50
    with pytest.raises(ValueError): freeze_protocol(frame,output=tmp_path/"bad.json",is_synthetic=True)

def test_ledger_immutable_and_oracles_idempotent(tmp_path):
    frame=generate_fixture(10); normalizer=RiskNormalizer.fit(frame.sequence); ledger=ImmutableLedger(tmp_path/"l.db",metadata={"m5_policy_mode":"adaptive_tool"})
    bo=BindingOracle(frame,ledger); do=DevelopabilityOracle(frame,ledger,normalizer)
    first=bo.query(["WT"],round_number=0); second=bo.query(["WT"],round_number=0); do.query(["WT"],round_number=0)
    assert not first[0]["cached"] and second[0]["cached"] and len(ledger.dataframe("oracle_reveal"))==2 and ledger.verify()
    with ledger.connect() as con:
        with pytest.raises(sqlite3.DatabaseError): con.execute("DELETE FROM entries")

def test_surrogate_and_linked_uncertainty():
    X=np.arange(20*4,dtype=float).reshape(20,4); y=np.sin(np.arange(20)); model=BindingSurrogate(3).fit(X,y); mu,sigma=model.predict(X)
    assert len(mu)==20 and np.all((0<=sigma)&(sigma<=1))
    linked=link_predictions(mu,sigma,np.ones(20)*.8,sigma,.7); assert np.allclose(linked.sigma_joint,np.sqrt(.65*sigma**2+.35*sigma**2))

def test_tool_registry_and_liability_profile():
    assert len(TOOL_REGISTRY)==6; configure_candidate_lookup({"WT":WT_SEQUENCE}); row=profile_developability_liabilities(["WT"])["rows"][0]; assert row["candidate_id"]=="WT"

def test_soft_prior_bounds():
    with pytest.raises(Exception): PositionalSoftPrior(position=1,weight_delta=.16)

def test_api_contract_and_risk_components():
    from app.server import app
    with TestClient(app) as client:
        # ``api_contract.json`` is a bundle of four endpoints now, so the state
        # document lives under ``contract["state"]``. The invariant is unchanged:
        # the contract and the live payload must have exactly the same key set, so
        # the frontend can never reference a field the backend does not serve.
        state=client.get("/api/state").json(); contract=json.load(open("api_contract.json"))["state"]; assert state.keys()==contract.keys()
        assert client.options("/api/state",headers={"Origin":"http://localhost:5173","Access-Control-Request-Method":"GET"}).headers["access-control-allow-origin"]=="*"
        candidate=client.get("/api/candidates?limit=1").json()["items"][0]; assert len(candidate["developability_components_raw"])==6 and len(candidate["developability_components_normalized"])==6

def test_real_abbibench_ingestion_preserves_replicates_without_fabricating_wt():
    prefix="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC"
    suffix="WGQGTLVTVSSA"
    source=pd.DataFrame({"heavy_chain_seq":[prefix+"ARSGGSSYYYYDY"+suffix,prefix+"ARSGGSSYYYYDY"+suffix],"binding_score":[6.1,5.9]})
    assert extract_hcdr3(source.heavy_chain_seq.iloc[0])=="ARSGGSSYYYYDY"
    result=prepare_abbibench_frame(source,prefix="CAND",expected_rows=2)
    assert len(result)==2 and result.candidate_id.nunique()==2 and result.sequence.nunique()==1

def test_protocol_same_hash_migrates_new_diagnostics(tmp_path):
    public=public_pool(generate_fixture())
    path=tmp_path/"protocol.json"
    protocol=freeze_protocol(public,output=path,is_synthetic=True)
    legacy={key:value for key,value in protocol.items() if key not in {"qualification_pass_rate","qualified_count","threshold_percentile"}}
    path.write_text(json.dumps(legacy))
    migrated=freeze_protocol(public,output=path,is_synthetic=True)
    assert {"qualification_pass_rate","qualified_count","threshold_percentile"} <= migrated.keys()
