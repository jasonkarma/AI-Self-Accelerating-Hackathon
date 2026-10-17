from data.datasets import generate_fixture
from core.developability import RiskNormalizer
from core.baselines import METHODS, run_method

def test_methods_share_seeds_and_spend_35_queries():
    frame=generate_fixture(70); norm=RiskNormalizer.fit(frame.sequence); initial=frame.candidate_id.head(5).tolist(); tau=norm.score(frame.sequence.iloc[0])["y_dev"]
    results=[run_method(frame,norm,tau,m,0,initial) for m in METHODS]
    assert all(r.initial_ids==initial and [x["query_count"] for x in r.curve]==list(range(5,36,5)) for r in results)
