"""Backend/ledger synchronization guards for the API surface.

Two layers:

* :func:`test_campaign_run_exposes_voi_and_hierarchical_sar_diagnostics` drives a
  six-round campaign against a **temporary** ledger, so the VoI/SAR diagnostics
  are asserted from a freshly built run rather than from whatever happens to be
  in the developer's ``results/`` directory.
* :func:`test_api_state_and_benchmark_match_the_frozen_artifacts` boots the real
  FastAPI app and checks the two endpoints the frontend reads. It skips when the
  frozen artifacts are absent, so a clean checkout still runs the suite — but
  when ``scripts/run_campaign.py`` and ``scripts/run_benchmark.py`` have been
  run, it is the guard that keeps the API and the artifacts in lockstep.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config.settings import SETTINGS
from core.acquisition import VOI_EXPLOIT_WEIGHT
from core.baselines import METHODS, PAIRED_METRICS
from core.campaign import Campaign
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import generate_fixture, public_pool


def _campaign(tmp_path) -> Campaign:
    frame = generate_fixture(160)
    protocol = freeze_protocol(public_pool(frame), is_synthetic=True)
    normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
    return Campaign(
        frame,
        normalizer,
        protocol,
        {},
        ledger_path=tmp_path / "campaign.sqlite3",
    )


@pytest.mark.asyncio
async def test_campaign_run_exposes_voi_and_hierarchical_sar_diagnostics(tmp_path):
    """Six offline rounds must persist and surface the new architecture.

    The offline council is the deterministic fallback, which is deliberate: this
    test is about the *loop* (surrogates -> quota -> oracles -> ledger -> state),
    not about model quality.
    """
    from config.settings import SETTINGS as settings

    lab = _campaign(tmp_path)
    for _ in range(min(2, settings.rounds)):
        await lab.execute_round()

    state = lab.state
    diagnostics = state["voi_diagnostics"]
    # The VoI terms are computed once per batch and shared by every slot.
    for key in ("voi", "horizon", "sar_maturity", "exploit_weight", "exploration_weight",
                "slot_split", "position_tolerance_effects"):
        assert key in diagnostics, key
    assert diagnostics["round_number"] == state["current_round"]
    assert 0.0 <= diagnostics["voi"] <= 1.0
    assert diagnostics["exploit_weight"] == pytest.approx(
        1.0 - VOI_EXPLOIT_WEIGHT * diagnostics["voi"], abs=1e-4
    )
    assert diagnostics["slot_split"] == "3-exploit / 2-experimental"

    # Every slot carries the paired-arm scores behind its selection.
    slots = state["batch_slots"]
    assert len(slots) == settings.batch_size
    for slot in slots:
        for key in ("a_null", "a_h1", "delta_agent", "sigma_joint", "voi",
                    "horizon", "sar_maturity"):
            assert key in slot, key
        assert slot["a_h1"] == pytest.approx(slot["a_null"] + slot["delta_agent"], abs=1e-3)
        assert slot["voi"] == pytest.approx(diagnostics["voi"], abs=1e-4)

    # The 3-exploit / 2-experimental split, read off slot order:
    # * slots 1-2 are the two strongest ``a_h1`` picks, in descending order;
    # * slot 3 is the VoI exploration arm (the only slot whose ``optimistic``
    #   score may exceed its own ``a_h1``);
    # * slot 4 is the third exploitation slot (``SLOT4_AGENT_BOOST == 0.0``, so
    #   its ``agent_promoted`` score *is* its ``a_h1``);
    # * slot 5 is the control *arm*, not a ``Δ_agent == 0`` filter — banning
    #   positively-boosted candidates would boycott the lab's own proven
    #   substitutions (the FLAW-2 regression this design fixes).
    assert slots[0]["a_h1"] >= slots[1]["a_h1"]
    assert [s["slot_type"] for s in slots] == [
        "Dual-Surrogate UCB",
        "Dual-Surrogate UCB",
        "Max Joint Uncertainty",
        "Agent Hypothesis Test",
        "Skeptic Falsification Probe",
    ]
    assert slots[2]["optimistic"] >= slots[2]["a_h1"] - 1e-9
    # With the promotion bonus retired, slot 4's promoted score is its own a_h1.
    if "agent_promoted" in slots[3]:
        assert slots[3]["agent_promoted"] == pytest.approx(slots[3]["a_h1"], abs=1e-6)

    # The batch design and the council debate are in the ledger, not only in
    # memory: a restart replays both, and the phase is re-derived from the round
    # number instead of being left at the ``empty_state`` default.
    assert lab.ledger.dataframe("round_slots").shape[0] == state["current_round"]
    assert lab.ledger.dataframe("round_council").shape[0] == state["current_round"]
    restarted = Campaign(
        lab.private, lab.normalizer, lab.protocol, {}, ledger_path=lab.ledger.path
    )
    assert restarted.state["current_round"] == state["current_round"]
    assert restarted.state["phase"] == state["phase"]
    assert [s["candidate_id"] for s in restarted.state["batch_slots"]] == [
        s["candidate_id"] for s in slots
    ]
    assert restarted.state["agent_council"], "council memos must survive a restart"
    assert [m["role"] for m in restarted.state["agent_council"]] == [
        m["role"] for m in state["agent_council"]
    ]
    assert restarted.state["voi_diagnostics"]["voi"] == pytest.approx(
        state["voi_diagnostics"]["voi"], abs=1e-9
    )


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"{path} not generated; run the corresponding script first")


def test_api_state_and_benchmark_match_the_frozen_artifacts():
    """``/api/state`` and ``/api/benchmark`` must agree with the run artifacts."""
    from app.server import app

    _require(SETTINGS.ledger_path)
    _require(SETTINGS.benchmark_path)

    with TestClient(app) as client:
        state = client.get("/api/state").json()
        contract = json.load(open("api_contract.json"))
        # ``api_contract.json`` is now generated from four endpoints, so it is a
        # *bundle* (``state``/``benchmark``/``cross_target``/``attribution`` plus
        # ``_meta``) rather than a bare state document. The state contract still
        # has to be a superset of what ``/api/state`` serves — that direction is
        # what keeps the frontend from referencing a field the backend lacks.
        assert set(contract["state"]) <= set(state)
        assert "voi_diagnostics" in state
        # Every endpoint the frontend codes against is in the bundle, so a
        # removed endpoint is a contract change rather than a silent 404.
        for key in ("state", "benchmark", "cross_target", "attribution"):
            assert key in contract, key
        assert contract["_meta"]["generated_by"] == "scripts/generate_api_contract.py"

        kpis = state["kpis"]
        assert kpis["query_budget_spent"] == kpis["total_budget"] == SETTINGS.total_budget
        assert state["current_round"] == SETTINGS.rounds == 6
        assert state["phase"] == "Self-Accelerating Payoff"

        diagnostics = state["voi_diagnostics"]
        for key in ("voi", "horizon", "sar_maturity", "exploit_weight",
                    "exploration_weight", "slot_split", "position_tolerance_effects"):
            assert key in diagnostics, key
        assert diagnostics["slot_split"] == "3-exploit / 2-experimental"

        # The council debate and the batch design must survive a restart: both
        # were previously in-memory only, so ``/api/state`` served an empty
        # ``agent_council`` and empty ``batch_slots`` for a completed campaign.
        assert state["batch_slots"], "a six-round campaign must expose its last batch"
        assert state["agent_council"], "the council memos must be on the record"
        for slot in state["batch_slots"]:
            for key in ("a_null", "a_h1", "delta_agent", "voi", "horizon", "sar_maturity"):
                assert key in slot, key
            assert slot["a_h1"] == pytest.approx(slot["a_null"] + slot["delta_agent"], abs=1e-3)
        # The final batch is exploitation-only: VoI collapses to 0 exactly when
        # no future query remains.
        assert diagnostics["voi"] == pytest.approx(0.0, abs=1e-9)
        assert diagnostics["exploit_weight"] == pytest.approx(1.0, abs=1e-9)
        assert diagnostics["position_tolerance_effects"], (
            "six rounds must produce at least one position-level SAR effect"
        )

        # The live campaign's Round-0 batch must be a *benchmark* seed, otherwise
        # ``run_campaign.py``'s same-seed comparison row is meaningless. Both sides
        # seed from ``np.random.default_rng(seed)`` for the same draw, so the live
        # ledger has to match some ``matched_initial_seeds`` entry exactly.
        from scripts.run_campaign import live_seed_index
        from app.server import current

        matched_seed = live_seed_index(current(), client.get("/api/benchmark").json())
        assert matched_seed is not None, (
            "live Round-0 batch matches no benchmark seed; the same-seed report row "
            "would silently fall back to the seed median"
        )

        benchmark = client.get("/api/benchmark").json()
        for key in ("curves", "aubqc", "normalized_aubqc", "top5_panel_curves",
                    "top5_panel_aubqc", "spearman_rho_curves", "top_decile_hit_rate",
                    "m5_warm_start", "paired_t_tests"):
            assert key in benchmark, key
        assert benchmark["seeds"] == 100
        assert set(benchmark["aubqc"]) == set(METHODS)
        assert [row["query_count"] for row in benchmark["curves"]] == list(range(5, 36, 5))

        warm = benchmark["m5_warm_start"]
        for key in ("definition", "curves", "top5_panel_curves", "summary", "initial_ids"):
            assert key in warm, key
        # The cold-start ablation is normalised against the matched random floor,
        # so its summary must be a genuine fraction of headroom, not the shrunk
        # artefact that normalising against its own stronger start produced.
        assert warm["summary"]["normalized_aubqc"] > 0.0
        assert warm["summary"]["normalized_aubqc"] > benchmark["normalized_aubqc"]["m5_hybrid"]

        paired = benchmark["paired_t_tests"]
        for key in ("m5_hybrid_minus_m3_ucb", "m5_hybrid_minus_m2_greedy"):
            assert key in paired, key
            assert set(paired[key]["metrics"]) == set(PAIRED_METRICS)
            for block in paired[key]["metrics"].values():
                assert np.isfinite(block["mean_diff"])
                assert block["n"] == 100

        # The served file is named ``benchmark_10seeds.json`` but holds a 100-seed
        # run. The endpoint must publish that honestly, so the frontend labels the
        # hero chart from data rather than from a filename it happens to read.
        source = benchmark["source"]
        assert source["seeds"] == 100
        assert source["label"] == "100-seed benchmark"
        assert source["filename_is_historical"] is True


#: Provenance fields the showcase renders beside ``a_null``. Adding one here is
#: how the contract stays truthful about *where the objective comes from*.
VOI_PROVENANCE_FIELDS: tuple[str, ...] = (
    "eta_sar",
    "sar_maturity_target",
    "slot_graduation",
    "steers_null_objective",
    "objective",
)


def test_state_exposes_objective_provenance():
    """``voi_diagnostics`` must say out loud that the anneal does not steer ``a_null``.

    ``a_null`` shipped as ``a_base`` at ``e567cd4``. A UI that plots
    ``exploit_weight`` next to the batch therefore has to be told, in the payload,
    that the weight is a diagnostic. ``steers_null_objective`` is that statement;
    without it the chart invites exactly the wrong reading.
    """
    from app.server import app

    _require(SETTINGS.ledger_path)
    _require(SETTINGS.benchmark_path)
    with TestClient(app) as client:
        _assert_provenance(client)


def _assert_provenance(client) -> None:
    diagnostics = client.get("/api/state").json()["voi_diagnostics"]
    for field in VOI_PROVENANCE_FIELDS:
        assert field in diagnostics, field
    assert diagnostics["steers_null_objective"] is False
    assert diagnostics["objective"] == "a_null := a_base"
    # The coefficients are real numbers on a live campaign, not placeholders: a
    # silent ``None`` here would let the badge render "n/a" forever.
    assert diagnostics["eta_sar"] is not None and diagnostics["eta_sar"] > 0.0
    assert diagnostics["sar_maturity_target"] is not None
    assert diagnostics["sar_maturity_target"] > 0.0
    assert isinstance(diagnostics["slot_graduation"], bool)

    # Every checkpoint and every method, so the hero chart has no gaps.
    curves = client.get("/api/state").json()["benchmark_curves"]
    assert [row["query_count"] for row in curves] == list(range(5, 36, 5))
    for row in curves:
        for method in METHODS:
            assert row.get(method) is not None, (row["query_count"], method)


def test_attribution_endpoint_serves_the_mechanism_ablations():
    """``GET /api/attribution`` must serve every variant the report discusses.

    The load-bearing rows here are the *negative* ones: a variant that ties
    production is evidence the mechanism is not earning its place. A UI that only
    renders improvements would make this artifact unreadable, so the endpoint's
    job is to hand over all of them, unscored and unfiltered.
    """
    from app.server import app

    with TestClient(app) as client:
        payload = client.get("/api/attribution").json()
    if not payload.get("available"):
        pytest.skip(payload.get("detail", "attribution artifact not generated"))

    assert payload["cells"], "an available artifact must carry at least one cell"
    for cell in payload["cells"]:
        assert cell["cell"]
        rows = cell["rows"]
        # Production is always present, keyed by the stringified None.
        assert "None" in rows
        # Every non-production row must come with its paired comparison, so the
        # table cannot show a variant without saying whether the gap is real.
        for name in rows:
            if name == "None":
                continue
            assert name in cell["production_minus"], name
            comparison = cell["production_minus"][name]
            assert "aubqc" in comparison and "rho" in comparison
        # Baselines are re-run on the matched seeds, never borrowed from another
        # artifact — that is what makes each delta a paired difference.
        for baseline in ("m2_greedy", "m3_ucb", "m4_fixed_prior"):
            assert baseline in rows, (cell["cell"], baseline)


def test_api_contract_is_generated_from_the_backend():
    """The committed ``api_contract.json`` must not drift from the live payloads.

    The file the frontend codes against used to carry handwritten numbers — four
    checkpoints instead of seven, three methods instead of five, and a
    ``best_qualified_binding`` no artifact ever produced. ``--check`` regenerates
    into memory and compares, so a stale contract fails here instead of being
    discovered by whoever builds the UI.
    """
    from scripts.generate_api_contract import main as contract_main

    assert contract_main(["--check"]) == 0, (
        "api_contract.json is stale; run `python scripts/generate_api_contract.py`"
    )


def test_cross_target_exposes_the_learnability_gate():
    """The gate must be reachable, and must never contain an M5 number.

    The gate's scientific value depends on having been run **without** M5: it is
    what makes the holdout choice independent of the thing being measured. If a
    future refactor starts writing M5 fields into it, the selection stops being
    blind and every claim resting on it becomes circular. This test is that guard.
    """
    from app.server import app

    if not (SETTINGS.root / "results/learnability_gate.json").exists():
        pytest.skip("learnability gate not generated")
    with TestClient(app) as client:
        payload = client.get("/api/cross-target").json()
    gate = payload["learnability_gate"]
    assert gate["available"] is True
    assert gate["rows"], "gate must carry its measured rows"
    assert gate.get("rho_gate") == 0.15
    for row in gate["rows"]:
        assert not any("m5" in str(key).lower() for key in row), row
        if row.get("skipped"):
            # A skipped target never ran a baseline, so it carries no rho. It must
            # say *why* it was skipped rather than silently reporting a zero.
            assert row["skipped"], row
            continue
        assert "M2_rho" in row and "M3_rho" in row, row
        best = max(v for v in (row["M2_rho"], row["M3_rho"]) if v is not None)
        assert row["passes_gate"] == (best >= gate["rho_gate"]), row
    # The partition must account for every CSV in the corpus.
    assert gate["partition"]["directories_enumerated"] == 17
    # ``1n8z`` is a reference row, never a member of the dynamic-sized matrix:
    # its 35-query protocol is not comparable row-for-row with the others.
    assert "1n8z" not in {t["dataset"] for t in payload["targets"]}
    assert [r["dataset"] for r in payload["reference_targets"]] == ["1n8z"]
