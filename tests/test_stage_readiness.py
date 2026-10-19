"""End-to-end stage-readiness suite for the Seed-21 demo.

Everything here is about what an operator can *do on stage*, so each test drives
the real FastAPI app rather than a helper:

* the round timeline can be inspected at every ``k`` without touching the
  database (:func:`test_state_at_every_round_matches_the_contract`);
* the hypothesis audit shows its claim and its graded pair for all six rounds;
* the golden ledger can be rewound to R5 and restored to R6 on demand
  (:func:`test_rewind_to_round_five_then_restore_golden`);
* Round 6 can be re-run live from the rewound state, 30 -> 35 queries;
* ``/api/benchmark`` carries every number the report quotes.

Isolation: the fixtures copy ``results/golden_seed21.sqlite3`` into a temporary
directory and point ``SETTINGS.ledger_path`` / ``SETTINGS.golden_ledger_path`` at
it, so a bug in rewind can never cost the developer the real campaign.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import SETTINGS
from core.baselines import LIVE_LABEL, METHODS, ZERO_SHOT_REFERENCES
from core.ledger import ledger_version

#: The contract keys every round state must carry.
CONTRACT_KEYS = set(json.load(open("api_contract.json")))

#: Rounds in the verified Seed-21 campaign.
SEED_ROUND = 0
LAST_ROUND = 6

#: What the report quotes for the frozen run.
GOLDEN_BEST = 8.58
GOLDEN_AUBQC = 1.963
GOLDEN_QUERIES = 35


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture()
def stage(tmp_path, monkeypatch):
    """A throwaway ledger pair seeded from the golden file.

    Returns ``(client, ledger_path, golden_path)``. The paths are threaded into
    :func:`app.server.initialize` rather than patched onto ``SETTINGS`` — the
    settings object is frozen, and more importantly a rewind bug in a *test* must
    not be able to shorten the real campaign. Both files are copies; the real
    ``results/`` pair is only ever read here.
    """
    golden_source = Path(SETTINGS.golden_ledger_path)
    if not golden_source.exists():
        pytest.skip(f"{golden_source} not generated; run the Seed-21 campaign first")
    if not Path(SETTINGS.benchmark_path).exists():
        pytest.skip(f"{SETTINGS.benchmark_path} not generated; run scripts/run_benchmark.py")

    ledger = tmp_path / "campaign.sqlite3"
    golden = tmp_path / "golden_seed21.sqlite3"
    ledger.write_bytes(golden_source.read_bytes())
    golden.write_bytes(golden_source.read_bytes())

    import app.server as server

    real_initialize = server.initialize
    monkeypatch.setattr(
        server,
        "initialize",
        lambda *_a, **_kw: real_initialize(
            ledger_path=ledger, golden_ledger_path=golden
        ),
    )
    monkeypatch.setattr(server, "campaign", None)
    server.invalidate_benchmark_cache()

    with TestClient(server.app) as client:
        yield client, ledger, golden

    server.invalidate_benchmark_cache()


def _round_numbers(client) -> list[int]:
    return [entry["round"] for entry in client.get("/api/rounds").json()["rounds"]]


# ── (a) round-by-round time travel ───────────────────────────────────────────
def test_state_at_every_round_matches_the_contract(stage):
    """``GET /api/state?round=k`` must serve a complete, valid state for all k."""
    client, ledger, _ = stage
    version_before = ledger_version(ledger)

    versions = []
    for round_number in range(SEED_ROUND, LAST_ROUND + 1):
        response = client.get("/api/state", params={"round": round_number})
        assert response.status_code == 200, response.text
        state = response.json()

        # The contract is intact at every round, not only the latest one.
        assert CONTRACT_KEYS <= set(state), sorted(CONTRACT_KEYS - set(state))
        assert state["round_number"] == state["current_round"] == round_number
        assert state["phase"] == (
            "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff"
        )
        kpis = state["kpis"]
        assert set(kpis) == set(json.load(open("api_contract.json"))["kpis"])
        assert kpis["query_budget_spent"] == SETTINGS.initial_queries + round_number * SETTINGS.batch_size
        assert kpis["total_budget"] == SETTINGS.total_budget
        assert 0.0 < kpis["best_qualified_binding"] <= 10.0
        assert kpis["hypotheses_supported"] + kpis["hypotheses_falsified"] <= round_number

        # The VoI/SAR diagnostics come from *that round's* persisted batch
        # design, so the exploration -> exploitation anneal is visible.
        diagnostics = state["voi_diagnostics"]
        for key in ("voi", "horizon", "sar_maturity", "exploit_weight",
                    "exploration_weight", "slot_split", "position_tolerance_effects"):
            assert key in diagnostics, key
        assert diagnostics["round_number"] == round_number
        assert diagnostics["slot_split"] == "3-exploit / 2-experimental"
        assert 0.0 <= diagnostics["voi"] <= 1.0

        # The seed round has no council and no designed batch yet; every later
        # round must show both, or the UI would render empty panels.
        if round_number == 0:
            assert state["agent_council"] == []
            assert state["batch_slots"] == []
        else:
            assert state["agent_council"], f"round {round_number} lost its council memos"
            assert state["batch_slots"], f"round {round_number} lost its batch design"
            for mem in state["agent_council"]:
                for key in ("role", "memo", "tools_invoked"):
                    assert key in mem, key
            assert len(state["batch_slots"]) == SETTINGS.batch_size
            for slot in state["batch_slots"]:
                for key in ("slot_id", "slot_type", "candidate_id", "sequence",
                            "mu_bind", "sigma_bind", "risk_score", "risk_passed",
                            "a_null", "a_h1", "delta_agent", "sigma_joint", "voi"):
                    assert key in slot, (round_number, key)
                assert slot["a_h1"] == pytest.approx(slot["a_null"] + slot["delta_agent"], abs=1e-3)
                assert len(slot["sequence"]) == 13
            assert [slot["slot_id"] for slot in state["batch_slots"]] == [1, 2, 3, 4, 5]
            assert len({slot["candidate_id"] for slot in state["batch_slots"]}) == SETTINGS.batch_size

        # The ribbon and the declared priors are the round's own provenance.
        assert [item["position"] for item in state["hcdr3_ribbon"]] == [1, 3, 6, 8, 11, 12, 13]
        for item in state["hcdr3_ribbon"]:
            assert item["status"] in {
                "anchor_conserved", "low_risk_swap", "epistasis_alert", "neutral"
            }
        assert isinstance(state["declared_priors"], list)
        if round_number > 0:
            assert state["declared_priors"], f"round {round_number} declared no priors"

        # The Pareto scatter is the revealed ledger, in query order.
        assert len(state["pareto_data"]) == kpis["query_budget_spent"]
        assert {row["status"] for row in state["pareto_data"]} <= {
            "wild_type", "pareto_frontier",
            *{f"queried_round_{r}" for r in range(LAST_ROUND + 1)},
        }

        versions.append(diagnostics["voi"])

        # The clickable tool pills must resolve for every round.
        outputs = state["tool_outputs"]
        assert set(outputs) == {
            "scan_positional_sar", "detect_epistasis_clashes",
            "inspect_joint_uncertainty_hotspots",
            "profile_developability_liabilities",
            "audit_past_hypotheses", "compute_cost_and_effort_saved",
        }
        for name, payload in outputs.items():
            assert payload.get("tool") == name, (round_number, name)
        if round_number > 0:
            # A round's debate ran on the ledger *before* its batch was measured,
            # so its tool tables are non-empty from R1 onward.
            assert outputs["scan_positional_sar"]["rows"], round_number
            assert outputs["profile_developability_liabilities"]["rows"], round_number
            assert outputs["inspect_joint_uncertainty_hotspots"]["rows"], round_number

    # VoI anneals 0.578 -> 0.000 across the six rounds and never rises. The
    # round-0 placeholder (0.0) is excluded: no batch has been designed yet.
    designed = versions[1:]
    assert designed[0] == pytest.approx(0.578, abs=5e-3)
    assert designed[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(later <= earlier + 1e-9 for earlier, later in zip(designed, designed[1:])), designed

    # Inspection is read-only: the ledger's content fingerprint is untouched.
    assert ledger_version(ledger) == version_before


def test_rounds_endpoint_enumerates_the_timeline(stage):
    """``GET /api/rounds`` drives ``[Seed] -> [R1] -> ... -> [R6]``."""
    client, _, _ = stage
    payload = client.get("/api/rounds").json()
    assert payload["seed"] == 21
    assert payload["current_round"] == LAST_ROUND
    assert payload["total_rounds"] == LAST_ROUND
    assert payload["total_budget"] == GOLDEN_QUERIES

    entries = payload["rounds"]
    assert [entry["round"] for entry in entries] == list(range(SEED_ROUND, LAST_ROUND + 1))
    assert [entry["label"] for entry in entries] == ["Seed", "R1", "R2", "R3", "R4", "R5", "R6"]
    assert [entry["query_count"] for entry in entries] == [5, 10, 15, 20, 25, 30, 35]
    for entry in entries:
        for key in ("phase", "voi", "sar_maturity", "exploit_weight",
                    "best_qualified_binding", "slots", "council_seats"):
            assert key in entry, key
    # The seed round is the q=5 checkpoint every matched baseline shares.
    assert entries[0]["query_count"] == SETTINGS.initial_queries
    assert entries[0]["slots"] == 0
    for entry in entries[1:]:
        assert entry["slots"] == SETTINGS.batch_size
        assert entry["council_seats"] >= 1
    assert entries[-1]["best_qualified_binding"] == pytest.approx(GOLDEN_BEST, abs=0.01)


def test_state_round_is_bounded_by_what_has_run(stage):
    """A round that has not happened is a conflict, not an empty state."""
    client, _, _ = stage
    assert client.get("/api/state", params={"round": LAST_ROUND + 1}).status_code in (409, 422)
    assert client.get("/api/state", params={"round": -1}).status_code == 422


# ── (b) hypothesis audit ─────────────────────────────────────────────────────
def test_hypotheses_cover_every_round_with_a_gradable_overlap(stage):
    """Every round's claim is graded on the ids it actually named.

    ``claim_about`` is what the hypothesis statement asks about; ``graded_on`` is
    what the paired audit compared. ``overlap == "FULL"`` iff the former is a
    subset of the latter — an empty intersection is a verdict that is arithmetically
    correct and scientifically meaningless.
    """
    client, _, _ = stage
    items = client.get("/api/hypotheses").json()["items"]
    assert len(items) == LAST_ROUND
    assert [item["round_number"] for item in items] == list(range(1, LAST_ROUND + 1))
    assert {item["verdict"] for item in items} <= {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}

    for item in items:
        assert item["hypothesis_id"] == f"HYP-R{item['round_number']}-01"
        assert item["statement"], item["round_number"]
        evidence = item["evidence"]
        claim = evidence["claim_about"]
        graded = evidence["graded_on"]
        assert len(claim) == 2, (item["round_number"], claim)
        assert claim != graded or set(claim) == set(graded)
        overlap = "FULL" if set(claim) <= set(graded) else (
            "PARTIAL" if set(claim) & set(graded) else "NONE"
        )
        assert overlap == "FULL", (item["round_number"], claim, graded)
        # The comparison the verdict used, and the numbers behind it.
        assert evidence["compared"] in {"claimed_pair", "arm_vs_control"}
        assert evidence["h1_binding"] is not None
        assert evidence["h0_binding"] is not None
        # All three are rounded to 3 dp independently, so the difference of two
        # rounded values can be 1e-3 away from the rounded difference.
        assert evidence["paired_lift"] == pytest.approx(
            evidence["h1_binding"] - evidence["h0_binding"], abs=2e-3
        )

    # Seed 21 is 4 supported / 2 falsified, which is what the KPI panel shows.
    assert sum(item["verdict"] == "SUPPORTED" for item in items) == 4
    assert sum(item["verdict"] == "FALSIFIED" for item in items) == 2


def test_live_state_equals_the_reconstructed_current_round(stage):
    """``/api/state`` must be ``/api/state?round=current``, field for field.

    The live view is assembled by ``_refresh_state`` during a round; the
    historical view is rebuilt from the ledger. They are two code paths producing
    one contract, so any field one of them forgets (``pareto_data`` and
    ``hcdr3_ribbon`` were both empty after a restart before this guard) has to
    fail here rather than on stage.
    """
    client, _, _ = stage
    live = client.get("/api/state").json()
    rebuilt = client.get("/api/state", params={"round": live["current_round"]}).json()
    for key in ("kpis", "phase", "agent_council", "batch_slots", "pareto_data",
                "hcdr3_ribbon", "voi_diagnostics"):
        assert live[key] == rebuilt[key], key
    assert live["pareto_data"], "a completed campaign must expose its revealed scatter"
    assert len(live["pareto_data"]) == live["kpis"]["query_budget_spent"]


def test_hypotheses_are_identical_at_every_historical_round(stage):
    """Rounding back through the timeline must not rewrite history."""
    client, _, _ = stage
    for round_number in range(1, LAST_ROUND + 1):
        state = client.get("/api/state", params={"round": round_number}).json()
        assert [item["round_number"] for item in state["hypotheses"]] == list(range(1, round_number + 1))
        latest = client.get("/api/hypotheses").json()["items"][round_number - 1]
        assert state["hypotheses"][-1]["verdict"] == latest["verdict"]


# ── (c) rewind / restore ─────────────────────────────────────────────────────
def test_rewind_to_round_five_then_restore_golden(stage):
    """``[Rewind to R5]`` lands on 30 queries; ``[Restore Golden]`` returns 35."""
    client, ledger, golden = stage
    assert client.get("/api/state").json()["current_round"] == LAST_ROUND

    response = client.post("/api/rewind", json={"round": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rewound"] is True
    assert body["round"] == 5
    assert body["queries_spent"] == 30
    assert body["verified"] is True
    assert body["current_round"] == 5
    assert body["entries_dropped"] == 13  # one council, one decision, one batch, ten reveals
    # Round 6 is genuinely gone, not merely hidden from one endpoint.
    assert body["state"]["kpis"]["query_budget_spent"] == 30
    assert [entry["round"] for entry in client.get("/api/rounds").json()["rounds"]] == list(range(0, 6))
    assert client.get("/api/state", params={"round": LAST_ROUND}).status_code in (409, 422)

    state = client.get("/api/state").json()
    assert state["current_round"] == 5
    assert state["kpis"]["query_budget_spent"] == 30
    assert state["kpis"]["best_qualified_binding"] == pytest.approx(8.52, abs=0.01)
    assert state["voi_diagnostics"]["voi"] == pytest.approx(0.07, abs=1e-6)
    assert len(client.get("/api/hypotheses").json()["items"]) == 5
    assert client.get("/api/ledger").json()["verified"] is True

    # The golden file is the read-only reference; rewinding must not touch it.
    assert ledger_version(golden) == (88, 88)

    # [Restore Golden] puts the verified six-round run back.
    response = client.post("/api/restore-golden")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["restored"] is True
    assert body["current_round"] == LAST_ROUND
    assert body["entries"] == 88
    assert body["verified"] is True

    state = client.get("/api/state").json()
    assert state["current_round"] == LAST_ROUND
    kpis = state["kpis"]
    assert kpis["query_budget_spent"] == kpis["total_budget"] == GOLDEN_QUERIES
    assert kpis["best_qualified_binding"] == pytest.approx(GOLDEN_BEST, abs=1e-9)
    assert kpis["delta_vs_wt"] == pytest.approx(0.872, abs=1e-3)
    assert kpis["hypotheses_supported"] == 4
    assert kpis["hypotheses_falsified"] == 2
    assert ledger_version(ledger) == ledger_version(golden) == (88, 88)
    assert client.get("/api/ledger").json()["verified"] is True
    # And the exact round-6 view is reconstructible again.
    assert client.get("/api/state", params={"round": LAST_ROUND}).json()["kpis"] == kpis


def test_rewind_restores_from_golden_when_the_live_ledger_is_short(stage):
    """Rewinding *forward* after a botched round must fall back to the golden copy."""
    client, ledger, golden = stage
    short = client.post("/api/rewind", json={"round": 2}).json()
    assert short["queries_spent"] == 15 and short["current_round"] == 2

    # The live ledger now only holds R2; asking for R5 requires the golden file.
    body = client.post("/api/rewind", json={"round": 5}).json()
    assert body["restored_from_golden"] is True
    assert body["current_round"] == 5
    assert body["queries_spent"] == 30
    assert body["verified"] is True
    assert client.get("/api/ledger").json()["verified"] is True
    assert ledger_version(golden) == (88, 88)


def test_rewind_rejects_out_of_range_rounds(stage):
    """The body is bounded, so a typo cannot truncate more than the demo allows."""
    client, _, _ = stage
    assert client.post("/api/rewind", json={"round": -1}).status_code == 422
    assert client.post("/api/rewind", json={"round": LAST_ROUND + 1}).status_code == 422
    assert client.post("/api/rewind", json={}).status_code == 422


# ── (c, live leg) Round 6 over the rewound state ─────────────────────────────
@pytest.mark.asyncio
async def test_round_six_replays_from_the_rewound_ledger(stage):
    """/api/round/run is accepted at R5, and the round closes at 35 queries.

    The council runs on the deterministic offline fallback (``tests/conftest.py``
    strips credentials), which is what makes this a *loop* test: surrogates ->
    selector -> oracles -> ledger -> state, starting from Round 5 rather than
    Round 0. The ledger is restored afterwards, so this cannot corrupt the
    fixture for later tests.
    """
    client, _, _ = stage
    rewound = client.post("/api/rewind", json={"round": 5}).json()
    assert rewound["current_round"] == 5

    # The UI's pre-flight gate must agree that a round is still runnable.
    assert client.get("/api/state").json()["current_round"] < SETTINGS.rounds
    assert client.post("/api/round/run").json()["accepted"] is True

    import app.server as server

    lab = server.current()
    await lab.execute_round()
    assert lab.current_round() == LAST_ROUND
    assert lab.state["kpis"]["query_budget_spent"] == GOLDEN_QUERIES
    assert len(lab.state["batch_slots"]) == SETTINGS.batch_size
    assert lab.ledger.verify() is True
    assert len(lab.hypotheses) == LAST_ROUND

    # A completed campaign refuses to start another round.
    assert client.post("/api/round/run").status_code == 409

    # ... and one call puts the verified run back.
    assert client.post("/api/restore-golden").json()["current_round"] == LAST_ROUND
    assert client.get("/api/state").json()["kpis"]["best_qualified_binding"] == pytest.approx(
        GOLDEN_BEST, abs=1e-9
    )


# ── (d) benchmark panel ─────────────────────────────────────────────────────
def test_benchmark_serves_every_reference_the_report_quotes(stage):
    """100-seed metrics, the warm start, paired tests, and both seed-21 panels."""
    client, _, _ = stage
    benchmark = client.get("/api/benchmark").json()

    # The frozen 100-seed artifact.
    assert benchmark["seeds"] == 100
    for key in ("curves", "aubqc", "normalized_curves", "normalized_aubqc",
                "top5_panel_curves", "top5_panel_aubqc", "spearman_rho_curves",
                "top_decile_hit_rate", "matched_initial_seeds",
                "m5_warm_start", "paired_t_tests"):
        assert key in benchmark, key
    assert set(benchmark["aubqc"]) == set(METHODS)
    assert [row["query_count"] for row in benchmark["curves"]] == list(range(5, 36, 5))
    for method in METHODS:
        assert all(row[method] is not None for row in benchmark["curves"]), method

    warm = benchmark["m5_warm_start"]
    for key in ("definition", "curves", "top5_panel_curves", "summary", "initial_ids"):
        assert key in warm, key
    assert warm["summary"]["normalized_aubqc"] > 0.0

    paired = benchmark["paired_t_tests"]
    assert "m5_hybrid_minus_m3_ucb" in paired
    assert "m5_hybrid_minus_m2_greedy" in paired
    for block in paired.values():
        assert block["performer"] and block["baseline"]
        assert block["metrics"]

    # The published zero-shot panel, byte-for-byte what the report quotes.
    assert benchmark["zero_shot_references"] == ZERO_SHOT_REFERENCES == {
        "alphafold3": -0.16, "proteinmpnn": -0.17, "esm2": -0.13,
    }

    # The same-seed panel: live M5 against the five matched baselines on Seed 21.
    same = benchmark["same_seed_comparison"]
    assert same["seed"] == 21
    assert set(same["methods"]) == set(METHODS)
    assert len(same["initial_ids"]) == SETTINGS.initial_queries
    for method in METHODS:
        summary = same["summary"][method]
        assert summary["q"] == 5
        assert summary["q5"] == pytest.approx(7.706, abs=1e-3)  # every row shares the seed
        assert 0.0 < summary["q35"] <= 10.0
        assert summary["aubqc"] > 0.0
    live = same["summary"][LIVE_LABEL]
    assert live["q5"] == pytest.approx(7.706, abs=2e-3)
    # The live curve is recorded to 3 dp (8.578); the KPI panel reports 2 dp
    # (8.58). They are the same measurement at different resolutions.
    assert live["q35"] == pytest.approx(GOLDEN_BEST, abs=5e-3)
    assert live["aubqc"] == pytest.approx(GOLDEN_AUBQC, abs=5e-3)

    # The trajectory is one plottable row per checkpoint, with live on the same axis.
    assert [row["query_count"] for row in same["trajectory"]] == list(range(5, 36, 5))
    for row in same["trajectory"]:
        assert set(row) == {"query_count", *METHODS, LIVE_LABEL}
        for method in METHODS:
            assert row[method] is not None
    assert same["trajectory"][-1][LIVE_LABEL] == pytest.approx(GOLDEN_BEST, abs=5e-3)
    # Monotone discovery: a method cannot un-find a binder. ``None`` is allowed on
    # the live row past its spend, but here the campaign is complete.
    for method in METHODS:
        values = [row[method] for row in same["trajectory"]]
        assert all(later >= earlier - 1e-9 for earlier, later in zip(values, values[1:])), method

    # The live row is reported separately and never mixed into the seed medians.
    assert LIVE_LABEL not in benchmark["aubqc"]
    assert LIVE_LABEL not in benchmark["curves"][0]
    assert same["live"]["label"] == LIVE_LABEL
    assert same["live"]["best_baseline"] in METHODS
    assert same["live"]["aubqc_gap_vs_best_baseline"] == pytest.approx(
        live["aubqc"] - same["live"]["best_baseline_aubqc"], abs=1e-4
    )


def test_benchmark_cache_follows_a_rewind(stage):
    """The same-seed live row must not survive a rewind as a stale number."""
    client, _, _ = stage
    before = client.get("/api/benchmark").json()["same_seed_comparison"]["live"]
    assert before["q35"] == pytest.approx(GOLDEN_BEST, abs=5e-3)

    client.post("/api/rewind", json={"round": 5})
    after = client.get("/api/benchmark").json()["same_seed_comparison"]["live"]
    # At R5 the live campaign has spent 30 queries, so its q=35 checkpoint carries
    # the last observed value forward (5 points below the completed run's).
    assert after["q35"] == pytest.approx(8.52, abs=0.01)
    assert after["q35"] < before["q35"]

    client.post("/api/restore-golden")
    restored = client.get("/api/benchmark").json()["same_seed_comparison"]["live"]
    assert restored["q35"] == pytest.approx(GOLDEN_BEST, abs=5e-3)
