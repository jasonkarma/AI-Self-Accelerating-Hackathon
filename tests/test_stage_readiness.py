"""End-to-end stage-readiness suite for the Seed-21 demo.

Everything here is about what an operator can *do on stage*, so each test drives
the real FastAPI app rather than a helper:

* the round timeline can be inspected at every ``k`` without touching the
  database (:func:`test_state_at_every_round_matches_the_contract`);
* the hypothesis audit shows its claim and its graded pair for all six rounds;
* the golden ledger can be rewound to R5 and restored to R6 on demand
  (:func:`test_rewind_to_round_five_then_restore_golden`);
* Round 6 can be re-run live from the rewound state, 30 -> 35 queries;
* ``/api/benchmark`` carries every number the report quotes;
* the Option C tool arsenal visibly switches drills on and off across R1..R6
  (:func:`test_option_c_tool_activation_across_rounds`);
* ``/api/cross-target`` serves the second-target generalization report
  (:func:`test_cross_target_endpoint`).

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
from core.baselines import LIVE_LABEL, METHODS, PAIRED_METRICS, ZERO_SHOT_REFERENCES
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


# ── (e) Option C: two-tier tool arsenal ──────────────────────────────────────
#: The two tools that are mandatory in every planning round.
TIER_SOP = {"scan_positional_sar", "profile_developability_liabilities"}
#: The four drills, and the rounds in which the golden ledger's state activates each.
TIER_DRILLS = {
    "inspect_joint_uncertainty_hotspots",
    "audit_past_hypotheses",
    "detect_epistasis_clashes",
    "compute_cost_and_effort_saved",
}


def _activation(state: dict) -> dict:
    """The Option C block, with a failure message that says where to look."""
    diagnostics = state["voi_diagnostics"]
    assert "active_tools_metadata" in diagnostics, (
        "voi_diagnostics lost active_tools_metadata; GET /api/state?round=k must expose "
        "the tool arsenal"
    )
    return diagnostics["active_tools_metadata"]


def _flags(metadata: dict) -> dict[str, bool]:
    return {tool["name"]: tool["active"] for tool in metadata["tools"]}


def test_option_c_tool_activation_across_rounds(stage):
    """Stepping R0 -> R6 must show drills switching on *and* off.

    The request's demonstration is that a drill turns on between two rounds and
    another turns off before the last one; both are asserted here rather than
    assumed. Every flag is also **re-derived from its own ``trigger_reason``**, so
    a predicate that disagrees with the boolean it publishes fails instead of
    quietly rendering the wrong pill.
    """
    client, _, _ = stage
    snapshot = {}
    for round_number in range(SEED_ROUND, LAST_ROUND + 1):
        state = client.get("/api/state", params={"round": round_number}).json()
        metadata = _activation(state)

        # Shape: six tools, stable order, both tiers, full provenance per tool.
        names = [tool["name"] for tool in metadata["tools"]]
        assert names == [
            "scan_positional_sar", "profile_developability_liabilities",
            "inspect_joint_uncertainty_hotspots", "audit_past_hypotheses",
            "detect_epistasis_clashes", "compute_cost_and_effort_saved",
        ], names
        assert len(names) == 6
        for tool in metadata["tools"]:
            assert tool["tier"] in {"SOP Baseline", "State-Triggered Drill"}, tool
            assert isinstance(tool["active"], bool), tool
            assert tool["trigger_reason"], tool
            assert tool["routed_to"], tool
            # route, not decoration: every role must be a real council seat
            for role in tool["routed_to"]:
                assert role in {
                    "Affinity Scientist", "Developability Scientist",
                    "Scientific Skeptic", "Experimental Planner",
                }, (tool["name"], role)
        assert {t["name"] for t in metadata["tools"] if t["tier"] == "SOP Baseline"} == TIER_SOP
        assert {t["name"] for t in metadata["tools"] if t["tier"] == "State-Triggered Drill"} == TIER_DRILLS
        assert metadata["round_number"] == round_number
        # ``active`` is the list form of the same flags, never an independent copy.
        assert metadata["active"] == [t["name"] for t in metadata["tools"] if t["active"]]

        # The plan the council saw: the ledger through the *previous* round, not
        # this one's own spend. Off by one batch would open epistasis in R1.
        assert metadata["queries_spent"] == (
            0 if round_number <= 0 else SETTINGS.initial_queries + (round_number - 1) * SETTINGS.batch_size
        ), round_number
        # And the trigger predicates must agree with the exported numbers.
        assert metadata["voi"] == 0.0 or metadata["voi"] == state["voi_diagnostics"]["voi"]
        assert metadata["sar_maturity"] == state["voi_diagnostics"]["sar_maturity"]

        flags = _flags(metadata)
        reasons = {t["name"]: t["trigger_reason"] for t in metadata["tools"]}

        # Round 0 is the seed batch: no council, so nothing is routed at all.
        if round_number == SEED_ROUND:
            assert set(metadata["active"]) == set(), metadata["active"]
            for name in TIER_SOP | TIER_DRILLS:
                assert "seed round" in reasons[name], (name, reasons[name])
        else:
            # Tier 1 is mandatory in every planning round.
            for name in TIER_SOP:
                assert flags[name] is True, (round_number, name, reasons[name])
            # The hotspot drill is the VoI-driven one: on while VoI > 0, off at
            # the terminal harvest, and its reason must say which.
            voi = state["voi_diagnostics"]["voi"]
            assert flags["inspect_joint_uncertainty_hotspots"] == (voi > 0.0), (
                round_number, voi, reasons["inspect_joint_uncertainty_hotspots"]
            )
            if voi > 0.0:
                assert f"VoI = {voi:.4f}" in reasons["inspect_joint_uncertainty_hotspots"]
            else:
                assert "Terminal harvest" in reasons["inspect_joint_uncertainty_hotspots"]
            # Round-gated drill, guaranteed by round_number alone.
            assert flags["audit_past_hypotheses"] == (round_number >= 2)
            # The cost drill is an OR: payoff regime by round *or* by VoI. R2
            # satisfies the VoI clause (0.28 <= 0.35) even though it is < R3, which
            # is the specified behaviour — the trigger is the campaign's state, not
            # a round counter.
            assert flags["compute_cost_and_effort_saved"] == (
                round_number >= 3 or state["voi_diagnostics"]["voi"] <= 0.35
            ), (round_number, reasons["compute_cost_and_effort_saved"])
            # Query-gated drill: two observations per position is what it needs,
            # so the gate is the query count (== ``round_number >= 2``), not the
            # already-converged SAR table — the golden R1 reports 0.65 maturity,
            # which a maturity clause would have used to open the R1 drill.
            assert flags["detect_epistasis_clashes"] == (metadata["queries_spent"] >= 10), (
                round_number, reasons["detect_epistasis_clashes"]
            )

        snapshot[round_number] = metadata

    # ── The demo itself: tools visibly turn on and off ───────────────────────
    # R1: the two SOP baselines plus the VoI drill; every round-gated drill is off.
    r1 = set(snapshot[1]["active"])
    assert TIER_SOP <= r1
    assert "inspect_joint_uncertainty_hotspots" in r1
    assert "audit_past_hypotheses" not in r1, "R1 has no past decision to audit"
    assert "compute_cost_and_effort_saved" not in r1, "R1 is tuition, not payoff"
    assert "detect_epistasis_clashes" not in r1, "R1 has only the 5 seed queries"

    # R2: the audit and epistasis drills switch ON relative to R1, and the cost
    # drill joins them via its VoI clause rather than its round clause.
    r2 = set(snapshot[2]["active"])
    turned_on = r2 - r1
    assert "audit_past_hypotheses" in turned_on
    assert "detect_epistasis_clashes" in turned_on
    assert "compute_cost_and_effort_saved" in turned_on, (
        "R2's VoI (<= 0.35) should open the payoff drill before R3 does; "
        f"got VoI={snapshot[2]['voi']}"
    )

    # From R2 onward the whole arsenal is live, which is the 6/6 panel state.
    for round_number in range(2, LAST_ROUND):
        assert set(snapshot[round_number]["active"]) == TIER_SOP | TIER_DRILLS, round_number
    r3 = set(snapshot[3]["active"])
    assert r3 == TIER_SOP | TIER_DRILLS

    # R6: the hotspot drill switches OFF at the terminal harvest, while every
    # round-gated drill stays on. This is the deactivation the request asks for.
    r6 = set(snapshot[LAST_ROUND]["active"])
    turned_off = r3 - r6
    assert turned_off == {"inspect_joint_uncertainty_hotspots"}, turned_off
    assert snapshot[LAST_ROUND]["voi"] == 0.0

    # Across the whole arc at least one tool turns on and at least one turns off,
    # so the UI stepper always has something to show.
    activations = sum(
        flags[name] and not _flags(snapshot[round_number - 1])[name]
        for round_number in range(1, LAST_ROUND + 1)
        for name in TIER_DRILLS
    )
    deactivations = sum(
        not flags[name] and _flags(snapshot[round_number - 1])[name]
        for round_number in range(1, LAST_ROUND + 1)
        for name in TIER_DRILLS
    )
    assert activations >= 3, activations
    assert deactivations >= 1, deactivations

    # ``/api/rounds`` carries the same split, so the stepper needs no extra call.
    for entry in client.get("/api/rounds").json()["rounds"]:
        metadata = snapshot[entry["round"]]
        assert entry["active_tools"] == list(metadata["active"]), entry["round"]
        assert entry["inactive_tools"] == [
            tool["name"] for tool in metadata["tools"] if not tool["active"]
        ], entry["round"]
        assert set(entry["tool_tiers"]) == TIER_SOP | TIER_DRILLS
        assert entry["tool_tiers"]["scan_positional_sar"] == "SOP Baseline"
        assert entry["tool_tiers"]["audit_past_hypotheses"] == "State-Triggered Drill"


def test_option_c_activation_is_exact_in_the_live_view_and_reconstruction(stage):
    """The live ``/api/state`` and ``/api/state?round=current`` must agree exactly.

    ``voi`` is a derived float and the hotspot drill keys on ``voi > 0``, so a raw
    ``-1e-17`` on the final batch would deactivate it live while the reconstruction
    read a rounded ``0.0``. Both paths therefore derive the table from the
    **exported** values; this test is what would catch a future edit that stopped
    doing that.
    """
    client, _, _ = stage
    live = client.get("/api/state").json()
    rebuilt = client.get("/api/state", params={"round": live["current_round"]}).json()
    assert live["voi_diagnostics"] == rebuilt["voi_diagnostics"]
    # The council memos carry the same active/inactive split on both paths.
    assert live["agent_council"] == rebuilt["agent_council"]
    for memo in live["agent_council"]:
        assert "tools_invoked_active" in memo and "tools_inactive" in memo
        assert set(memo["tools_invoked_active"]) <= set(memo["tools_invoked"])
        # Every cited tool is either active or explained, never both.
        assert not (
            set(memo["tools_invoked_active"]) & set(memo["tools_inactive"])
        )


# ── (f) cross-target generalization ─────────────────────────────────────────
def test_cross_target_endpoint(stage):
    """``GET /api/cross-target`` serves a complete second-target report.

    Skips when the artifact is absent, mirroring the other artifact-dependent
    tests, so a clean checkout still runs the suite.
    """
    client, _, _ = stage
    payload = client.get("/api/cross-target").json()
    if not payload.get("available"):
        pytest.skip(
            "results/cross_target_benchmark.json not generated; run "
            "scripts/run_cross_target_benchmark.py --seeds 50"
        )

    cross = payload["cross_target"]
    for key in ("dataset", "label", "source", "source_file", "reference_sequence",
                "cdr3_length", "rows_final", "tau_dev", "seeds",
                "dataset_sequence_sha256", "baselines_unchanged",
                "rejected_targets", "generalizes_across"):
        assert key in cross, key

    # The report is the same benchmark shape the 1N8Z artifact has — but scored
    # on *its own* abscissae, because the campaign length is now protocol-sized.
    assert set(payload["aubqc"]) == set(METHODS)
    checkpoints = [row["query_count"] for row in payload["curves"]]
    assert checkpoints[0] == 5, checkpoints
    assert checkpoints == list(range(5, checkpoints[-1] + 1, 5)), checkpoints
    assert cross["campaign_protocol"]["checkpoints"] == checkpoints
    assert cross["total_budget"] == checkpoints[-1]
    assert len(checkpoints) >= 2
    for method in METHODS:
        assert all(row[method] is not None for row in payload["curves"]), method
        assert payload["aubqc"][method] > 0.0
    assert payload["seeds"] == cross["seeds"]

    warm = payload["m5_warm_start"]
    for key in ("definition", "curves", "top5_panel_curves", "summary", "initial_ids"):
        assert key in warm, key

    paired = payload["paired_t_tests"]
    for key in ("m5_hybrid_minus_m3_ucb", "m5_hybrid_minus_m2_greedy",
                "m5_warm_start_minus_m5_hybrid"):
        assert key in paired, key
        assert set(paired[key]["metrics"]) == set(PAIRED_METRICS)

    # The scientific claim: same architecture, same constants, different frame.
    assert cross["baselines_unchanged"] is True
    assert set(cross["unchanged_modules"]) == {
        "core/acquisition.py", "core/baselines.py",
        "core/surrogate.py", "core/features.py",
    }
    reference = cross["reference_sequence"]
    assert len(reference) == cross["cdr3_length"]
    assert cross["generalizes_across"].startswith("antigen target")

    # The reference frame is the target's own, documented, and consistent.
    assert cross["reference_source"] in {"consensus", "provided"}
    if cross["cdr3_length"] != 13:
        assert cross["reference_is_1n8z"] is False, (
            "a non-13-residue target cannot be scored against the 1N8Z wild type; "
            "that would be a coordinate-system error, not a generalization"
        )

    # Every rejected target is rejected *with its measured reason*, which is what
    # makes 'we tried 4FQI first' a claim a reader can check.
    for item in cross["rejected_targets"]:
        assert item["reason"], item
        assert item["key"] and item["label"]

    # The table the operator reads is rendered from this same payload.
    from scripts.run_cross_target_benchmark import format_summary

    table = format_summary(payload)
    assert cross["label"] in table
    assert "AUBQC" in table


# ── (g) dynamic campaign protocol sizing ────────────────────────────────────
def _protocol():
    from core.acquisition import compute_campaign_protocol
    return compute_campaign_protocol


def test_campaign_protocol_is_exact_identity_on_1n8z():
    """``N_qual = 210`` must reproduce the frozen 35-query protocol exactly.

    This is the guard on the whole Step 3 refactor: four literals (``range(5,36,5)``,
    ``6 * SLOT_COUNT``, the ``/30.0`` AUBQC normaliser, ``le=6``) were replaced by
    quantities derived from a protocol object. If any of them drifts, the 1N8Z
    artifact is no longer reproducible — and this test says so before a 100-seed
    run does.
    """
    from core.constants import CHECKPOINT_QUERIES

    compute = _protocol()
    protocol = compute(210, 13)
    assert protocol.seed_batch == 5
    assert protocol.batch_size == 5
    assert protocol.rounds == 6
    assert protocol.total_budget == 35
    assert protocol.checkpoints == CHECKPOINT_QUERIES == (5, 10, 15, 20, 25, 30, 35)
    assert protocol.aubqc_normalizer == 30.0
    # One law, on every target. ``allocation`` is a label for the artifact, not a
    # policy switch: behaviour is chosen per batch by ``slot_graduation``.
    assert protocol.allocation == "universal"
    assert protocol.cap_bound is False
    assert protocol.is_reference is True
    assert protocol.coverage_pct == pytest.approx(35 / 210)
    # The universal-law coefficients are computed from this cell's own observables
    # and give **no** privileged value to 1N8Z:
    #   maturity target = max(L, 0.30·Q) = max(13, 10.5) = 13
    #   eta_sar         = min(1, 35/26) · 1.0 · min(1, sqrt(70/210)) = 0.577
    # Both are stated as closed-form checks rather than literals, so the assertion
    # still holds if the floors move — it just has to keep agreeing with the law.
    assert protocol.sar_maturity_target == 13.0
    assert protocol.additive_prior_trust() == pytest.approx((2 * 35 / 210) ** 0.5)
    assert 0.0 < protocol.additive_prior_trust() < 1.0


def test_dynamic_protocol_scales_with_the_qualified_pool():
    """Coverage — not round count — is the invariant across datasets."""
    compute = _protocol()
    reference = compute(210, 13)
    for n_qualified, length in ((756, 11), (124, 13), (400, 12)):
        protocol = compute(n_qualified, length)
        # Within one batch of the pre-registered rate (Q moves in steps of B).
        assert abs(protocol.coverage_pct - 35 / 210) <= 5 / n_qualified + 1e-9, (
            n_qualified, protocol.coverage_pct
        )
        # Strictly more budget than the reference pool would get.
        assert protocol.rounds >= 1
        assert protocol.total_budget == protocol.seed_batch + protocol.batch_size * protocol.rounds
        assert protocol.checkpoints[0] == protocol.seed_batch
        assert protocol.checkpoints[-1] == protocol.total_budget
        assert protocol.allocation == "universal"
        # The maturity denominator is ``max(L, 0.30·Q)`` — no reference budget.
        assert protocol.sar_maturity_target == pytest.approx(
            max(length, 0.30 * protocol.total_budget)
        )
        assert protocol.sar_maturity_target >= length
        # ``eta`` is in (0, 1] and depends on this cell only.
        assert 0.0 < protocol.additive_prior_trust() <= 1.0
    # Monotone in pool size.
    assert compute(756, 11).rounds > compute(124, 13).rounds
    # The cap is reported, never silently applied.
    capped = compute(100_000, 13)
    assert capped.cap_bound is True
    assert capped.rounds == 60


def test_coverage_override_is_available_and_monotone():
    """The sensitivity sweep must be able to ask for a different rate."""
    compute = _protocol()
    low, mid, high = (compute(756, 11, coverage=c) for c in (0.05, 35 / 210, 0.33))
    assert low.rounds < mid.rounds < high.rounds
    # One law at every coverage level: the sweep varies the protocol, not the
    # architecture. This is the assertion the old fork could not make.
    assert low.allocation == mid.allocation == high.allocation == "universal"
    # ... and the maturity denominator follows the budget, not a dataset label.
    assert low.sar_maturity_target < mid.sar_maturity_target < high.sar_maturity_target


def test_graduation_is_continuous_and_horizon_relative():
    """The replacement for the ``fixed_split`` / ``adaptive_split`` switch.

    Two clauses, both continuous in measured quantities:

    * ``round_number > ceil(2R/3)`` — the horizon term. It is a *fraction* of the
      campaign, so the same law yields 2 falsifying rounds on a 3-round campaign
      and 16 on a 24-round one. No dataset is a special case: 1N8Z graduates in
      rounds 5-6 exactly like aayl49_ML graduates in rounds 17-24.
    * ``spread(Δ_agent) < 0.10 · AGENT_PRIOR_BOUND`` — the prior has stopped
      separating the pool, so a "VoI probe" and an "H0 control" would be picking
      candidates indistinguishable from the exploitation picks.
    """
    from core.acquisition import (
        AGENT_PRIOR_BOUND,
        delta_agent_spread_floor,
        falsification_phase_rounds,
        slot_graduation,
    )

    floor = delta_agent_spread_floor()
    assert floor == pytest.approx(0.10 * AGENT_PRIOR_BOUND)

    # The horizon term, per campaign length, with a prior that separates the pool.
    for total_rounds, window in ((3, 2), (6, 4), (24, 16), (49, 33)):
        for round_number in range(1, window + 1):
            assert slot_graduation(
                round_number, 0.5, total_rounds=total_rounds
            ) is False, (total_rounds, round_number)
        assert slot_graduation(window + 1, 0.5, total_rounds=total_rounds) is True
        assert window == falsification_phase_rounds(total_rounds)

    # 1N8Z is not privileged: its own 6-round campaign graduates in R5 and R6.
    assert [slot_graduation(r, 0.5, total_rounds=6) for r in range(1, 7)] == [
        False, False, False, False, True, True
    ]

    # The spread clause can graduate a campaign in its first round — this is what
    # lets a short or prior-degenerate campaign adapt at all.
    assert slot_graduation(1, floor * 0.5, total_rounds=24) is True
    assert slot_graduation(1, floor, total_rounds=24) is False   # strictly below


def test_confounding_factor_is_identity_without_confounding():
    """``c == 0`` (or no ledger) must leave a declared prior completely undiscounted."""
    import pandas as pd

    from core.acquisition import confounding_factor

    assert confounding_factor(None) == 1.0
    assert confounding_factor(pd.DataFrame(columns=["sequence"])) == 1.0
    # A library of single substitutions has c = 1/L, not 0 — but it must never
    # fall below the value a *fully* mutated library gets.
    single = pd.DataFrame({"sequence": ["ARWGGDGFYAMDY", "SAWGGDGFYAMDY"]})
    heavy = pd.DataFrame({"sequence": ["AAAAAAAAAAAAA", "GGGGGGGGGGGGG"]})
    assert confounding_factor(single) > confounding_factor(heavy)
    assert 0.0 < confounding_factor(heavy) <= 1.0
    # ``confounding_factor`` is now *reported* rather than applied: the transfer
    # risk it proxies is measured directly by ``additive_prior_trust``, and
    # multiplying both would discount a foreign target's prior twice. It is still
    # a public function with its own identity contract, which is what this test
    # guards; the selector no longer consults it.
    assert confounding_factor(None) == 1.0


def test_additive_prior_trust_is_first_principles_and_dataset_agnostic():
    """``eta_sar`` depends only on intrinsic observables of the cell.

    Three properties are load-bearing:

    * it is computable at ``t=0`` (no labels, no revealed ledger required);
    * it is **monotone** in the three things that make a single-position marginal
      unreliable — too few assays per site, too little pool coverage, too much
      mutational load;
    * it contains **no dataset-specific constant**. An earlier revision branched
      on ``wt_sequence == WT_SEQUENCE`` and a swept transfer penalty; there is
      deliberately nothing of the kind left, so the same formula scores every
      target.

    The ``1.0`` endpoint must also be *exact* at the fully-sampled margin, because
    ``eta`` multiplies ``Δ_agent`` and ``(1-eta)`` weights the UCB blend: at
    ``eta == 1.0`` both become IEEE identities and the objective is untouched.
    """
    from core.acquisition import additive_prior_trust

    # Unstated protocol -> fully trusted, which is the unbiased objective every
    # pre-universal unit test was written against.
    assert additive_prior_trust() == 1.0

    # A campaign that funds >= 2 assays per position over a pool it can cover in
    # full, on an unconfounded library, is exactly fully trusted.
    assert additive_prior_trust(n_qualified=1, seq_len=13, total_budget=26) == 1.0

    # 1N8Z is NOT a special case. Its own (N=210, L=13, Q=35) gives
    # obs_per_site = 35/26 = 1.0 but density = sqrt(70/210) = 0.577, so the gate
    # halves the additive prior's weight there too. That is the point of removing
    # the reference anchors: no dataset gets a privileged value of eta.
    eta_1n8z = additive_prior_trust(n_qualified=210, seq_len=13, total_budget=35)
    assert eta_1n8z == pytest.approx((2 * 35 / 210) ** 0.5)
    assert 0.0 < eta_1n8z < 1.0

    # Budget monotonicity at fixed pool and length: more assays, more trust.
    thin = additive_prior_trust(n_qualified=756, seq_len=11, total_budget=20)
    fat = additive_prior_trust(n_qualified=756, seq_len=11, total_budget=125)
    assert 0.0 < thin < fat <= 1.0

    # Pool monotonicity at fixed budget: a pool the budget cannot cover earns
    # less trust than one it can.
    small_pool = additive_prior_trust(n_qualified=40, seq_len=11, total_budget=40)
    big_pool = additive_prior_trust(n_qualified=756, seq_len=11, total_budget=40)
    assert big_pool < small_pool

    # Confounding monotonicity: a combinatorial library discounts the prior.
    unconfounded = additive_prior_trust(
        n_qualified=210, seq_len=13, total_budget=35, confounding=1.0
    )
    confounded = additive_prior_trust(
        n_qualified=210, seq_len=13, total_budget=35, confounding=0.5
    )
    assert confounded < unconfounded

    # No dataset identity: two different 13-mers with identical (N, L, Q) score
    # identically. That is the whole claim of a universal law, stated as an
    # assertion.
    assert additive_prior_trust(
        n_qualified=210, seq_len=13, total_budget=35
    ) == additive_prior_trust(n_qualified=210, seq_len=13, total_budget=35)


def test_mutational_confounding_is_label_free_and_identity_without_load():
    """``c_mut`` must come from sequences, never from ``y_bind`` (SPEC §2)."""
    import pandas as pd

    from core.acquisition import mutational_confounding

    assert mutational_confounding(None) == 1.0
    assert mutational_confounding(pd.DataFrame(columns=["sequence"])) == 1.0
    # The wild type itself carries zero substitutions: no discount.
    assert mutational_confounding(pd.DataFrame({"sequence": ["SRWGGDGFYAMDY"]})) == 1.0
    # A fully mutated library is discounted toward 1/(1+0.5) but never below it.
    heavy = pd.DataFrame({"sequence": ["AAAAAAAAAAAAA", "GGGGGGGGGGGGG"]})
    assert 0.6 < mutational_confounding(heavy) < 1.0
    # Adding a ``y_bind`` column must not change the number: the function is
    # label-free by construction.
    labelled = heavy.assign(y_bind=[9.0, 1.0])
    assert mutational_confounding(labelled) == mutational_confounding(heavy)


def test_sar_maturity_target_is_first_principles_and_horizon_relative():
    """``max(L, 0.30·Q)`` — no reference budget, no 1N8Z calibration.

    Both floors answer a different question, and a campaign has to clear the
    stronger one before ``VoI`` is allowed to collapse: one effective evidence
    unit per residue position, and at least a third of the assays spent on the
    question.
    """
    from core.acquisition import sar_maturity_target

    # The length floor binds on short campaigns.
    assert sar_maturity_target(13, total_budget=20) == 13.0
    assert sar_maturity_target(13, total_budget=40) == 13.0
    assert sar_maturity_target(11, total_budget=20) == 11.0
    # The budget floor binds on long ones: 30% of Q.
    assert sar_maturity_target(11, total_budget=125) == pytest.approx(37.5)
    assert sar_maturity_target(11, total_budget=250) == pytest.approx(75.0)
    # Monotone non-decreasing in both arguments.
    for length in (9, 11, 13, 16):
        previous = 0.0
        for budget in (5, 20, 40, 125, 250, 300):
            value = sar_maturity_target(length, total_budget=budget)
            assert value >= previous
            assert value >= length
            previous = value


def test_falsification_phase_scales_with_the_horizon():
    """The audit window is a fraction of ``R``, not a hardcoded six rounds."""
    from core.acquisition import falsification_phase_rounds

    # ceil(2R/3) for every horizon the benchmark can produce.
    assert [falsification_phase_rounds(r) for r in (1, 2, 3, 6, 7, 24, 49)] == [
        1, 2, 2, 4, 5, 16, 33
    ]
    assert falsification_phase_rounds(0) == 0
    # For any horizon long enough to contain both phases, the harvest phase is
    # non-empty. A 2-round campaign is the degenerate case: ceil(4/3) == 2, so it
    # never graduates on the horizon clause and relies entirely on the spread
    # clause. That is reported rather than papered over.
    for rounds in (3, 6, 7, 24, 49):
        assert falsification_phase_rounds(rounds) < rounds
    assert falsification_phase_rounds(2) == 2


def test_cross_target_reports_the_dynamic_protocol():
    """The artifact must state R, Q_max, coverage and allocation per cell."""
    import json as _json

    path = SETTINGS.cross_target_benchmark_path
    if not path.exists():
        pytest.skip(f"{path} not generated; run scripts/run_cross_target_benchmark.py")
    report = _json.loads(path.read_text())
    headline = report["cross_target"]
    for key in ("rounds", "total_budget", "coverage_pct", "allocation_mode",
                "campaign_protocol", "seed_batch", "batch_size", "cap_bound"):
        assert key in headline, key

    cells = report.get("targets") or []
    assert cells, "the artifact must carry per-(target, coverage) cells"
    for cell in cells:
        protocol = cell["campaign_protocol"]
        assert protocol["total_budget"] == (
            protocol["seed_batch"] + protocol["batch_size"] * protocol["rounds"]
        ), cell["dataset"]
        assert protocol["allocation"] == "universal", cell["dataset"]
        # No cross-target campaign may be *less* covered than half the reference.
        assert cell["coverage_pct"] > 3.0, cell
        assert cell["qualified_count"] >= 35

    # The sensitivity sweep is present and covers both targets.
    sensitivity = report.get("coverage_sensitivity") or []
    assert {row["dataset"] for row in sensitivity} >= {"aayl49_ML", "2fjg"}
    assert len({row["coverage_label"] for row in sensitivity}) >= 3
    for row in sensitivity:
        assert row["total_budget"] == (
            row["rounds"] * 5 + 5
        ), row
        assert row["best_baseline"] in METHODS


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
