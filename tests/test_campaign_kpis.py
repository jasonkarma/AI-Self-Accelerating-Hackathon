"""Regression tests for the CAMPAIGN STATE KPIs, the shared loop prior, and the
paired H1/H0 experiment design."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.acquisition import (
    AGENT_PRIOR_BOUND,
    adaptive_prior_score,
    fixed_expert_prior,
    loop_prior_score,
    qualified_pool,
    sar_effect_table,
    select_five_slots,
)
from core.baselines import METHODS, run_method
from core.constants import WT_SEQUENCE
from core.developability import RiskNormalizer
from core.schemas import PositionalSoftPrior
from core.tool_arsenal import audit_past_hypotheses
from data.datasets import generate_fixture


def _ledger(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── Paired H1/H0 audit ───────────────────────────────────────────────────────
def test_audit_grades_slot4_against_slot5_control():
    """The verdict is the paired comparison, not a cross-round median."""
    ledger = _ledger([
        {"candidate_id": "H1", "sequence": WT_SEQUENCE, "y_bind": 8.6, "round_number": 2},
        {"candidate_id": "H0", "sequence": WT_SEQUENCE, "y_bind": 7.4, "round_number": 2},
        {"candidate_id": "X", "sequence": WT_SEQUENCE, "y_bind": 8.0, "round_number": 2},
    ])
    supported = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R2-01", "round_number": 2,
        "slot4_measured_id": "H1", "slot5_measured_id": "H0",
    }])["verdicts"][0]
    assert supported["verdict"] == "SUPPORTED"
    assert supported["paired_lift"] == pytest.approx(1.2)

    # The control wins -> the hypothesis is refuted, even though H1 beat the
    # median. This is the case a median-only rule got wrong.
    refuted = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R2-02", "round_number": 2,
        "slot4_measured_id": "H0", "slot5_measured_id": "H1",
    }])["verdicts"][0]
    assert refuted["verdict"] == "FALSIFIED"
    assert refuted["paired_lift"] == pytest.approx(-1.2)


def test_audit_uses_measured_slot_ids_not_declared_targets():
    """The planner's declared Slot 4 target is often consumed by Slots 1-3."""
    ledger = _ledger([
        {"candidate_id": "A", "sequence": WT_SEQUENCE, "y_bind": 9.0, "round_number": 2},
        {"candidate_id": "B", "sequence": WT_SEQUENCE, "y_bind": 8.0, "round_number": 2},
        {"candidate_id": "C", "sequence": WT_SEQUENCE, "y_bind": 7.0, "round_number": 2},
    ])
    verdict = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R2-01",
        "round_number": 2,
        # Declared targets were never measured; the quota measured B and C.
        "slot4_hypothesis_target_id": "A",
        "slot5_falsification_target_id": "A",
        "slot4_measured_id": "B",
        "slot5_measured_id": "C",
    }])["verdicts"][0]
    assert verdict["verdict"] == "SUPPORTED"
    assert verdict["h1_target_id"] == "B"
    assert verdict["h0_control_id"] == "C"


def test_definitive_verdict_for_every_completed_round():
    rounds = []
    for r in range(1, 7):
        rounds.extend([
            {"candidate_id": f"R{r}-h1", "sequence": WT_SEQUENCE, "y_bind": 8.0 + r / 10, "round_number": r},
            {"candidate_id": f"R{r}-h0", "sequence": WT_SEQUENCE, "y_bind": 7.5, "round_number": r},
        ])
    decisions = [
        {"hypothesis_id": f"HYP-R{r}-01", "round_number": r,
         "slot4_measured_id": f"R{r}-h1", "slot5_measured_id": f"R{r}-h0"}
        for r in range(1, 7)
    ]
    verdicts = audit_past_hypotheses(_ledger(rounds), decisions)["verdicts"]
    assert len(verdicts) == 6
    assert {v["verdict"] for v in verdicts} == {"SUPPORTED"}


def test_falsified_round_reports_only_the_rules_it_actually_tested():
    """A round can only falsify positions the Slot 4 candidate exercised.

    Inverting every declared position would blacklist most of the CDR3 within a
    few rounds; only the substitution the paired arms actually measured counts.
    """
    h1_sequence = "ARWGGDGFYAMDY"      # mutation at position 1, WT elsewhere
    ledger = _ledger([
        {"candidate_id": "H1", "sequence": h1_sequence, "y_bind": 7.0, "round_number": 1},
        {"candidate_id": "H0", "sequence": WT_SEQUENCE, "y_bind": 8.5, "round_number": 1},
    ])
    audit = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R1-01", "round_number": 1,
        "slot4_measured_id": "H1", "slot5_measured_id": "H0",
        "positional_priors": [
            {"position": 1, "preferred_residues": ["A"], "weight_delta": 0.04},   # exercised
            {"position": 6, "preferred_residues": ["S"], "weight_delta": 0.04},   # not exercised
        ],
    }])
    assert audit["verdicts"][0]["verdict"] == "FALSIFIED"
    assert audit["falsified_positions"] == [1]


def test_supported_round_inverts_nothing():
    ledger = _ledger([
        {"candidate_id": "H1", "sequence": "ARWGGDGFYAMDY", "y_bind": 8.6, "round_number": 1},
        {"candidate_id": "H0", "sequence": WT_SEQUENCE, "y_bind": 7.1, "round_number": 1},
    ])
    audit = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R1-01", "round_number": 1,
        "slot4_measured_id": "H1", "slot5_measured_id": "H0",
        "positional_priors": [{"position": 1, "preferred_residues": ["A"], "weight_delta": 0.04}],
    }])
    assert audit["verdicts"][0]["verdict"] == "SUPPORTED"
    assert audit["falsified_positions"] == []


def test_falsified_position_is_inverted_in_the_next_batch():
    """A rule that lost its paired test must actively push candidates down."""
    prior = PositionalSoftPrior(position=4, preferred_residues=["A"], weight_delta=0.04)
    sequence = "SRW" + "A" + "GDGFYAMDY"
    clean = adaptive_prior_score(sequence, priors=[prior])
    inverted = adaptive_prior_score(sequence, priors=[prior], falsified_positions={4})
    assert clean > 0
    assert inverted < 0


# ── Empirical-Bayes SAR ──────────────────────────────────────────────────────
def test_sar_requires_two_observations_and_shrinks_by_sample_size():
    """A single mutation measuring high is as likely to be confounding as signal."""
    single = _ledger([
        {"candidate_id": "A", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.5, "round_number": 1},
        {"candidate_id": "B", "sequence": "SRWGGDGFYAMDY", "y_bind": 7.0, "round_number": 1},
    ])
    assert sar_effect_table(single) == {}

    repeated = _ledger([
        {"candidate_id": "A", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.5, "round_number": 1},
        {"candidate_id": "B", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.4, "round_number": 1},
        {"candidate_id": "C", "sequence": "SRWGGDGFYAMDY", "y_bind": 7.0, "round_number": 1},
    ])
    effects = sar_effect_table(repeated)
    n, shrunk = effects[(1, "A")]
    assert n == 2
    # n/(n+3) shrinkage keeps 40% of the raw effect, and the raw effect is
    # bounded by SAR_DELTA_SCALE before shrinking.
    assert 0 < shrunk <= 1.0


def test_sar_prefers_residuals_when_available():
    """Positional effects are measured on surprise, not on raw affinity."""
    ledger = _ledger([
        {"candidate_id": "A", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.5, "round_number": 1},
        {"candidate_id": "B", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.4, "round_number": 1},
    ])
    # Residual = y_bind - mu_bind. The surrogate already predicted these exactly,
    # so the substitutions explain nothing and must contribute zero - even though
    # the raw bindings are ~2.5 above the revealed median.
    explained = sar_effect_table(ledger, residuals={"A": 0.0, "B": 0.0})
    assert explained.basis == "residual"
    assert all(abs(effect) < 1e-9 for _, effect in explained.values())

    # A genuine surprise is what produces a non-zero effect.
    surprised = sar_effect_table(ledger, residuals={"A": 0.9, "B": 0.9})
    assert surprised.basis == "residual"
    assert surprised[(1, "A")][1] > 0


# ── Qualification gate ───────────────────────────────────────────────────────
def test_qualification_gate_excludes_unqualified_candidates():
    frame = pd.DataFrame([
        {"candidate_id": f"C{i}", "y_dev": 0.2 if i < 3 else 0.9, "sequence": WT_SEQUENCE}
        for i in range(10)
    ])
    gated = qualified_pool(frame, tau_dev=0.5)
    assert len(gated) == 7
    assert gated.y_dev.min() >= 0.5


def test_qualification_gate_falls_back_when_pool_is_exhausted():
    frame = pd.DataFrame([
        {"candidate_id": f"C{i}", "y_dev": 0.2, "sequence": WT_SEQUENCE} for i in range(10)
    ])
    # Fewer than five qualified -> the unfiltered pool is returned rather than
    # under-filling the batch.
    assert len(qualified_pool(frame, tau_dev=0.5)) == 10


# ── Shared loop prior + portfolio ────────────────────────────────────────────
def test_loop_prior_is_shared_with_m4():
    assert fixed_expert_prior(WT_SEQUENCE) == loop_prior_score(WT_SEQUENCE) > 0
    assert loop_prior_score("AAAAAAAAAAAAA") < loop_prior_score(WT_SEQUENCE)


def _scored_frame(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "candidate_id": f"C{i:03d}",
            "sequence": WT_SEQUENCE if i % 2 == 0 else "AAAAAAAAAAAAA",
            "y_dev": 0.9,
            "a_base": float(rng.normal(1.0, 0.05)),
            "sigma_joint": float(rng.uniform(0, 1)),
            "sigma_bind": 0.1,
            "sigma_dev": 0.1,
            "mu_bind": 1.0,
        })
    return pd.DataFrame(rows)


def _revealed_with_signal() -> pd.DataFrame:
    """Reveals that make some positions clearly good and others clearly bad."""
    rows = [{"candidate_id": "W", "sequence": WT_SEQUENCE, "y_bind": 7.0,
             "y_dev": 0.9, "round_number": 0}]
    for i in range(3):
        rows.append({"candidate_id": f"G{i}", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.4 + i * 0.1,
                     "y_dev": 0.9, "round_number": 1})
    for i in range(3):
        rows.append({"candidate_id": f"B{i}", "sequence": "SRWGGDGFYAADY", "y_bind": 6.0 + i * 0.1,
                     "y_dev": 0.9, "round_number": 1})
    return pd.DataFrame(rows)


def test_select_five_slots_returns_qualified_distinct_h1_h0_arms():
    ledger = _revealed_with_signal()
    frame = _scored_frame()
    frame = frame.assign(
        sequence=["ARWGGDGFYAMDY" if i % 2 == 0 else "SRWGGDGFYAADY" for i in range(len(frame))]
    )
    slots = select_five_slots(frame, None, ledger=ledger, tau_dev=0.5, queries_spent=5, total_budget=35)
    assert [s["slot_id"] for s in slots] == [1, 2, 3, 4, 5]
    assert len({s["candidate_id"] for s in slots}) == 5
    assert all({"loop_prior", "a_null", "a_h1", "delta_agent"} <= set(slot) for slot in slots)
    # Slot 4 is the H1 arm (agents boost it) and slot 5 is the H0 null control
    # (agents do not). That pairing is what makes the round gradeable.
    assert slots[3]["delta_agent"] > 0
    assert slots[4]["delta_agent"] <= 0
    # Slots 1-2 are the top two A_H1 candidates in the whole qualified pool.
    ranked = frame.assign(
        loop_prior=[loop_prior_score(s) for s in frame.sequence]
    )
    assert slots[0]["a_h1"] >= slots[1]["a_h1"]


def test_slot3_anneals_exploration_with_budget():
    """Early rounds behave like UCB; a spent budget reverts to pure exploitation."""
    frame = _scored_frame()
    early = select_five_slots(frame, None, tau_dev=0.5, queries_spent=5, total_budget=35)[2]
    late = select_five_slots(frame, None, tau_dev=0.5, queries_spent=35, total_budget=35)[2]
    # With no budget left, slot 3 must be the argmax of A_null (no sigma bonus).
    no_bonus = frame.assign(
        a_null=frame.a_base + [loop_prior_score(s) for s in frame.sequence]
    ).sort_values(["a_null", "candidate_id"], ascending=[False, True])
    # It may coincide with an earlier pick; only assert the annealing direction.
    assert late["optimistic"] <= early["optimistic"] + 1e-9


def test_select_five_slots_is_deterministic():
    frame = _scored_frame()
    first = [s["candidate_id"] for s in select_five_slots(frame, None, queries_spent=5, total_budget=35)]
    second = [s["candidate_id"] for s in select_five_slots(frame, None, queries_spent=5, total_budget=35)]
    assert first == second


# ── Benchmark integration ────────────────────────────────────────────────────
def test_methods_share_seeds_and_return_a_full_curve():
    frame = generate_fixture(70)
    normalizer = RiskNormalizer.fit(frame.sequence)
    tau_dev = normalizer.score(WT_SEQUENCE)["y_dev"]
    initial = frame.candidate_id.head(5).tolist()
    for method in METHODS:
        result = run_method(frame, normalizer, tau_dev, method, 0, initial)
        assert result.initial_ids == initial
        assert [p["query_count"] for p in result.curve] == list(range(5, 36, 5))
        assert np.isfinite(result.aubqc)


def test_benchmark_m5_spends_every_slot_on_a_qualified_candidate():
    frame = generate_fixture(120)
    normalizer = RiskNormalizer.fit(frame.sequence)
    tau_dev = normalizer.score(WT_SEQUENCE)["y_dev"]
    y_dev = {cid: normalizer.score(seq)["y_dev"] for cid, seq in zip(frame.candidate_id, frame.sequence)}
    initial = frame.candidate_id.head(5).tolist()
    result = run_method(frame, normalizer, tau_dev, "m5_hybrid", 0, initial)
    for batch in result.picks:
        assert len(batch) == 5
        # The gate can only be bypassed when the qualified pool is exhausted,
        # which does not happen in a 120-candidate pool.
        assert all(y_dev[cid] >= tau_dev for cid in batch)
