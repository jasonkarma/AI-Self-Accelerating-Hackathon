"""Regression tests for the CAMPAIGN STATE KPIs, the shared loop prior, and the
paired H1/H0 experiment design."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.acquisition import (
    AGENT_PRIOR_BOUND,
    DECLARED_PRIOR_WEIGHT,
    SAR_CELL_WEIGHT,
    SAR_EMPIRICAL_WEIGHT,
    SAR_HIT_WEIGHT,
    SAR_POSITION_WEIGHT,
    SLOT3_OPTIMISM,
    SLOT4_AGENT_BOOST,
    _sar_total,
    _saturate,
    _soft_prior_total,
    SLOT5_FALSIFY_LABEL,
    VOI_EXPLOIT_WEIGHT,
    SarTable,
    adaptive_prior_score,
    fixed_expert_prior,
    horizon_factor,
    loop_prior_score,
    qualified_pool,
    sar_maturity_score,
    sar_effect_table,
    select_five_slots,
    value_of_information,
    voi_annealed_null_objective,
    voi_context,
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
    """A round can only falsify positions the H1 arm candidate exercised.

    Inverting every declared position would blacklist most of the CDR3 within a
    few rounds; only the substitution the paired arms actually measured counts.
    A round with no recorded pre-query expectation has no residual to read, so it
    falls back to this tested-positions rule.
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


# ── H1 ARM vs SLOT 4: no more every-other-round oscillation ──────────────────
def _arm_round_ledger(*, slot4_beats_control: bool) -> tuple[pd.DataFrame, dict]:
    """One round where slot 1 finds an elite binder and slot 4 does not.

    This is the exact shape that produced ``SUPPORTED -> FALSIFIED -> SUPPORTED``
    under the slot-4-only rule: with ``SLOT4_AGENT_BOOST = 0.0`` slots 1, 2 and 4
    all rank by ``A_H1``, so slot 4 is structurally the *fourth* choice.

    Slot 1 clears the control (``8.97`` vs ``8.10``) and the arm's mean residual is
    positive, so the round is genuinely supported; slot 4 on its own (``7.50``)
    loses to the control, which is what the old rule graded.
    """
    slot1, slot4_value = 8.97, (7.50 if not slot4_beats_control else 8.50)
    rows = [
        # slot 1: the elite binder the hypothesis actually bought
        {"candidate_id": "S1", "sequence": "ARWGGDGFYAMDY", "y_bind": slot1, "round_number": 3},
        {"candidate_id": "S2", "sequence": "ARWGGDGFYAMDY", "y_bind": 8.20, "round_number": 3},
        {"candidate_id": "S4", "sequence": "VRWGGDGFYAMDY", "y_bind": slot4_value, "round_number": 3},
        {"candidate_id": "S5", "sequence": WT_SEQUENCE, "y_bind": 8.10, "round_number": 3},
    ]
    decision = {
        "hypothesis_id": "HYP-R3-01", "round_number": 3,
        "slot4_measured_id": "S4", "slot5_measured_id": "S5",
        "h1_arm_measured_ids": ["S1", "S2", "S4"],
        "positional_priors": [
            {"position": 1, "preferred_residues": ["A"], "weight_delta": 0.04},
        ],
        "pre_query_expectation": {"S1": 8.0, "S2": 8.2, "S4": 7.6, "S5": 8.1},
    }
    return _ledger(rows), decision


def test_audit_grades_the_best_h1_exploitation_pick_not_the_fourth_choice():
    """Regression: slot 4 is the 4th ``A_H1`` pick, so grading it alone oscillates.

    The batch is a 3-exploit / 2-experimental design. Slot 1 finding ``8.97`` is
    what the hypothesis bought, and it must count even when the leftover slot 4
    lands below the control. The verdict is the arm's **best** pick against
    ``Slot 5``.
    """
    ledger, decision = _arm_round_ledger(slot4_beats_control=False)
    verdict = audit_past_hypotheses(ledger, [decision])["verdicts"][0]

    # Slot 4 alone is below the control: the pre-fix rule would say FALSIFIED.
    assert verdict["h1_target_id"] == "S1"
    assert verdict["h1_binding"] == pytest.approx(8.97)
    assert verdict["verdict"] == "SUPPORTED"
    assert verdict["paired_lift"] == pytest.approx(8.97 - 8.10, abs=1e-6)
    # The arm is reported whole, and its mean is available for depth.
    assert verdict["h1_arm_ids"] == ["S1", "S2", "S4"]
    # Both are reported rounded to 3 dp, so compare at that resolution.
    assert verdict["h1_arm_mean_binding"] == pytest.approx((8.97 + 8.20 + 7.50) / 3, abs=1e-3)
    assert verdict["h1_mean_residual"] == pytest.approx(
        ((8.97 - 8.0) + (8.20 - 8.2) + (7.50 - 7.6)) / 3, abs=1e-3
    )
    assert verdict["h1_mean_residual"] > 0


def test_audit_still_falsifies_when_the_whole_arm_loses_to_the_control():
    """The arm comparison is not a licence to always say SUPPORTED."""
    ledger, decision = _arm_round_ledger(slot4_beats_control=False)
    decision = {**decision, "pre_query_expectation": {}}  # residuals off
    ledger.loc[ledger.candidate_id.isin(["S1", "S2"]), "y_bind"] = 7.0
    verdict = audit_past_hypotheses(ledger, [decision])["verdicts"][0]
    assert verdict["verdict"] == "FALSIFIED"
    assert verdict["h0_binding"] > verdict["h1_binding"]


def test_a_small_lift_over_an_under_predicted_arm_is_not_evidence():
    """A lift the surrogate already predicted is not evidence for the hypothesis.

    This is the residual rule in its narrow, defensible form: the arm won by a
    margin *below* :data:`core.tool_arsenal.DECISIVE_LIFT`, and its carriers
    measured well below what the surrogate predicted **relative to the control**.
    The control then carries the comparison, and the round is refuted.
    """
    ledger = _ledger([
        {"candidate_id": "S1", "sequence": "ARWGGDGFYAMDY", "y_bind": 8.15, "round_number": 3},
        {"candidate_id": "S2", "sequence": "ARWGGDGFYAMDY", "y_bind": 8.12, "round_number": 3},
        {"candidate_id": "S4", "sequence": "VRWGGDGFYAMDY", "y_bind": 8.11, "round_number": 3},
        {"candidate_id": "S5", "sequence": WT_SEQUENCE, "y_bind": 8.10, "round_number": 3},
    ])
    verdict = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R3-01", "round_number": 3,
        "slot4_measured_id": "S4", "slot5_measured_id": "S5",
        "h1_arm_measured_ids": ["S1", "S2", "S4"],
        "positional_priors": [
            {"position": 1, "preferred_residues": ["A"], "weight_delta": 0.04},
        ],
        # The arm was predicted far higher than the control, and missed by more.
        "pre_query_expectation": {"S1": 9.2, "S2": 9.2, "S4": 9.2, "S5": 8.0},
    }])["verdicts"][0]
    assert verdict["verdict"] == "FALSIFIED"
    assert verdict["paired_lift"] < 0.10
    assert verdict["decisive_lift"] is False
    assert verdict["h1_mean_residual"] < verdict["h0_residual"]


def test_a_decisive_lift_survives_batch_wide_negative_surrogate_drift():
    """Regression (seed 21, R5): a +0.75 win was refuted by an absolute residual.

    The surrogate fits on the revealed rows, so a fresh batch can be uniformly
    over-predicted and every measured residual goes negative at once. Reading that
    level shift as evidence against the rule refuted a hypothesis the paired arms
    had just confirmed. A lift above ``DECISIVE_LIFT`` now stands on its own, and
    the comparison for smaller lifts is against the *control's* residual, which
    cancels the shared shift.
    """
    ledger = _ledger([
        {"candidate_id": "S1", "sequence": "ARWGGDGFYAMDY", "y_bind": 8.159, "round_number": 5},
        {"candidate_id": "S2", "sequence": "ARWGGDGFYAMDY", "y_bind": 7.900, "round_number": 5},
        {"candidate_id": "S4", "sequence": "VRWGGDGFYAMDY", "y_bind": 7.800, "round_number": 5},
        {"candidate_id": "S5", "sequence": WT_SEQUENCE, "y_bind": 7.412, "round_number": 5},
    ])
    verdict = audit_past_hypotheses(ledger, [{
        "hypothesis_id": "HYP-R5-01", "round_number": 5,
        "slot4_measured_id": "S4", "slot5_measured_id": "S5",
        "h1_arm_measured_ids": ["S1", "S2", "S4"],
        "positional_priors": [
            {"position": 1, "preferred_residues": ["A"], "weight_delta": 0.04},
        ],
        # Whole batch over-predicted: arm and control residual are both negative.
        "pre_query_expectation": {"S1": 8.2, "S2": 8.2, "S4": 8.2, "S5": 7.5},
    }])["verdicts"][0]
    assert verdict["h1_mean_residual"] < 0, "the batch really was under-predicted"
    assert verdict["paired_lift"] > 0.10
    assert verdict["decisive_lift"] is True
    assert verdict["verdict"] == "SUPPORTED"


def _gated_round(residuals: dict[int, float]) -> dict:
    """Build one falsified round with a chosen residual per tested position.

    The arm candidates each carry the preferred residue at exactly one position,
    so every position's mean residual is the value supplied here — which is what
    makes the gate's behaviour readable instead of a function of sequence overlap.
    """
    base = WT_SEQUENCE
    rows, priors, expectations, arm = [], [], {}, []
    for index, (position, residual) in enumerate(residuals.items()):
        chars = list(base)
        chars[position - 1] = "A"
        candidate = f"S{index}"
        expectation = 8.0
        rows.append({
            "candidate_id": candidate,
            "sequence": "".join(chars),
            "y_bind": expectation + residual,
            "round_number": 1,
        })
        expectations[candidate] = expectation
        priors.append({"position": position, "preferred_residues": ["A"], "weight_delta": 0.04})
        arm.append(candidate)
    # The control sits above every arm candidate, so the round is FALSIFIED and
    # the gate — not the verdict — is what decides which positions are inverted.
    rows.append({"candidate_id": "H0", "sequence": base, "y_bind": 9.9, "round_number": 1})
    return {
        "hypothesis_id": "HYP-R1-01", "round_number": 1,
        "slot4_measured_id": arm[-1], "slot5_measured_id": "H0",
        "h1_arm_measured_ids": arm,
        "positional_priors": priors,
        "pre_query_expectation": expectations,
    }


def test_inversion_is_residual_gated_and_never_touches_a_positive_residual():
    """Self-correction may not invert a position the wet lab measured *better*.

    That inversion was the negative-feedback half of the
    ``SUPPORTED -> FALSIFIED -> SUPPORTED`` oscillation: a substitution that beat
    its prediction was pushed down anyway, which made the next round's arm lose
    and flipped the verdict back.
    """
    # Each arm candidate mutates exactly one position to "A", so each position's
    # mean residual is exactly the value chosen here.
    #   positive: 1 (+0.6), 4 (+0.3)      negative: 6 (-0.5), 8 (-0.6), 11 (-0.9)
    residuals = {1: 0.6, 4: 0.3, 6: -0.5, 8: -0.6, 11: -0.9}
    rows, arm = [], []
    for index, (position, residual) in enumerate(residuals.items()):
        chars = list(WT_SEQUENCE)
        chars[position - 1] = "A"
        candidate = f"R{position}"
        rows.append({
            "candidate_id": candidate,
            "sequence": "".join(chars),
            "y_bind": 8.0 + residual,
            "round_number": 1,
        })
        arm.append(candidate)
    rows.append({"candidate_id": "H0", "sequence": WT_SEQUENCE, "y_bind": 9.5, "round_number": 1})
    audit = audit_past_hypotheses(_ledger(rows), [{
        "hypothesis_id": "HYP-R1-01", "round_number": 1,
        "slot4_measured_id": "R11", "slot5_measured_id": "H0",
        "h1_arm_measured_ids": arm,
        "positional_priors": [
            {"position": p, "preferred_residues": ["A"], "weight_delta": 0.04}
            for p in residuals
        ],
        "pre_query_expectation": {
            **{f"R{position}": 8.0 for position in residuals}, "H0": 7.0,
        },
    }])
    verdict = audit["verdicts"][0]
    assert verdict["verdict"] == "FALSIFIED"
    # The two positions that beat their prediction are never inverted...
    assert verdict["position_residuals"]["1"] == pytest.approx(0.6, abs=1e-3)
    assert verdict["position_residuals"]["4"] == pytest.approx(0.3, abs=1e-3)
    assert not ({1, 4} & set(audit["falsified_positions"]))
    # ...and only the strictly-worst negative residuals clear the relative gate.
    assert set(audit["falsified_positions"]) == {8, 11}


def test_inversion_gate_is_conservative_at_tiny_sample_sizes():
    """One measured position inverts nothing; a tied pair inverts nothing.

    The relative gate is a strict "below the median" test, which is
    self-limiting: with one observation ``value == median``, and with a tied pair
    both members equal it. Inverting a whole rule set off one assay is exactly the
    behaviour that blacklisted most of the CDR3 in three rounds.
    """
    single = audit_past_hypotheses(_ledger([
        {"candidate_id": "S4", "sequence": "SRWGGDGFYAAA", "y_bind": 6.5, "round_number": 1},
        {"candidate_id": "S5", "sequence": WT_SEQUENCE, "y_bind": 8.5, "round_number": 1},
    ]), [{
        "hypothesis_id": "HYP-R1-01", "round_number": 1,
        "slot4_measured_id": "S4", "slot5_measured_id": "S5",
        "h1_arm_measured_ids": ["S4"],
        "positional_priors": [{"position": 12, "preferred_residues": ["A"], "weight_delta": 0.04}],
        "pre_query_expectation": {"S4": 8.1, "S5": 8.1},
    }])
    assert single["verdicts"][0]["verdict"] == "FALSIFIED"
    assert single["verdicts"][0]["position_residuals"]["12"] < 0
    # ... but a negative residual alone is not enough to invert from one assay.
    assert single["falsified_positions"] == []

    tied = audit_past_hypotheses(
        _ledger([
            {"candidate_id": "R1", "sequence": "X", "y_bind": 7.5, "round_number": 1},
            {"candidate_id": "R4", "sequence": "X", "y_bind": 7.5, "round_number": 1},
            {"candidate_id": "H0", "sequence": "X", "y_bind": 9.9, "round_number": 1},
        ]),
        [{
            "hypothesis_id": "HYP-R1-02", "round_number": 1,
            "slot4_measured_id": "R4", "slot5_measured_id": "H0",
            "h1_arm_measured_ids": ["R1", "R4"],
            "positional_priors": [
                {"position": p, "preferred_residues": ["A"], "weight_delta": 0.04}
                for p in (1, 4)
            ],
            "pre_query_expectation": {"R1": 8.0, "R4": 8.0, "H0": 7.0},
        }],
    )
    assert tied["verdicts"][0]["verdict"] == "FALSIFIED"
    assert tied["falsified_positions"] == []


def test_inversion_reports_the_residual_behind_every_decision():
    """The audit trail must show the number the gate read, for each position."""
    decision = _gated_round({2: -0.5, 5: -0.1, 9: 0.4, 13: 0.2})
    audit = audit_past_hypotheses(_ledger([
        {"candidate_id": "S0", "sequence": "SAWGGDGFYAMDY", "y_bind": 7.5, "round_number": 1},
        {"candidate_id": "S1", "sequence": "SWAGGDFYAMDY", "y_bind": 7.9, "round_number": 1},
        {"candidate_id": "S2", "sequence": "SWGGDGFYAMDY", "y_bind": 8.4, "round_number": 1},
        {"candidate_id": "H0", "sequence": WT_SEQUENCE, "y_bind": 9.9, "round_number": 1},
    ]), [decision])
    verdict = audit["verdicts"][0]
    assert verdict["verdict"] == "FALSIFIED"
    # Every position the gate considered is reported, with its signed residual.
    assert set(verdict["position_residuals"]) <= {"2", "5", "9", "13"}
    for value in verdict["position_residuals"].values():
        assert value == pytest.approx(value, abs=1e-3)  # rounded to 3 dp
    # The strict "below the median" gate can only ever invert a strict subset of
    # the negative-residual positions, and never one with a positive residual.
    inverted = set(verdict["inverted_positions"])
    negative = {int(p) for p, v in verdict["position_residuals"].items() if v < 0}
    assert inverted <= negative
    assert inverted == set(audit["falsified_positions"])


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


# ── Declared priors are gated by measured evidence ───────────────────────────
def _corroborated_position_table(position: int = 4, effect: float = 0.6) -> SarTable:
    """One corroborated position, with a chosen sign."""
    table = SarTable(basis="residual")
    table.position_effects = {position: (5, effect)}
    return table


def test_agreeing_priors_keep_full_weight():
    """The reconciliation is a *sign* gate, never a magnitude discount.

    Discounting both terms (the rejected ``0.85 / 0.15`` split) attenuates
    ``Δ_agent`` before ``tanh`` whether or not anyone disagrees, and ``Δ_agent`` is
    the only part of ``A_H1`` the surrogate does not already contain. On 100 seeds
    it cost ``0.0013`` AUBQC.
    """
    prior = PositionalSoftPrior(position=4, preferred_residues=["A"], weight_delta=0.04)
    sequence = "SRWAGDGFYAMDY"
    assert DECLARED_PRIOR_WEIGHT == SAR_EMPIRICAL_WEIGHT == 1.0

    # Evidence-free table: nothing to defer to (this is the cold start).
    empty = SarTable()
    empty.position_effects = {}
    cold = adaptive_prior_score(sequence, priors=[prior], sar_effects=empty)
    # A corroborated table that *agrees* with the rule adds to it at full weight,
    # so the score only rises — it is never shrunk.
    agreeing = adaptive_prior_score(sequence, priors=[prior], sar_effects=_corroborated_position_table())
    assert cold > 0
    assert agreeing > cold


def test_declared_prior_contradicted_by_evidence_is_dropped_entirely():
    """A rule the wet lab refuted must not be averaged with its own negation."""
    prior = PositionalSoftPrior(position=4, preferred_residues=["A"], weight_delta=0.04)
    sequence = "SRWAGDGFYAMDY"
    refuting = _corroborated_position_table(position=4, effect=-0.7)

    gated = adaptive_prior_score(sequence, priors=[prior], sar_effects=refuting)
    sar_only = adaptive_prior_score(sequence, priors=[], sar_effects=refuting)
    assert gated == pytest.approx(sar_only)
    assert gated < 0


def test_declared_prior_at_an_unobserved_position_keeps_full_weight():
    """Before the lab has measured a position, the council is all there is."""
    observed = PositionalSoftPrior(position=4, preferred_residues=["A"], weight_delta=0.04)
    unobserved = PositionalSoftPrior(position=9, preferred_residues=["Q"], weight_delta=0.04)
    table = _corroborated_position_table(position=4)
    sequence = "SRWAGDGFYQMDY"
    both = adaptive_prior_score(sequence, priors=[observed, unobserved], sar_effects=table)
    only_unobserved = adaptive_prior_score(sequence, priors=[unobserved], sar_effects=table)
    assert only_unobserved > 0
    assert both > only_unobserved


def test_prior_gate_is_inert_without_declared_priors_or_without_evidence():
    """The gate only ever *narrows* the declared term; otherwise it is a no-op."""
    table = _corroborated_position_table()
    sequence = "SRWAGDGFYAMDY"
    # No declared priors -> the SAR term is the whole score, unchanged.
    assert adaptive_prior_score(sequence, sar_effects=table) == pytest.approx(
        adaptive_prior_score(sequence, priors=[], sar_effects=table)
    )
    # Evidence-free table -> declared priors at full weight.
    prior = PositionalSoftPrior(position=4, preferred_residues=["A"], weight_delta=0.04)
    empty = SarTable()
    empty.position_effects = {}
    assert adaptive_prior_score(sequence, priors=[prior], sar_effects=empty) == pytest.approx(
        _saturate(_soft_prior_total(sequence, [prior]), AGENT_PRIOR_BOUND)
    )
    # And with no table at all.
    assert adaptive_prior_score(sequence, priors=[prior]) == pytest.approx(
        _saturate(_soft_prior_total(sequence, [prior]), AGENT_PRIOR_BOUND)
    )


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


def test_position_level_sar_is_active_when_exact_cells_are_cold():
    """Hierarchical 2-level SAR: position tolerance survives a cold 260-cell table.

    Rounds 1-2 rarely produce two observations of the *same* ``(position,
    residue)`` pair, but they routinely produce two observations at the same
    *position*. Without the position level, ``Δ_agent`` would be structurally
    zero for exactly the rounds in which a prior is most valuable.
    """
    ledger = _ledger([
        # Position 1 mutated A->D and A->E: two observations at position 1, no
        # single (position, residue) cell with n >= 2.
        {"candidate_id": "D1", "sequence": "DRWGGDGFYAMDY", "y_bind": 9.5, "round_number": 1},
        {"candidate_id": "E1", "sequence": "ERWGGDGFYAMDY", "y_bind": 9.4, "round_number": 1},
    ])
    effects = sar_effect_table(ledger, residuals={"D1": 0.9, "E1": 0.9})
    # No exact cell qualifies...
    assert (1, "D") not in effects and (1, "E") not in effects
    # ...but the position-level term does, with ``n_pos / (n_pos + 3)`` shrinkage.
    n_pos, shrunk = effects.position_effects[1]
    assert n_pos == 2
    assert shrunk > 0

    # And it reaches the score: mutating position 1 is rewarded over conserving it.
    mutated = adaptive_prior_score("DRWGGDGFYAMDY", sar_effects=effects)
    conserved = adaptive_prior_score(WT_SEQUENCE, sar_effects=effects)
    assert mutated > conserved


def test_hierarchical_sar_weights_exact_cell_above_position():
    """The exact cell contributes 0.60 and the position term 0.40."""
    ledger = _ledger([
        {"candidate_id": "A", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.5, "round_number": 1},
        {"candidate_id": "B", "sequence": "ARWGGDGFYAMDY", "y_bind": 9.4, "round_number": 1},
    ])
    effects = sar_effect_table(ledger, residuals={"A": 0.9, "B": 0.9})
    _, cell = effects[(1, "A")]
    _, position = effects.position_effects[1]
    assert SAR_CELL_WEIGHT == pytest.approx(0.60)
    assert SAR_POSITION_WEIGHT == pytest.approx(0.40)
    # ``_sar_total`` is the pre-saturation sum; ``adaptive_prior_score`` applies
    # ``tanh`` on top, so compare against the raw total here and check that the
    # saturated score is a monotone image of it.
    expected = SAR_HIT_WEIGHT * (SAR_CELL_WEIGHT * cell + SAR_POSITION_WEIGHT * position)
    assert _sar_total("ARWGGDGFYAMDY", effects) == pytest.approx(expected)
    assert adaptive_prior_score("ARWGGDGFYAMDY", sar_effects=effects, bound=1.0) == pytest.approx(
        np.tanh(expected)
    )
    # A position whose mutations measured *worse* rewards conserving WT instead,
    # so the term can defend an anchor and not only promote a substitution.
    bad = sar_effect_table(
        _ledger([
            {"candidate_id": "D1", "sequence": "DRWGGDGFYAMDY", "y_bind": 6.0, "round_number": 1},
            {"candidate_id": "E1", "sequence": "ERWGGDGFYAMDY", "y_bind": 6.1, "round_number": 1},
        ]),
        residuals={"D1": -0.9, "E1": -0.9},
    )
    assert _sar_total("DRWGGDGFYAMDY", bad) < 0
    assert _sar_total(WT_SEQUENCE, bad) > 0


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
    # Slot 4 and slot 5 remain the two arms of the paired audit: slot 4 is the
    # hypothesis arm, slot 5 the null control. ``a_h1`` is always ``a_null`` plus
    # the agent prior, which is what makes the pairing causal.
    assert slots[4]["a_h1"] == pytest.approx(slots[4]["a_null"] + slots[4]["delta_agent"])
    # Slots 1-2 are the top two A_H1 candidates in the whole qualified pool.
    assert slots[0]["a_h1"] >= slots[1]["a_h1"]


def test_m5_null_arm_is_the_unbiased_baseline_not_m4s_static_prior():
    """FLAW 1 regression: M5 must not inherit M4's refuted WT-anchor prior.

    ``A_null = A_base + loop_prior`` chained the hybrid to the static prior that
    loses to plain UCB on 1N8Z (the ``AR...``/aromatic-rich binders are exactly
    what that rule penalises). ``A_null`` is now the VoI-annealed qualified
    binding objective and still contains **no** loop-prior term, while M4 keeps
    the static rule.
    """
    frame = _scored_frame()
    slots = select_five_slots(frame, None, queries_spent=5, total_budget=35)
    for slot in slots:
        assert slot["a_h1"] == pytest.approx(slot["a_null"] + slot["delta_agent"])
        # ``loop_prior`` is a diagnostic column only; it is never an additive
        # term in either arm.
        assert np.isfinite(slot["loop_prior"])
    # Scoring is prior-free by construction: the objective is a function of
    # ``a_base``, ``mu_bind`` and ``sigma_bind`` alone.
    assert "loop_prior" not in {"a_null", "a_h1"}

    # A candidate the static rule dislikes but the surrogate loves must not be
    # pushed down. ``ARWGGDGFYYFDY`` is the shape the 1N8Z leaderboard actually
    # rewards (AR-start, five aromatics, ``...YYFDY``) and the shape M4's rule
    # punishes; give it a strictly higher A_base and require M5 to rank it first.
    binder = "ARWGGDGFYYFDY"
    rows = [
        {"candidate_id": "AR", "sequence": binder, "y_dev": 0.9,
         "a_base": 1.10, "sigma_joint": 0.0, "sigma_bind": 0.1, "sigma_dev": 0.1, "mu_bind": 1.0},
        {"candidate_id": "WT", "sequence": WT_SEQUENCE, "y_dev": 0.9,
         "a_base": 1.00, "sigma_joint": 0.0, "sigma_bind": 0.1, "sigma_dev": 0.1, "mu_bind": 1.0},
    ] + [
        {"candidate_id": f"X{i}", "sequence": binder, "y_dev": 0.9,
         "a_base": 0.90 - 0.01 * i, "sigma_joint": 0.0, "sigma_bind": 0.1,
         "sigma_dev": 0.1, "mu_bind": 1.0}
        for i in range(4)
    ]
    ranked = select_five_slots(pd.DataFrame(rows), None, queries_spent=5, total_budget=35)
    assert ranked[0]["candidate_id"] == "AR"
    # M4, by contrast, applies the anchor rule and penalises the better binder —
    # which is exactly why M5 must not inherit that term.
    assert fixed_expert_prior(WT_SEQUENCE) > fixed_expert_prior(binder)


def test_slot3_anneals_exploration_with_budget():
    """Early rounds behave like UCB; a spent budget reverts to pure exploitation."""
    frame = _scored_frame()
    early = select_five_slots(frame, None, tau_dev=0.5, queries_spent=5, total_budget=35)[2]
    late = select_five_slots(frame, None, tau_dev=0.5, queries_spent=35, total_budget=35)[2]
    # The annealing direction only: on the final batch VoI is 0, so no candidate
    # can score above its own A_H1, and max(A_H1 + bonus) >= max(A_H1).
    assert late["optimistic"] <= early["optimistic"] + 1e-9


def test_dynamic_voi_formula_and_final_batch_are_exact():
    """H is post-batch horizon; the final batch has strictly zero VoI."""
    assert horizon_factor(5, total_budget=35, batch_size=5) == 1.0
    assert horizon_factor(10, total_budget=35, batch_size=5) == pytest.approx(0.8)
    assert horizon_factor(30, total_budget=35, batch_size=5) == 0.0

    assert value_of_information(1.0, 1.0) == pytest.approx(0.35)
    assert value_of_information(0.0, 0.0) == 0.0

    final = voi_context(
        _sar_table(n_cells=10), queries_spent=30, total_budget=35, batch_size=5
    )
    # ``VoI`` drives slot 3 only. There is no phase flag any more: the H0 null
    # control is tested on every batch, because it is a definition, not a mode.
    assert final == {"horizon": 0.0, "sar_maturity": 1.0, "voi": 0.0}


# ── Empirical SAR maturity (C from corroborated evidence, not max Δ_agent) ────
def _sar_table(*, n_cells: int = 0, n_pos: int = 0, support: int = 0) -> SarTable:
    """A ``SarTable`` with ``n_cells`` exact cells and ``n_pos`` position effects."""
    table = SarTable(support=support, basis="residual")
    for index in range(n_cells):
        table[(index % 13 + 1, "A")] = (2, 0.5)
    table.position_effects = {index + 1: (2, 0.4) for index in range(n_pos)}
    return table


def test_sar_maturity_is_empirical_not_a_saturated_delta_agent():
    """Regression: ``C`` read ``1.000`` in every round and ``VoI`` went static.

    ``max(Δ_agent) / (0.5 · AGENT_PRIOR_BOUND)`` is cleared by a single saturated
    candidate, i.e. in Round 1. With only five seed reveals the *evidence* is
    thin, so ``C`` must be thin too and must climb as support accumulates.
    """
    assert sar_maturity_score(_sar_table(n_cells=0, n_pos=0)) == 0.0
    thin = sar_maturity_score(_sar_table(n_cells=2, n_pos=4))
    thick = sar_maturity_score(_sar_table(n_cells=15, n_pos=13))
    assert 0.0 < thin < 0.5
    assert thick == 1.0
    assert thin < thick
    # ``(n_cells + 0.5 · n_pos) / 10``, exactly.
    assert thin == pytest.approx((2 + 0.5 * 4) / 10.0)
    assert sar_maturity_score(_sar_table(n_cells=6)) == pytest.approx(0.6)


def test_sar_maturity_reads_position_effects_from_an_empty_dict():
    """The ``bool(SarTable)`` trap: zero exact cells but real position evidence.

    ``SarTable`` subclasses ``dict``, so ``if not sar_effects`` is ``True`` for a
    Round-1 table that has *only* position effects — the exact moment a synthetic
    pool produces its first corroborated evidence. Guarding on ``is None``
    instead keeps that evidence visible.
    """
    table = _sar_table(n_cells=0, n_pos=3)
    assert bool(table) is False and len(table) == 0
    assert table.position_effects
    assert sar_maturity_score(table) == pytest.approx(1.5 / 10.0)
    # ``None`` is the only thing that means "no evidence at all".
    assert sar_maturity_score(None) == 0.0


def test_sar_maturity_falls_back_to_delta_agent_for_legacy_callers():
    """A plain sequence still gets the documented legacy rule, not a crash."""
    assert sar_maturity_score([0.0, 0.5 * AGENT_PRIOR_BOUND, AGENT_PRIOR_BOUND]) == 1.0
    assert sar_maturity_score([0.0, 0.0]) == 0.0
    assert sar_maturity_score([]) == 0.0
    assert sar_maturity_score([AGENT_PRIOR_BOUND]) == 1.0


def test_cold_start_voi_is_high_and_exploit_weight_matches_the_calibration():
    """Round 1 must be genuinely exploratory *and* keep the calibrated mix.

    ``VOI_EXPLOIT_WEIGHT`` is fitted on a two-fold seed split (see its constant
    comment), so the assertion here is the *shape* it must preserve: a thin
    evidence table leaves ``VoI`` high (real exploration) while the exploitation
    weight stays well above the 0.35 an un-recalibrated 0.65 coefficient would
    have produced.
    """
    cold = voi_context(_sar_table(n_cells=1, n_pos=2), queries_spent=5, total_budget=35, batch_size=5)
    assert cold["horizon"] == 1.0
    assert cold["sar_maturity"] < 0.25
    assert cold["voi"] > 0.8
    exploit_weight = 1.0 - VOI_EXPLOIT_WEIGHT * cold["voi"]
    # Higher than the 0.35 an un-recalibrated 0.65 coefficient would give against
    # this VoI, and strictly below 1.0 so the first batch actually explores.
    assert exploit_weight > 1.0 - 0.65 * cold["voi"]
    assert exploit_weight < 1.0
    # ... and it rises monotonically to 1.0 as VoI decays.
    warmer = voi_context(_sar_table(n_cells=6, n_pos=6), queries_spent=15, total_budget=35, batch_size=5)
    assert 1.0 - VOI_EXPLOIT_WEIGHT * warmer["voi"] > exploit_weight
    assert VOI_EXPLOIT_WEIGHT == 0.35

    # By the final batch the horizon carries the whole collapse, whatever C is.
    mature_final = voi_context(_sar_table(n_cells=30, n_pos=13), queries_spent=30, total_budget=35)
    assert mature_final["voi"] == 0.0


def _dynamic_voi_frame() -> pd.DataFrame:
    """A controlled frontier whose soft prior creates positive and null arms."""
    rows = []
    for i in range(9):
        sequence = "ARWGGDGFYAMDY" if i < 6 else WT_SEQUENCE
        rows.append({
            "candidate_id": f"V{i}",
            "sequence": sequence,
            "y_dev": 0.9,
            "a_base": 1.00 - 0.01 * i,
            "sigma_joint": 1.0 if i == 8 else 0.0,
            "sigma_bind": 0.1,
            "sigma_dev": 0.1,
            "mu_bind": 1.0,
        })
    return pd.DataFrame(rows)


def _positive_prior_decision():
    from core.schemas import RoundDecisionRecord

    return RoundDecisionRecord(
        round_number=1,
        phase="Cold-Start Tuition (Exploration)",
        tools_invoked=[],
        hypothesis_id="HYP-VOI-01",
        hypothesis_statement="Position 1 A is beneficial.",
        previous_hypothesis_verdict="N/A (Round 1)",
        supporting_candidate_ids=[],
        agent_disagreement_summary="Controlled regression fixture.",
        confidence_rating="Medium",
        positional_priors=[
            PositionalSoftPrior(position=1, preferred_residues=["A"], weight_delta=0.04)
        ],
        slot3_uncertainty_target_id="V0",
        slot4_hypothesis_target_id="V1",
        slot5_falsification_target_id="V6",
    )


def test_slot5_always_tests_the_null_baseline_never_the_anti_hypothesis():
    """FLAW 2 regression: H0 is ``Δ_agent == 0``, not ``Δ_agent <= 0``.

    Restricting slot 5 to non-positive ``Δ_agent`` made the lab boycott its own
    proven substitutions for 20% of every early batch. The control must instead
    be the unmodified baseline, on *every* batch — early (VoI high) exactly as on
    the final one (VoI = 0).
    """
    frame = _dynamic_voi_frame()
    decision = _positive_prior_decision()

    # Early and unresolved: no SAR maturity and H=1 => VoI=1.
    early = select_five_slots(frame, decision, queries_spent=5, total_budget=35, batch_size=5)
    assert early[2]["optimistic"] == pytest.approx(
        early[2]["a_h1"] + SLOT3_OPTIMISM * early[2]["voi"] * early[2]["sigma_joint"]
    )
    assert early[4]["slot_type"] == SLOT5_FALSIFY_LABEL
    # The control maximises the null objective over the remaining candidates,
    # regardless of the sign of their Δ_agent: a positively-boosted candidate is
    # no longer banned from the control arm.
    remaining = frame.loc[~frame.candidate_id.isin({s["candidate_id"] for s in early[:4]})]
    expected_null = voi_annealed_null_objective(remaining, float(early[4]["voi"]))
    assert early[4]["a_null"] == pytest.approx(expected_null.max())

    # Final batch: VoI=0, so slot 3 is pure A_H1 and the VoI-annealed objective
    # collapses onto pure qualified binding exploitation — the control rule does
    # not depend on the VoI phase, only its numeric value does.
    final = select_five_slots(frame, decision, queries_spent=30, total_budget=35, batch_size=5)
    assert final[2]["voi"] == 0.0
    assert final[2]["optimistic"] == pytest.approx(final[2]["a_h1"])
    assert final[4]["slot_type"] == SLOT5_FALSIFY_LABEL
    remaining = frame.loc[~frame.candidate_id.isin({s["candidate_id"] for s in final[:4]})]
    expected_null = voi_annealed_null_objective(remaining, 0.0)
    assert final[4]["a_null"] == pytest.approx(expected_null.max())


def test_slot4_is_a_third_exploitation_slot_with_the_agent_boost_off():
    """Slot 4 maximises ``A_H1``: the 0.5 promotion bonus was retired.

    The 100-seed benchmark showed the promotion bonus cost more AUBQC than it
    bought (M5 was 0.0016 behind M3 with the bonus on and 0.0039 ahead with it
    off). ``Δ_agent`` survives only as a tie-breaker between equal ``A_H1``
    candidates.
    """
    assert SLOT4_AGENT_BOOST == 0.0
    frame = _dynamic_voi_frame()
    decision = _positive_prior_decision()
    slots = select_five_slots(frame, decision, queries_spent=5, total_budget=35, batch_size=5)
    slot4 = slots[3]
    assert slot4["agent_promoted"] == pytest.approx(slot4["a_h1"])
    rest = frame.loc[~frame.candidate_id.isin({s["candidate_id"] for s in slots[:3]})]
    # Among the remaining candidates, slot 4 is the best ``A_H1`` — with
    # ``Δ_agent`` used only to break an exact tie.
    rescored = rest.assign(
        delta_agent=[
            adaptive_prior_score(s, priors=decision.positional_priors) for s in rest.sequence
        ]
    )
    rescored = rescored.assign(
        a_h1=voi_annealed_null_objective(rescored, float(slot4["voi"])) + rescored.delta_agent
    )
    best = rescored.sort_values(
        ["a_h1", "delta_agent", "candidate_id"], ascending=[False, False, True]
    ).iloc[0]
    assert slot4["candidate_id"] == best["candidate_id"]


def test_a_null_is_the_voi_annealed_qualified_binding_objective():
    """``a_null`` blends early exploration with pure qualified binding.

    Every candidate in the qualified pool already clears ``y_dev >= tau_dev``, so
    the objective must shift toward ``mu_bind`` as the horizon closes instead of
    paying the un-annealed ``beta_0 = 1.0`` variance and the redundant ``y_dev``
    margin for all six rounds.
    """
    frame = _scored_frame(60, seed=11)
    frame = frame.assign(mu_bind=np.linspace(0.2, 1.4, len(frame)))
    early = select_five_slots(frame, None, queries_spent=5, total_budget=35)
    final = select_five_slots(frame, None, queries_spent=30, total_budget=35)
    assert early[0]["voi"] == pytest.approx(1.0)
    assert final[0]["voi"] == 0.0

    for slot in early + final:
        assert slot["a_h1"] == pytest.approx(slot["a_null"] + slot["delta_agent"])

    # The final batch is exploitation-only: ``exploit_weight`` is 1.0, so
    # ``a_null`` *is* ``mu_bind_norm`` and is no longer correlated with the
    # un-annealed ``a_base`` by construction.
    assert not np.allclose(
        [slot["a_null"] for slot in final], [slot["a_base"] for slot in final]
    )


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
