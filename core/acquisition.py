"""Bounded soft priors and deterministic five-slot batch selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .constants import SOFT_PRIOR_BOUND
from .schemas import PositionalSoftPrior, RoundDecisionRecord


def soft_prior_score(sequence: str, priors: list[PositionalSoftPrior]) -> float:
    contributions = []
    for prior in priors:
        residue = sequence[prior.position - 1]
        if residue in prior.preferred_residues:
            contributions.append(abs(prior.weight_delta))
        elif residue in prior.discouraged_residues:
            contributions.append(-abs(prior.weight_delta))
    return float(np.clip(np.mean(contributions) if contributions else 0.0, -SOFT_PRIOR_BOUND, SOFT_PRIOR_BOUND))


def fixed_expert_prior(sequence: str) -> float:
    from .constants import WT_SEQUENCE, ANCHOR_POSITIONS
    conserved = sum(sequence[p - 1] == WT_SEQUENCE[p - 1] for p in ANCHOR_POSITIONS) / len(ANCHOR_POSITIONS)
    aromatic_penalty = max(0, sum(a in "FWY" for a in sequence) - 3)
    return float(np.clip(0.15 * conserved - 0.05 * aromatic_penalty, -0.15, 0.15))


def select_five_slots(unqueried: pd.DataFrame, decision: RoundDecisionRecord | None = None) -> list[dict]:
    if len(unqueried) < 5:
        raise ValueError("At least five unqueried candidates are required")
    frame = unqueried.copy()
    frame["delta_agent"] = [soft_prior_score(s, decision.positional_priors) if decision else 0.0 for s in frame.sequence]
    frame["a_agent"] = frame.a_base + frame.delta_agent
    selected: list[dict] = []

    def take(row: pd.Series, slot_type: str) -> None:
        selected.append({"slot_id": len(selected) + 1, "slot_type": slot_type, **row.to_dict()})

    available = frame.sort_values(["a_base", "candidate_id"], ascending=[False, True])
    for _, row in available.head(2).iterrows():
        take(row, "Dual-Surrogate UCB")
    used = {row["candidate_id"] for row in selected}
    uncertainty = frame[~frame.candidate_id.isin(used)].sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).iloc[0]
    take(uncertainty, "Max Joint Uncertainty"); used.add(uncertainty.candidate_id)
    hypotheses = frame[~frame.candidate_id.isin(used)].sort_values(["a_agent", "candidate_id"], ascending=[False, True])
    if decision and decision.slot4_hypothesis_target_id in set(hypotheses.candidate_id):
        hypothesis = hypotheses[hypotheses.candidate_id == decision.slot4_hypothesis_target_id].iloc[0]
    else:
        hypothesis = hypotheses.iloc[0]
    take(hypothesis, "Agent Hypothesis Test"); used.add(hypothesis.candidate_id)
    remaining = frame[~frame.candidate_id.isin(used)]
    violating = remaining[remaining.delta_agent < 0]
    probes = violating if not violating.empty else remaining
    if decision and decision.slot5_falsification_target_id in set(probes.candidate_id):
        probe = probes[probes.candidate_id == decision.slot5_falsification_target_id].iloc[0]
    else:
        probe = probes.sort_values(["a_base", "candidate_id"], ascending=[False, True]).iloc[0]
    take(probe, "Skeptic Falsification Probe")
    return selected
