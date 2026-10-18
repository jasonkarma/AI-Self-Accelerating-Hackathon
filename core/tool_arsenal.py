"""Six deterministic, JSON-serializable scientific tools."""
from __future__ import annotations

from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd

from .constants import WT_SEQUENCE
from .developability import raw_components

_CANDIDATE_LOOKUP: dict[str, str] = {}


def configure_candidate_lookup(mapping: dict[str, str]) -> None:
    global _CANDIDATE_LOOKUP
    _CANDIDATE_LOOKUP = dict(mapping)


def scan_positional_sar(ledger_df: pd.DataFrame) -> dict:
    needed = {"candidate_id", "sequence", "y_bind"}
    if ledger_df.empty or not needed <= set(ledger_df.columns):
        return {"tool": "scan_positional_sar", "rows": [], "wt_binding": None}
    wt = ledger_df.loc[ledger_df.sequence == WT_SEQUENCE, "y_bind"]
    reference = float(wt.mean()) if len(wt) else float(ledger_df.y_bind.mean())
    records = []
    for position in range(13):
        for residue, group in ledger_df[ledger_df.sequence.str[position] != WT_SEQUENCE[position]].groupby(ledger_df.sequence.str[position]):
            mean = float(group.y_bind.mean())
            records.append({"position": position + 1, "residue": residue, "n": int(len(group)),
                            "mean_y_bind": mean, "delta_vs_wt": mean - reference,
                            "std": float(group.y_bind.std(ddof=1)) if len(group) > 1 else 0.0,
                            "candidate_ids": sorted(group.candidate_id.astype(str).tolist())})
    return {"tool": "scan_positional_sar", "wt_binding": reference, "rows": records}


def detect_epistasis_clashes(ledger_df: pd.DataFrame) -> dict:
    if ledger_df.empty or not {"sequence", "y_bind"} <= set(ledger_df.columns):
        return {"tool": "detect_epistasis_clashes", "clashes": []}
    base = float(ledger_df.loc[ledger_df.sequence == WT_SEQUENCE, "y_bind"].mean()) if (ledger_df.sequence == WT_SEQUENCE).any() else float(ledger_df.y_bind.mean())
    effects: dict[int, list[float]] = {i: [] for i in range(13)}
    for row in ledger_df.itertuples():
        positions = [i for i, (a, b) in enumerate(zip(row.sequence, WT_SEQUENCE)) if a != b]
        if len(positions) == 1:
            effects[positions[0]].append(float(row.y_bind) - base)
    residuals = []
    for row in ledger_df.itertuples():
        positions = [i for i, (a, b) in enumerate(zip(row.sequence, WT_SEQUENCE)) if a != b]
        if len(positions) == 2 and all(effects[p] for p in positions):
            expected = base + sum(np.mean(effects[p]) for p in positions)
            residuals.append((row, positions, float(row.y_bind) - expected))
    sigma = float(np.std([r[2] for r in residuals], ddof=1)) if len(residuals) > 1 else 0.0
    threshold = 1.5 * sigma
    clashes = [{"candidate_id": str(r.candidate_id), "positions": [p + 1 for p in ps], "residual": residual,
                "direction": "positive" if residual > 0 else "negative"}
               for r, ps, residual in residuals if abs(residual) > threshold and (sigma > 0 or residual != 0)]
    return {"tool": "detect_epistasis_clashes", "sigma": sigma, "threshold": threshold, "clashes": clashes}


def profile_developability_liabilities(candidate_ids: Iterable[str]) -> dict:
    rows = []
    for candidate_id in candidate_ids:
        if candidate_id not in _CANDIDATE_LOOKUP:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        sequence = _CANDIDATE_LOOKUP[candidate_id]
        raw = raw_components(sequence)
        rows.append({"candidate_id": candidate_id, "sequence": sequence, "max_5mer_hydropathy": raw["window"],
                     "aromatic_fraction": raw["aromatic"], "net_charge_abs_fraction": raw["charge"],
                     "adjacent_charge_clusters": int(raw["cluster"]),
                     "deamidation_motifs": [m for m in ("NG", "NS") if m in sequence],
                     "isomerization_motifs": [m for m in ("DG", "DS") if m in sequence], "unpaired_cys": sequence.count("C")})
    return {"tool": "profile_developability_liabilities", "rows": rows}


def inspect_joint_uncertainty_hotspots(unqueried_df: pd.DataFrame, top_k: int = 15) -> dict:
    needed = {"candidate_id", "sigma_bind", "sigma_dev", "sigma_joint"}
    if not needed <= set(unqueried_df.columns):
        raise ValueError(f"Missing uncertainty columns: {sorted(needed - set(unqueried_df.columns))}")
    rows = unqueried_df.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).head(top_k)
    return {"tool": "inspect_joint_uncertainty_hotspots", "rows": rows[list(needed)].to_dict("records")}


SUPPORTED = "SUPPORTED"
FALSIFIED = "FALSIFIED"
INCONCLUSIVE = "INCONCLUSIVE"


def _measured_bindings(frame: pd.DataFrame | None) -> dict[str, float]:
    """candidate_id -> revealed y_bind, tolerating an empty or partial ledger."""
    if frame is None or getattr(frame, "empty", True) or not {"candidate_id", "y_bind"} <= set(frame.columns):
        return {}
    return {
        str(cid): float(value)
        for cid, value in zip(frame.candidate_id, frame.y_bind)
        if pd.notna(value)
    }


def _tested_positions(decision: dict, sequence: str | None) -> set[int]:
    """Positions whose declared rule the *hypothesis candidate actually exercised*.

    A round can only falsify a rule it tested. Inverting every position in the
    prior list would blacklist most of the CDR3 after two or three rounds (the
    deterministic fallback declares a rule at nearly every mutated position),
    which is not self-correction — it is self-destruction. A position counts as
    tested only when the Slot 4 candidate carries the exact residue that rule
    preferred, because that is the substitution whose effect the paired arms
    actually measure.
    """
    if not sequence:
        return set()
    tested: set[int] = set()
    for prior in decision.get("positional_priors") or []:
        if not isinstance(prior, dict):
            continue
        position = prior.get("position")
        if not position or not (1 <= int(position) <= len(sequence)):
            continue
        residue = sequence[int(position) - 1]
        if residue in (prior.get("preferred_residues") or []):
            tested.add(int(position))
    return tested


def audit_past_hypotheses(ledger_df: pd.DataFrame, previous_decisions: list[dict]) -> dict:
    """Grade each completed round as a paired H1-vs-H0 experiment.

    Each batch is a designed experiment: Slot 4 is the **H1 arm** (a candidate
    the agents' hypothesis actively boosts) and Slot 5 is the **H0 null control**
    (the best candidate the hypothesis does *not* boost). Grading those two
    against each other is what makes the verdict causal rather than descriptive
    — no cross-round median is needed, and the comparison is inside the batch.

    Rule order (deterministic):

    1. Slot 5 beats Slot 4 ⇒ ``FALSIFIED``. The null arm out-performed the
       hypothesis arm, so the prior is not earning its keep.
    2. Slot 4 beats Slot 5 ⇒ ``SUPPORTED``.
    3. Equal (or Slot 5 unmeasurable) ⇒ fall back to the round median, then to
       the stored pre-query surrogate expectation. This keeps a round decidable
       even when the paired arms happen to tie.

    The returned ``falsified_positions`` is the self-correction signal: the
    positions of the round's hypothesis that lost, which
    :func:`core.acquisition.select_five_slots` inverts next round so a wrong rule
    cannot persist for the rest of the campaign.

    ``slot4_measured_id`` / ``slot5_measured_id`` are the ids the quota actually
    queried (stamped by the campaign), not the planner's declared targets —
    which Slots 1-3 of the same batch routinely consume.
    """
    verdicts, falsified_positions = [], set()
    has_rounds = ledger_df is not None and "round_number" in getattr(ledger_df, "columns", ())
    # candidate_id -> sequence, used to decide which declared rules the Slot 4
    # candidate actually exercised. Sequences are free at t=0 (SPEC §2), so this
    # is not a label read.
    sequences: dict[str, str] = {}
    if ledger_df is not None and "sequence" in getattr(ledger_df, "columns", ()):
        sequences = dict(zip(ledger_df.candidate_id.astype(str), ledger_df.sequence.astype(str)))
    for decision in previous_decisions:
        round_no = decision.get("round_number")
        slot4 = decision.get("slot4_measured_id") or decision.get("slot4_hypothesis_target_id")
        slot5 = decision.get("slot5_measured_id") or decision.get("slot5_falsification_target_id")

        round_rows = ledger_df[ledger_df.round_number == round_no] if has_rounds else ledger_df
        measured = _measured_bindings(round_rows)
        if not measured:
            # Reveals can carry a different round tag (cached seed re-query);
            # judge against whatever the ledger has measured rather than
            # declaring the round untestable.
            measured = _measured_bindings(ledger_df)

        observed = measured.get(slot4) if slot4 else None
        control = measured.get(slot5) if slot5 else None
        median = float(np.median(list(measured.values()))) if measured else float("nan")
        expectations = decision.get("pre_query_expectation") or {}
        expected = expectations.get(slot4) if slot4 else None

        paired = observed is not None and control is not None
        if observed is None:
            verdict = INCONCLUSIVE
            reason = f"hypothesis candidate {slot4!r} was not measured this round"
        elif paired and control > observed:
            verdict = FALSIFIED
            reason = (f"H0 control {slot5} ({control:.3f}) beat H1 pick "
                      f"{slot4} ({observed:.3f})")
        elif paired and observed > control:
            verdict = SUPPORTED
            reason = (f"H1 pick {slot4} ({observed:.3f}) beat H0 control "
                      f"{slot5} ({control:.3f})")
        elif not np.isnan(median) and observed >= median:
            verdict = SUPPORTED
            reason = (f"paired arms tied; H1 pick {slot4} ({observed:.3f}) reached the "
                      f"round median ({median:.3f})")
        elif not np.isnan(median):
            verdict = FALSIFIED
            reason = (f"paired arms tied; H1 pick {slot4} ({observed:.3f}) fell below the "
                      f"round median ({median:.3f})")
        elif expected is not None:
            hit = observed >= float(expected)
            verdict = SUPPORTED if hit else FALSIFIED
            reason = (f"H1 pick {slot4} ({observed:.3f}) "
                      f"{'met' if hit else 'missed'} its pre-query expectation ({float(expected):.3f})")
        else:
            verdict = INCONCLUSIVE
            reason = "no control, median or pre-query expectation available"

        if verdict == FALSIFIED:
            falsified_positions |= _tested_positions(decision, sequences.get(slot4) if slot4 else None)

        verdicts.append({
            "hypothesis_id": decision.get("hypothesis_id"),
            "round_number": round_no,
            "verdict": verdict,
            "reason": reason,
            "design": "paired H1 (slot 4) vs H0 control (slot 5)",
            "h1_target_id": slot4,
            "h0_control_id": slot5,
            "round_median": None if np.isnan(median) else round(median, 3),
            "h1_binding": None if observed is None else round(observed, 3),
            "h0_binding": None if control is None else round(control, 3),
            "paired_lift": (
                None if observed is None or control is None else round(observed - control, 3)
            ),
            "expected_binding": None if expected is None else round(float(expected), 3),
        })
    return {
        "tool": "audit_past_hypotheses",
        "design": "paired H1 / H0",
        "verdicts": verdicts,
        "falsified_positions": sorted(falsified_positions),
    }


def compute_cost_and_effort_saved(ledger_df: pd.DataFrame, assay_cost_usd: int = 1500) -> dict:
    queried = int(ledger_df.candidate_id.nunique()) if "candidate_id" in ledger_df else 0
    full_pool = 419
    return {"tool": "compute_cost_and_effort_saved", "queries": queried,
            "spent_usd": queried * assay_cost_usd, "full_pool_cost_usd": full_pool * assay_cost_usd,
            "saved_vs_exhaustive_usd": max(0, (full_pool - queried) * assay_cost_usd),
            "assay_days_parallel_batches": int(np.ceil(queried / 5))}


TOOL_REGISTRY: dict[str, Callable] = {fn.__name__: fn for fn in (
    scan_positional_sar, detect_epistasis_clashes, profile_developability_liabilities,
    inspect_joint_uncertainty_hotspots, audit_past_hypotheses, compute_cost_and_effort_saved,
)}
