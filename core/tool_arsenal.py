"""Six deterministic, JSON-serializable scientific tools."""
from __future__ import annotations

from collections.abc import Callable, Iterable
import itertools
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


def audit_past_hypotheses(ledger_df: pd.DataFrame, previous_decisions: list[dict]) -> dict:
    verdicts = []
    for decision in previous_decisions:
        round_no = decision["round_number"]
        round_rows = ledger_df[ledger_df.round_number == round_no] if "round_number" in ledger_df else pd.DataFrame()
        slot4 = decision.get("slot4_hypothesis_target_id")
        slot5 = decision.get("slot5_falsification_target_id")
        measured = round_rows.set_index("candidate_id")["y_bind"] if not round_rows.empty else pd.Series(dtype=float)
        if slot4 not in measured:
            verdict = "INCONCLUSIVE"
        else:
            median = float(round_rows.y_bind.median())
            verdict = "SUPPORTED" if measured[slot4] > median else "FALSIFIED"
            if slot5 in measured and measured[slot5] > measured[slot4]:
                verdict = "FALSIFIED"
        verdicts.append({"hypothesis_id": decision.get("hypothesis_id"), "round_number": round_no, "verdict": verdict})
    return {"tool": "audit_past_hypotheses", "verdicts": verdicts}


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
