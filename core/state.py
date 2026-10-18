"""Exact API-contract state builders."""
from __future__ import annotations
from .constants import WT_SEQUENCE

def empty_state(benchmark: dict | None = None) -> dict:
    # SPEC §7: the contract's required trio (m1/m3/m5) plus m2/m4 for the UI.
    # Missing keys degrade to None rather than raising, so a partially written
    # benchmark artifact cannot crash ``GET /api/state``.
    curves = []
    for row in (benchmark or {}).get("curves", []):
        curves.append({
            key: row.get(key)
            for key in ("query_count", "m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")
        })
    return {"current_round": 0, "phase": "Cold-Start Tuition", "kpis": {"best_qualified_binding": 0.0, "delta_vs_wt": 0.0, "query_budget_spent": 0, "total_budget": 35, "wet_lab_cost_saved_usd": 0, "hypotheses_supported": 0, "hypotheses_falsified": 0},
            "hcdr3_ribbon": [{"position": p, "wt_residue": WT_SEQUENCE[p-1], "soft_prior_weight": 0.0, "status": "anchor_conserved" if p in (3,11,12,13) else "neutral"} for p in (1,3,6,8,11,12,13)],
            "agent_council": [], "batch_slots": [], "pareto_data": [], "benchmark_curves": curves}
