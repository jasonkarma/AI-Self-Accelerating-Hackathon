"""Pre-flight audit for the live agent test: prove cognitive diversity is real.

Run ``python -m scripts.check_council`` (or ``python scripts/check_council.py``)
before a live campaign.  It prints the pinned model per council seat, whether the
seat resolved a live transport (i.e. credentials were found), the effective base
URL, and the resolved thinking budget.  Exits non-zero if any seat silently fell
back to the deterministic reasoner, so an "echo chamber" run cannot be mistaken
for a multi-model run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # allow `python scripts/check_council.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agents import thinking_configuration  # noqa: E402
from core.llm import ModelRouter  # noqa: E402


def main() -> int:
    router = ModelRouter()
    roster = router.describe_roster()
    budgets = {row["role"]: row for row in thinking_configuration(router)}

    header = f"{'ROLE':26} {'REQUESTED':32} {'SERVING':32} {'LIVE':5} {'THINK':>6}  BASE_URL"
    print(header)
    print("-" * len(header))
    for row in roster:
        budget = budgets[row["role"]]["thinking_budget"]
        requested = f"{row['provider']}/{row['model']}"
        serving = f"{row['effective_provider']}/{row['effective_model']}" if row["live"] else "offline"
        marker = " *" if row["fallback_used"] else ""
        print(
            f"{row['role']:26} {requested:32} {serving + marker:32} "
            f"{str(row['live']):5} {str(budget):>6}  {row['base_url'] or '-'}"
        )

    distinct = len({row["effective_model"] for row in roster if row["live"]})
    offline = [row["role"] for row in roster if not row["live"]]
    rehomed = [row["role"] for row in roster if row["fallback_used"]]
    print()
    print(f"distinct live models: {distinct}/{len(roster)}  (* = served by a fallback)")
    print(f"extended thinking: {[r for r, b in budgets.items() if b['extended_thinking']]}")
    if rehomed:
        print(f"re-homed seats: {rehomed}")
    if offline:
        print(f"offline (deterministic): {offline}")
        print(json.dumps({"status": "degraded", "offline": offline, "rehomed": rehomed}, indent=2))
        return 1
    if distinct < 3:
        print("WARNING: fewer than 3 distinct models are answering — cognitive diversity is reduced.")
    print(json.dumps({"status": "live", "distinct_models": distinct, "rehomed": rehomed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
