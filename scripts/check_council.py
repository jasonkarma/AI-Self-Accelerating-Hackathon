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

    header = f"{'ROLE':26} {'PROVIDER':10} {'MODEL':18} {'LIVE':5} {'THINK':>6}  BASE_URL"
    print(header)
    print("-" * len(header))
    for row in roster:
        budget = budgets[row["role"]]["thinking_budget"]
        print(
            f"{row['role']:26} {row['provider']:10} {row['model']:18} "
            f"{str(row['live']):5} {str(budget):>6}  {row['base_url'] or '-'}"
        )

    distinct = len({row["model"] for row in roster})
    offline = [row["role"] for row in roster if not row["live"]]
    print()
    print(f"distinct models: {distinct}/{len(roster)}")
    print(f"extended thinking: {[r for r, b in budgets.items() if b['extended_thinking']]}")
    if offline:
        print(f"offline (deterministic fallback): {offline}")
        print(json.dumps({"status": "degraded", "offline": offline}, indent=2))
        return 1
    print(json.dumps({"status": "live", "distinct_models": distinct}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
