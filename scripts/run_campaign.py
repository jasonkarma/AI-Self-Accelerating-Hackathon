#!/usr/bin/env python3
"""Run the closed-loop campaign from the terminal with live council output.

    python scripts/run_campaign.py --rounds 1

What this script guarantees, unlike the API-server entrypoint:

* ``M5_POLICY_MODE=live_agent`` is forced *before* ``config.settings`` is
  imported, so the ledger is stamped with :class:`core.baselines.LiveAgentPolicy`
  rather than the deterministic ``adaptive_tool`` baseline.
* Rounds are actually executed (``Campaign.execute_round``), not just the t=0
  state that used to be dumped.
* Each scientist's name, thinking trace, and memo is streamed to the terminal as
  it finishes, plus tool-output summaries and planner events.
* The final ``RoundDecisionRecord`` is printed as validated Pydantic JSON.

Exit code is non-zero when a seat silently fell back to the deterministic
reasoner, unless ``--allow-offline`` is passed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

# ── Force live-agent policy before config.settings evaluates its defaults ──────
# ``Settings.m5_policy_mode`` reads the environment at class-definition time, so
# this assignment must happen before any ``config``/``core`` import.
os.environ["M5_POLICY_MODE"] = "live_agent"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.server import initialize  # noqa: E402
from core.baselines import LiveAgentPolicy  # noqa: E402
from core.schemas import RoundDecisionRecord  # noqa: E402
from config.settings import SETTINGS  # noqa: E402


# ── Terminal formatting (no third-party dependency) ───────────────────────────
class Console:
    """Minimal ANSI writer; degrades to plain text when piped or NO_COLOR is set."""

    _CODES = {
        "dim": "\033[2m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "grey": "\033[90m",
    }
    _RESET = "\033[0m"

    def __init__(self, colour: bool = True) -> None:
        self.enabled = colour and sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self.width = max(60, min(120, shutil.get_terminal_size((100, 24)).columns))

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(self._CODES.get(s, "") for s in styles)
        return f"{prefix}{text}{self._RESET}"

    def rule(self, char: str = "─") -> str:
        return self.paint(char * self.width, "grey")

    def banner(self, text: str) -> None:
        print()
        print(self.rule("═"))
        print(self.paint(f" {text}", "bold"))
        print(self.rule("═"))

    def section(self, text: str) -> None:
        print()
        print(self.paint(f"── {text} ", "bold", "cyan") + self.paint("─" * max(0, self.width - len(text) - 4), "grey"))

    def kv(self, key: str, value: Any) -> None:
        print(f"  {self.paint(key + ':', 'dim'):<24} {value}")

    def wrap(self, text: str, indent: str = "    ") -> None:
        if not text or not text.strip():
            print(f"{indent}{self.paint('(empty)', 'grey')}")
            return
        for line in text.splitlines() or [""]:
            wrapped = textwrap.wrap(line, width=self.width - len(indent)) or [""]
            for chunk in wrapped:
                print(f"{indent}{chunk}")


CONSOLE = Console()


# ── Event rendering ───────────────────────────────────────────────────────────
def _tool_summary(output: dict) -> str:
    """One-line digest of a deterministic tool result."""
    if not isinstance(output, dict):
        return str(output)[:80]
    for key in ("rows", "clashes", "verdicts"):
        value = output.get(key)
        if isinstance(value, list):
            return f"{len(value)} {key}"
    if "wt_binding" in output:
        return f"wt_binding={output['wt_binding']}"
    if "error" in output:
        return f"ERROR {output['error']}"
    return "ok"


def make_renderer(*, verbose: bool = False):
    """Build the SSE-event -> terminal renderer used as the runner's listener."""
    seen: set[tuple[str, str]] = set()

    def render(message: dict[str, Any]) -> None:
        event, data = message["event"], message["data"]

        if event == "round_started":
            CONSOLE.banner(f"ROUND {data['round']} · {data['phase']}")
            return

        if event == "tool_output":
            name = data.get("tool", "?")
            if verbose:
                CONSOLE.section(f"TOOL · {name}")
                CONSOLE.wrap(_tool_summary(data.get("output", {})))
            else:
                print(f"  {CONSOLE.paint('tool', 'grey')} {name:<38} {CONSOLE.paint(_tool_summary(data.get('output', {})), 'dim')}")
            return

        if event == "agent_memo":
            role = data.get("role", "?")
            model = data.get("model") or "?"
            provider = data.get("provider") or "?"
            budget = data.get("thinking_budget")
            effort = data.get("reasoning_effort")
            deliberation = (
                f"thinking={budget}" if budget else (f"effort={effort}" if effort else "direct")
            )
            key = (role, str(data.get("memo", ""))[:120])
            if key in seen:  # guard against replay
                return
            seen.add(key)

            CONSOLE.section(f"{role.upper()}  {CONSOLE.paint(f'{provider}/{model}', 'magenta')}  {CONSOLE.paint(deliberation, 'yellow')}")
            trace = data.get("thinking_trace") or ""
            memo = data.get("memo") or ""
            if trace and trace != memo:
                print(f"  {CONSOLE.paint('thinking_trace', 'dim')}")
                CONSOLE.wrap(trace)
            print(f"  {CONSOLE.paint('memo', 'dim')}")
            CONSOLE.wrap(CONSOLE.paint(memo, "bold"))
            return

        if event == "planner_started":
            CONSOLE.section(f"{data.get('role', 'PLANNER').upper()} · synthesising decision")
            return

        if event == "agent_error":
            print(
                f"  {CONSOLE.paint('AGENT ERROR', 'red', 'bold')} "
                f"{data.get('role')}: {data.get('error')}"
            )
            return

        if event == "planner_error":
            print(f"  {CONSOLE.paint('PLANNER ERROR', 'red', 'bold')} {data.get('error')}")
            print(f"  {CONSOLE.paint('→ falling back to', 'dim')} {data.get('fallback')}")
            return

        if event == "planner_reconciled":
            print(f"  {CONSOLE.paint('reconciled', 'grey')} {data.get('note')}")
            return

        if event == "planner_synthesis":
            CONSOLE.section("ROUND DECISION (RoundDecisionRecord)")
            try:
                decision = RoundDecisionRecord.model_validate(data)
                print(decision.model_dump_json(indent=2))
            except Exception as exc:  # noqa: BLE001 - surface, never crash the round
                print(f"  {CONSOLE.paint('invalid decision payload', 'red')}: {exc}")
                print(data)
            return

        if event == "slot_selected":
            print(
                f"  {CONSOLE.paint('slot', 'grey')} {data.get('slot_id')} "
                f"{data.get('slot_type', ''):<28} {data.get('candidate_id')} "
                f"{CONSOLE.paint(str(data.get('sequence', '')), 'cyan')}"
            )
            return

        if event == "oracle_reveal":
            print(
                f"    {CONSOLE.paint('reveal', 'grey')} {data.get('candidate_id')} "
                f"y_bind={data.get('y_bind')} y_dev={data.get('y_dev')}"
            )
            return

        if event == "round_complete":
            state = data.get("state", {})
            kpis = state.get("kpis", {})
            CONSOLE.section("ROUND COMPLETE")
            CONSOLE.kv("queried", data.get("round_summary", {}).get("queried"))
            CONSOLE.kv("best qualified binding", kpis.get("best_qualified_binding"))
            CONSOLE.kv("delta vs WT", kpis.get("delta_vs_wt"))
            CONSOLE.kv("budget spent", f"{kpis.get('query_budget_spent')}/{kpis.get('total_budget')}")
            return

        if event == "stream_end":
            return

        if verbose:
            print(f"  {CONSOLE.paint(event, 'grey')} {data}")

    return render


# ── Preflight ─────────────────────────────────────────────────────────────────
def preflight(router, *, allow_offline: bool) -> list[dict]:
    roster = router.describe_roster()
    CONSOLE.banner("COUNCIL PREFLIGHT")
    header = f"  {'ROLE':26} {'REQUESTED':30} {'SERVING':30} {'LIVE':6} DELIBERATION"
    print(CONSOLE.paint(header, "dim"))
    for row in roster:
        requested = f"{row['provider']}/{row['model']}"
        serving = (
            f"{row['effective_provider']}/{row['effective_model']}"
            if row["live"]
            else "offline (deterministic)"
        )
        marker = " *" if row["fallback_used"] else ""
        budget = row.get("thinking_budget")
        effort = row.get("reasoning_effort")
        deliberation = f"thinking={budget}" if budget else (f"effort={effort}" if effort else "direct")
        colour = "green" if row["live"] else "red"
        serving_cell = f"{(serving + marker)[:30]:30}"
        print(
            f"  {row['role']:26} {requested:30} "
            f"{CONSOLE.paint(serving_cell, colour)} "
            f"{str(row['live']):6} {deliberation}"
        )

    offline = [row["role"] for row in roster if not row["live"]]
    distinct = len({row["effective_model"] for row in roster if row["live"]})
    print()
    print(f"  distinct live models: {distinct}/{len(roster)}")
    if offline:
        print(f"  {CONSOLE.paint('OFFLINE:', 'red', 'bold')} {offline}")
        print(f"  {CONSOLE.paint('These seats will answer from the deterministic stub — no HTTP calls.', 'red')}")
        if not allow_offline:
            raise SystemExit(
                "Refusing to run a 'live agent test' with offline seats. "
                "Fix credentials/base URLs in .env, or pass --allow-offline."
            )
    rehomed = [row["role"] for row in roster if row["fallback_used"]]
    if rehomed:
        print(f"  {CONSOLE.paint('re-homed:', 'yellow')} {rehomed}")
    return roster


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-accelerating virtual-lab campaign.")
    parser.add_argument("--rounds", type=int, default=1, help="number of rounds to execute (default: 1)")
    parser.add_argument("--allow-offline", action="store_true", help="run even if seats fall back to the stub")
    parser.add_argument("--verbose", "-v", action="store_true", help="print full tool outputs and unknown events")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    global CONSOLE
    CONSOLE = Console(colour=not args.no_color)

    campaign = initialize()
    router = campaign.runner.router

    CONSOLE.banner("SELF-ACCELERATING VIRTUAL LAB · LIVE CAMPAIGN")
    CONSOLE.kv("ledger", campaign.ledger.path)
    CONSOLE.kv("m5 policy (requested)", os.getenv("M5_POLICY_MODE"))
    CONSOLE.kv("m5 policy (stamped)", campaign.ledger.metadata().get("m5_policy_mode"))
    CONSOLE.kv("agent policy class", LiveAgentPolicy.__name__)
    CONSOLE.kv("current round", campaign.state["current_round"])
    CONSOLE.kv("budget", f"{SETTINGS.total_budget} queries ({SETTINGS.initial_queries} seed + {SETTINGS.rounds}×{SETTINGS.batch_size})")

    if SETTINGS.m5_policy_mode != LiveAgentPolicy.mode:
        print(f"  {CONSOLE.paint('WARNING', 'red', 'bold')}: settings resolved to {SETTINGS.m5_policy_mode!r}, expected {LiveAgentPolicy.mode!r}")
    if campaign.state["current_round"] > 0:
        print(
            f"  {CONSOLE.paint('NOTE', 'yellow')}: ledger already holds "
            f"{campaign.state['current_round']} round(s); continuing from there. "
            "Delete results/campaign.sqlite3 for a fresh campaign."
        )

    preflight(router, allow_offline=args.allow_offline)

    campaign.runner.events.subscribe_sync(make_renderer(verbose=args.verbose))

    requested = min(args.rounds, SETTINGS.rounds - campaign.state["current_round"])
    if requested <= 0:
        print("\nCampaign already completed; nothing to run.")
        return 0

    for index in range(requested):
        remaining = requested - index
        print()
        print(CONSOLE.paint(f"▶ executing round {campaign.state['current_round'] + 1} ({remaining} in this invocation)", "bold", "green"))
        try:
            await campaign.execute_round()
        except RuntimeError as exc:
            print(f"{CONSOLE.paint('ABORTED', 'red', 'bold')}: {exc}")
            return 1

    CONSOLE.banner("CAMPAIGN STATE")
    kpis = campaign.state["kpis"]
    CONSOLE.kv("current round", campaign.state["current_round"])
    CONSOLE.kv("phase", campaign.state["phase"])
    CONSOLE.kv("best qualified binding", kpis["best_qualified_binding"])
    CONSOLE.kv("delta vs WT", kpis["delta_vs_wt"])
    CONSOLE.kv("queries spent", f"{kpis['query_budget_spent']}/{kpis['total_budget']}")
    CONSOLE.kv("hypotheses supported", kpis["hypotheses_supported"])
    CONSOLE.kv("hypotheses falsified", kpis["hypotheses_falsified"])
    CONSOLE.kv("ledger verified", campaign.ledger.verify())

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
