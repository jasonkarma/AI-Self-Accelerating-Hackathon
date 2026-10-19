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
import json
import math
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
from core.acquisition import VOI_EXPLOIT_WEIGHT  # noqa: E402
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

#: Methods shown beside the live run. ``m5_hybrid`` is the **benchmark** M5
#: (``AdaptiveToolPolicy`` driving the same portfolio), not the council — it is
#: labelled with the artifact's own seed count so it can never be mistaken for
#: the live number, which is reported separately as "live M5".
BASELINE_METHODS = ("m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")
BASE_BASELINE_LABELS = {
    "m1_random": "M1",
    "m2_greedy": "M2",
    "m3_ucb": "M3",
    "m4_fixed_prior": "M4",
    "m5_hybrid": "M5",
}


def baseline_labels(benchmark: dict | None) -> dict[str, str]:
    """M5's label carries the artifact's seed count, so it is never misread."""
    seeds = (benchmark or {}).get("seeds")
    suffix = f" ({seeds}-seed)" if seeds else ""
    return {**BASE_BASELINE_LABELS, "m5_hybrid": f"M5{suffix}"}


def live_seed_index(campaign, benchmark: dict | None) -> int | None:
    """Which benchmark seed the live campaign's Round-0 batch corresponds to.

    ``Campaign.execute_round`` seeds its five initial queries with
    ``np.random.default_rng(0)``, and ``run_benchmark`` uses
    ``np.random.default_rng(seed)`` for the same draw — so the live campaign *is*
    benchmark seed 0. The artifact records every seed's ``initial_ids``; matching
    them is far more trustworthy than assuming the index, because it fails closed
    if either side's seeding ever changes.
    """
    matched = (benchmark or {}).get("matched_initial_seeds") or {}
    revealed = campaign.revealed()
    if revealed.empty:
        return None
    seed_ids = set(revealed.loc[revealed.round_number == 0, "candidate_id"].astype(str))
    for seed, ids in matched.items():
        if set(map(str, ids)) == seed_ids:
            return int(seed)
    return None


def _live_seed_checkpoint(campaign) -> float:
    """The live campaign's own ``y*(5)`` — its Round-0 seed batch score."""
    revealed = campaign.revealed()
    if revealed.empty:
        return float("nan")
    seed = revealed[revealed.round_number == 0]
    qualified = seed[seed.y_dev >= campaign.protocol["tau_dev"]]
    pool = qualified if len(qualified) else seed
    return float(pool.y_bind.max()) if len(pool) else float("nan")


def seed_matched_baselines(campaign, benchmark: dict | None, seed: int) -> dict | None:
    """M1-M5 numbers for the *live campaign's own seed*.

    A 100-seed median answers "does the design work on average?". It cannot
    attribute this campaign's result: a seed whose five random probes contain no
    qualified binder starts from a lower ``y*(5)`` and has more headroom, so the
    single run and the median are not the same quantity. Re-running the five
    matched baselines on the identical seed costs a couple of seconds and makes
    the comparison like-for-like.
    """
    if benchmark is None:
        return None
    try:
        from core.baselines import METHODS, run_method
        from core.developability import RiskNormalizer
        from core.preregistration import load_protocol
        from data.datasets import load_primary

        frame, synthetic = load_primary()
        protocol = campaign.protocol or load_protocol()
        normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
        initial = [str(cid) for cid in (benchmark.get("matched_initial_seeds") or {}).get(str(seed), [])]
        if len(initial) != 5:
            return None
        # ``seed_floor`` stays at its default: these five methods share the same
        # Round-0 batch, so their normalized curves have the same 0% mark anyway.
        results = {
            method: run_method(frame, normalizer, protocol["tau_dev"], method, seed, initial)
            for method in METHODS
        }
    except Exception as exc:  # noqa: BLE001 - a report must never fail the run
        print(f"  {CONSOLE.paint(f'same-seed baselines unavailable: {exc}', 'yellow')}")
        return None
    return {
        method: {
            "q5": float(next(p["best_qualified_binding"] for p in result.curve if p["query_count"] == 5)),
            "q35": float(next(p["best_qualified_binding"] for p in result.curve if p["query_count"] == 35)),
            "aubqc": float(result.aubqc),
        }
        for method, result in results.items()
    }


def load_benchmark(path: Path | None = None) -> dict | None:
    """Read the frozen benchmark artifact; ``None`` when it is absent or stale."""
    path = path or SETTINGS.benchmark_path
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt(value: Any, width: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"
    return f"{'n/a' if not math.isfinite(number) else f'{number:.3f}':>{width}}"


def print_voi_trajectory(campaign) -> None:
    """Per-round ``VoI`` / ``sar_maturity`` trajectory from the ledger.

    Read from the persisted ``round_slots`` events rather than from
    ``state["voi_diagnostics"]``, which only holds the *last* batch. The whole
    point of the empirical ``C`` is that it is a trajectory — thin in Round 1,
    rising as corroborated SAR accumulates — and a single final number cannot
    show that. The ``CELLS``/``POS`` columns are the evidence ``C`` was computed
    from, read from the same revealed ledger the selector saw.
    """
    from core.acquisition import sar_effect_table

    with campaign.ledger.connect() as con:
        rows = con.execute(
            "SELECT round_number, payload FROM entries WHERE round_number > 0 "
            "AND event_type='round_slots' ORDER BY sequence_no"
        ).fetchall()

    CONSOLE.section("VOI / SAR MATURITY TRAJECTORY")
    if not rows:
        print(f"  {CONSOLE.paint('no round_slots events in the ledger', 'yellow')}")
        return

    revealed = campaign.revealed()
    tau_dev = campaign.protocol["tau_dev"]
    print(CONSOLE.paint(
        f"  {'ROUND':6} {'q':>4} {'VoI':>7} {'H':>6} {'C':>7} {'EXPLOIT':>8} {'CELLS':>6} {'POS':>4}",
        "dim",
    ))
    seen: set[int] = set()
    for row in rows:
        round_number = int(row["round_number"])
        slots = (json.loads(row["payload"]).get("slots") or [])
        if not slots or round_number in seen:
            continue
        seen.add(round_number)
        lead = slots[0]
        voi = float(lead.get("voi") or 0.0)
        # The batch is designed from the reveals *before* it, i.e. rounds < it.
        before = revealed[revealed.round_number < round_number]
        effects = sar_effect_table(before, tau_dev=tau_dev)
        print(
            f"  R{round_number:<5} {int(len(before)):>4} {voi:>7.4f} "
            f"{float(lead.get('horizon') or 0.0):>6.2f} {float(lead.get('sar_maturity') or 0.0):>7.4f} "
            f"{1.0 - VOI_EXPLOIT_WEIGHT * voi:>8.4f} {len(effects):>6} {len(effects.position_effects):>4}"
        )
    print(
        f"  {CONSOLE.paint(f'exploit_weight = 1 - {VOI_EXPLOIT_WEIGHT}·VoI; C = min(1, (cells + 0.5·pos) / 10)', 'dim')}"
    )


def print_benchmark_comparison(campaign) -> None:
    """Compact q=35 comparison of this live run against M1-M5.

    Reports **two** references, because they answer different questions:

    * the **same-seed** baselines — M1-M5 re-run on the live campaign's own five
      Round-0 queries. This is the like-for-like number: same starting hand, same
      pool, same budget;
    * the **100-seed median** — does the design work on average?

    They are never averaged together, and the live run is never compared to the
    median alone. A seed whose five random probes happen to contain no qualified
    binder starts from a lower ``y*(5)`` and therefore has *more* headroom, so
    ``AUBQC`` and ``y*(q)`` from a single run and from a seed median are not the
    same quantity.
    """
    benchmark = load_benchmark()
    kpis = campaign.state["kpis"]
    live_best = float(kpis["best_qualified_binding"])
    live_aubqc = campaign.aubqc()
    seed = live_seed_index(campaign, benchmark)

    CONSOLE.section("BASELINE COMPARISON · q=35")
    if not benchmark:
        print(f"  {CONSOLE.paint('results/benchmark_10seeds.json unavailable — run scripts/run_benchmark.py', 'yellow')}")
        return

    curves = benchmark.get("curves") or []
    final = next((row for row in curves if int(row.get("query_count", 0)) == 35), None)
    if final is None and curves:
        final = max(curves, key=lambda row: int(row.get("query_count", 0)))
    final = final or {}
    aubqc = benchmark.get("aubqc") or {}
    labels = baseline_labels(benchmark)

    cells = "".join(
        f"  {labels[method]} {_fmt(final.get(method))}"
        for method in BASELINE_METHODS
    )
    CONSOLE.kv("best qualified binding", f"live M5 {_fmt(live_best)}" + cells)
    aubqc_cells = "".join(
        f"  {labels[method]} {_fmt(aubqc.get(method))}"
        for method in BASELINE_METHODS
    )
    CONSOLE.kv("AUBQC (1/30 scaling)", f"live M5 {_fmt(live_aubqc)}" + aubqc_cells)

    # Same-seed reference first: it is the only comparison that attributes *this*
    # campaign's result. The seed median follows and answers a different question.
    # These rows use the plain ``M1..M5`` labels: they are single-seed runs, so the
    # ``(100-seed)`` suffix would be a lie.
    if seed is not None:
        same = seed_matched_baselines(campaign, benchmark, seed)
        if same:
            same_labels = dict(BASE_BASELINE_LABELS)
            live_q5 = _live_seed_checkpoint(campaign)
            CONSOLE.kv(
                f"same seed {seed} · q=5",
                f"live {_fmt(live_q5)}"
                + "".join(f"  {same_labels[m]} {_fmt(v['q5'])}" for m, v in same.items()),
            )
            CONSOLE.kv(
                f"same seed {seed} · q=35",
                f"live {_fmt(live_best)}"
                + "".join(f"  {same_labels[m]} {_fmt(v['q35'])}" for m, v in same.items()),
            )
            CONSOLE.kv(
                f"same seed {seed} · AUBQC",
                f"live {_fmt(live_aubqc)}"
                + "".join(f"  {same_labels[m]} {_fmt(v['aubqc'])}" for m, v in same.items()),
            )
            matched_aubqc = {m: v["aubqc"] for m, v in same.items()}
            name, value = max(matched_aubqc.items(), key=lambda item: float(item[1]))
            gap = live_aubqc - float(value)
            verdict = "ahead of" if gap > 0 else ("level with" if gap == 0 else "behind")
            print(
                f"  {CONSOLE.paint('→', 'grey')} live M5 AUBQC {verdict} the best *same-seed* "
                f"baseline ({same_labels[name]} {_fmt(value)}) by {gap:+.4f}"
            )

    # The 100-seed median answers "does the design work on average?" — a different
    # question from "did it work here", and never averaged with it.
    print()
    print(f"  {CONSOLE.paint('seed median (all 100 seeds) — average behaviour, not this run', 'dim')}")
    benchmark_values = {
        method: final.get(method) for method in BASELINE_METHODS if final.get(method) is not None
    }
    benchmark_aubqc = {
        method: aubqc.get(method) for method in BASELINE_METHODS if aubqc.get(method) is not None
    }
    if benchmark_values:
        name, value = max(benchmark_values.items(), key=lambda item: float(item[1]))
        gap = live_best - float(value)
        verdict = "ahead of" if gap > 0 else ("level with" if gap == 0 else "behind")
        print(
            f"  {CONSOLE.paint('→', 'grey')} live M5 {verdict} the best q=35 baseline "
            f"({labels[name]} {_fmt(value)}) by {gap:+.3f} binding units"
        )
    if benchmark_aubqc:
        name, value = max(benchmark_aubqc.items(), key=lambda item: float(item[1]))
        gap = live_aubqc - float(value)
        verdict = "ahead of" if gap > 0 else ("level with" if gap == 0 else "behind")
        print(
            f"  {CONSOLE.paint('→', 'grey')} live M5 AUBQC {verdict} the best baseline "
            f"({labels[name]} {_fmt(value)}) by {gap:+.4f}"
        )
    seeds = benchmark.get("seeds", "?")
    policy = benchmark.get("m5_policy_mode", "?")
    print(
        f"  {CONSOLE.paint(f'seed-median reference: {seeds} seeds, median y*(q) (M5 benchmark policy={policy})', 'dim')}"
    )
    print(
        f"  {CONSOLE.paint('live M5 is one campaign; compare it to the same-seed row, not to the seed median', 'dim')}"
    )
    if seed is None:
        print(
            f"  {CONSOLE.paint('same-seed row unavailable: the live ledger does not match any benchmark seed', 'yellow')}"
        )


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
        ceiling = row.get("max_tokens")
        deliberation = f"thinking={budget}" if budget else (f"effort={effort}" if effort else "direct")
        if ceiling:
            deliberation = f"{deliberation} cap={ceiling}"
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
    if campaign.hypotheses:
        history = "  ".join(
            f"R{item['round_number']}:{item['verdict']}" for item in campaign.hypotheses
        )
        CONSOLE.kv("verdicts by round", history)
    CONSOLE.kv("ledger verified", campaign.ledger.verify())
    print_voi_trajectory(campaign)

    print_benchmark_comparison(campaign)

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
