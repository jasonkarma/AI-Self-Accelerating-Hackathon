#!/usr/bin/env python3
"""Freeze the current campaign ledger as the golden Seed-21 reference.

    python scripts/promote_golden.py --expect-best 8.580 --expect-aubqc 1.963

``results/golden_seed21.sqlite3`` is what ``POST /api/restore-golden`` copies back
over the live ledger during a demo, so it must never be written by anything on the
stage path. This script is the one deliberate way to refresh it, and it refuses to
overwrite a good file unless the *new* run passes every gate:

* the hash chain verifies (nothing was edited after the fact);
* the run is complete (``SETTINGS.rounds`` rounds of reveals);
* the spend matches the budget (``5 + 6 x 5 = 35``);
* the best qualified binding and AUBQC match the numbers the report quotes.

That last check is the point: promoting a ledger because it is the newest one is
how a demo ends up quoting a number nobody can reproduce. ``--force`` skips the
value checks (not the integrity ones) for a legitimate re-run whose numbers have
moved, and prints exactly what changed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SETTINGS  # noqa: E402
from core.developability import RiskNormalizer  # noqa: E402
from core.ledger import copy_ledger, ledger_version  # noqa: E402
from core.preregistration import load_protocol  # noqa: E402
from data.datasets import load_primary  # noqa: E402


def summarize(path: Path) -> dict:
    """Round count, spend, best qualified binding and AUBQC of a ledger file."""
    from core.campaign import Campaign

    frame, _synthetic = load_primary()
    protocol = load_protocol()
    normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
    campaign = Campaign(
        frame, normalizer, protocol, {}, ledger_path=path,
        golden_ledger_path=path,
    )
    return {
        "rounds": campaign.current_round(),
        "queries": int(len(campaign.revealed())),
        "best_qualified_binding": campaign.state["kpis"]["best_qualified_binding"],
        "aubqc": campaign.aubqc(),
        "verified": campaign.ledger.verify(),
        "entries": campaign.snapshot().version[1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the campaign ledger as the golden reference.")
    parser.add_argument("--source", type=Path, default=SETTINGS.ledger_path)
    parser.add_argument("--destination", type=Path, default=SETTINGS.golden_ledger_path)
    parser.add_argument("--expect-best", type=float, default=8.580)
    parser.add_argument("--expect-aubqc", type=float, default=1.963)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--force", action="store_true",
                        help="promote even if the KPI values moved (integrity is still checked)")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"refusing: {args.source} does not exist", file=sys.stderr)
        return 1
    summary = summarize(args.source)

    failures = []
    if not summary["verified"]:
        failures.append("hash chain does not verify")
    if summary["rounds"] != SETTINGS.rounds:
        failures.append(f"ledger holds {summary['rounds']} rounds, expected {SETTINGS.rounds}")
    if summary["queries"] != SETTINGS.total_budget:
        failures.append(f"ledger spent {summary['queries']} queries, expected {SETTINGS.total_budget}")
    if not args.force:
        if abs(summary["best_qualified_binding"] - args.expect_best) > args.tolerance:
            failures.append(
                f"best qualified binding {summary['best_qualified_binding']} "
                f"!= {args.expect_best} (+-{args.tolerance})"
            )
        if abs(summary["aubqc"] - args.expect_aubqc) > args.tolerance:
            failures.append(
                f"AUBQC {summary['aubqc']:.4f} != {args.expect_aubqc} (+-{args.tolerance})"
            )
    if failures:
        print("refusing to promote; the current ledger is not the verified run:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 1

    if args.destination.exists():
        backup = args.destination.with_suffix(".previous.sqlite3")
        shutil.copyfile(args.destination, backup)
        print(f"previous golden backed up to {backup}")
    written = copy_ledger(args.source, args.destination)
    print(f"wrote {args.destination} ({written} entries, version {ledger_version(args.destination)})")
    print(json.dumps(summarize(args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
