#!/usr/bin/env python3
"""100-seed M1-M5 benchmark, saving and printing every self-learning metric."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SETTINGS
from core.baselines import METHODS, run_benchmark
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import load_primary, public_pool

LABELS = {
    "m1_random": "M1",
    "m2_greedy": "M2",
    "m3_ucb": "M3",
    "m4_fixed_prior": "M4",
    "m5_hybrid": "M5",
}


def _cell(value, width: int, digits: int = 3) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"
    if number != number:  # NaN
        return f"{'n/a':>{width}}"
    return f"{number:>{width}.{digits}f}"


def _table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    line = "  ".join(cell.ljust(width) for cell, width in zip(header, widths))
    print(f"\n{title}")
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


def print_report(report: dict) -> None:
    methods = [m for m in METHODS if m in report.get("curves", [{}])[0]] or list(METHODS)
    header = [""] + [LABELS[m] for m in methods]

    curve_rows = [
        [f"q={row['query_count']}"] + [_cell(row[m], 7) for m in methods]
        for row in report["curves"]
    ]
    _table("BEST QUALIFIED BINDING · y*(q) median over seeds", header, curve_rows)

    panel_rows = [
        [f"q={row['query_count']}"] + [_cell(row[m], 7) for m in methods]
        for row in report["top5_panel_curves"]
    ]
    _table("TOP-5 PANEL MEAN · portfolio depth, median over seeds", header, panel_rows)

    normalized_rows = [
        [f"q={row['query_count']}"] + [_cell(row[m], 7, 4) for m in methods]
        for row in report["normalized_curves"]
    ]
    _table("NORMALIZED DISCOVERY · 0.0 = seed score, 1.0 = pool ceiling", header, normalized_rows)

    rho_rows = [
        [f"q={row['query_count']}"] + [_cell(row[m], 7, 4) for m in methods]
        for row in report["spearman_rho_curves"]
    ]
    _table(
        "SPEARMAN rho · method score vs true binding over the qualified pool",
        header,
        rho_rows,
    )

    summary_rows = [
        ["AUBQC (1/30)"] + [_cell(report["aubqc"][m], 7, 4) for m in methods],
        ["normalized AUBQC"] + [_cell(report["normalized_aubqc"][m], 7, 4) for m in methods],
        ["top-5 panel AUBQC"] + [_cell(report["top5_panel_aubqc"][m], 7, 4) for m in methods],
        ["top-decile hit rate"] + [_cell(report["top_decile_hit_rate"][m], 7, 4) for m in methods],
    ]
    _table(f"SUMMARY · {report['seeds']} seeds, M5 policy={report['m5_policy_mode']}", header, summary_rows)
    print(
        "\n  n/a on the Spearman row means the correlation is undefined (M1 has no scoring\n"
        "  model), not zero. Zero-shot references on 1N8Z: AF3 -0.16, ProteinMPNN -0.17,\n"
        "  ESM-2 -0.13."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--m5-policy-mode", choices=["adaptive_tool", "live_agent"], default="adaptive_tool")
    args = parser.parse_args()

    frame, synthetic = load_primary()
    protocol = freeze_protocol(public_pool(frame), is_synthetic=synthetic)
    normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
    result = run_benchmark(
        frame, normalizer, protocol["tau_dev"], seeds=args.seeds, m5_policy_mode=args.m5_policy_mode
    )

    path = SETTINGS.root / f"results/benchmark_{args.seeds}seeds.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    if args.seeds == 10:
        SETTINGS.benchmark_path.write_text(json.dumps(result, indent=2) + "\n")

    print_report(result)
    print(f"\nwrote {path}")
