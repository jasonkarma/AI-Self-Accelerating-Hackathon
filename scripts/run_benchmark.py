#!/usr/bin/env python3
"""100-seed M1-M5 benchmark, saving and printing every self-learning metric."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SETTINGS
from core.baselines import METHODS, PAIRED_METRICS, WARM_START_METHOD, run_benchmark
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import load_primary, public_pool

LABELS = {
    "m1_random": "M1",
    "m2_greedy": "M2",
    "m3_ucb": "M3",
    "m4_fixed_prior": "M4",
    "m5_hybrid": "M5",
    WARM_START_METHOD: "M5-warm",
}

PAIRED_METRIC_LABELS = {
    "aubqc": "AUBQC",
    "normalized_aubqc": "normalized AUBQC",
    "panel_aubqc": "top-5 panel AUBQC",
    "top_decile_hit_rate": "top-decile hit rate",
    "spearman_rho_at_q35": "Spearman rho @ q=35",
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

    rho_q35 = {
        row["query_count"]: row for row in report["spearman_rho_curves"]
    }.get(35, {})
    aubqc_se = report.get("aubqc_se") or {}

    def _with_se(method: str) -> str:
        """``mean ± SE`` — the width that decides whether a gap is evidence."""
        mean = report["aubqc"][method]
        se = aubqc_se.get(method)
        if se is None or not np.isfinite(se):
            return _cell(mean, 13, 4)
        return f"{mean:>7.4f} ±{se:.4f}"

    summary_rows = [
        ["AUBQC (1/30), mean ±SE"] + [_with_se(m) for m in methods],
        ["AUBQC (1/30), point"] + [_cell(report["aubqc"][m], 7, 4) for m in methods],
        ["normalized AUBQC"] + [_cell(report["normalized_aubqc"][m], 7, 4) for m in methods],
        ["top-5 panel AUBQC"] + [_cell(report["top5_panel_aubqc"][m], 7, 4) for m in methods],
        ["top-decile hit rate"] + [_cell(report["top_decile_hit_rate"][m], 7, 4) for m in methods],
        ["Spearman rho @ q=35"] + [_cell(rho_q35.get(m), 7, 4) for m in methods],
    ]
    _table(f"SUMMARY · {report['seeds']} seeds, M5 policy={report['m5_policy_mode']}", header, summary_rows)
    print(
        "\n  n/a on the Spearman row means the correlation is undefined (M1 has no scoring\n"
        "  model), not zero. Zero-shot references on 1N8Z: AF3 -0.16, ProteinMPNN -0.17,\n"
        "  ESM-2 -0.13."
    )

    paired = report.get("paired_t_tests") or {}
    if paired:
        test_rows = []
        for key, block in paired.items():
            for metric in PAIRED_METRICS:
                stats = block["metrics"].get(metric)
                if stats is None:
                    continue
                test_rows.append([
                    f"{block['performer']} - {block['baseline']}",
                    PAIRED_METRIC_LABELS[metric],
                    _cell(stats["mean_diff"], 10, 5),
                    _cell(stats["se"], 8, 5),
                    _cell(stats["t_stat"], 8, 3),
                    _cell(stats["p_value"], 8, 4),
                    f"{stats['seeds_won']}/{stats['n']}",
                ])
        _table(
            "PAIRED T-TESTS · identical initial seeds, two-sided",
            ["comparison", "metric", "mean diff", "SE", "t", "p", "won"],
            test_rows,
        )
        print(
            "\n  Paired over seeds: each seed contributes one difference, so the seed's own\n"
            "  difficulty cancels. One-sided evidence for improvement is p/2 when the mean\n"
            "  difference is positive."
        )

    warm = report.get("m5_warm_start")
    if warm:
        rows = []
        warm_curve = {row["query_count"]: row[WARM_START_METHOD] for row in warm["curves"]}
        warm_panel = {row["query_count"]: row[WARM_START_METHOD] for row in warm["top5_panel_curves"]}
        for q in range(5, 36, 5):
            rows.append([
                f"q={q}",
                _cell(warm_curve.get(q), 7),
                _cell(warm_panel.get(q), 7),
            ])
        _table(
            "M5 WARM START · Round 0 chosen by the cold-start prior (separate ablation)",
            ["checkpoint", "y*(q)", "top-5 panel"],
            rows,
        )
        s = warm["summary"]
        _table(
            "M5 WARM START SUMMARY · not part of the matched M1-M5 comparison",
            ["metric", "value"],
            [
                ["AUBQC (1/30)", _cell(s["aubqc"], 10, 4)],
                ["normalized AUBQC", _cell(s["normalized_aubqc"], 10, 4)],
                ["top-5 panel AUBQC", _cell(s["top5_panel_aubqc"], 10, 4)],
                ["top-decile hit rate", _cell(s["top_decile_hit_rate"], 10, 4)],
                ["Spearman rho @ q=35", _cell(s["spearman_rho_at_q35"], 10, 4)],
            ],
        )
        print(
            "\n  Reported separately on purpose: M1-M5 keep their five matched random seeds so\n"
            "  the primary ablation isolates post-seed feedback learning. This row shows what\n"
            "  the Virtual Lab's cold-start rule buys before it has any labels. Its normalized\n"
            "  curve is measured from the seed's *matched random* y*(5), so it shares M5's\n"
            "  [0%, 100%] ruler instead of being normalised against its own stronger start."
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

    # ``SETTINGS.benchmark_path`` is what ``/api/benchmark`` serves, and the stage
    # tests assert it carries the full 100-seed ablation. It therefore must not be
    # overwritten by a *shorter* exploratory run: a 10-seed pass used to clobber
    # the published artifact, which silently downgraded the served report and
    # broke the UI's seed-21 same-seed row. Publishing only when the new run is at
    # least as large as what is already on disk makes the artifact monotone in
    # evidence, which is the property a reader expects of a published table.
    served = SETTINGS.benchmark_path
    published_seeds = 0
    if served.exists():
        try:
            published_seeds = int(json.loads(served.read_text()).get("seeds") or 0)
        except (ValueError, OSError):
            published_seeds = 0
    if args.seeds >= published_seeds:
        served.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {served} (served by /api/benchmark)")
    else:
        print(
            f"kept {served}: it holds a {published_seeds}-seed run, which a "
            f"{args.seeds}-seed pass must not downgrade"
        )

    print_report(result)
    print(f"\nwrote {path}")
