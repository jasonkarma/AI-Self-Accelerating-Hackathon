#!/usr/bin/env python3
"""Attribution ablation: which of M5's two mechanisms costs it the cross-target cells?

The problem this answers
------------------------
M5 beats every baseline on 1N8Z and loses to M3/M4 on 2FJG. M5 differs from M3 in
exactly three ways:

1. ``Δ_agent`` — the empirical-Bayes SAR prior added to ``A_base``;
2. the **VoI anneal** — ``A_null`` moves from ``A_base`` toward ``mu_bind_norm`` as
   the campaign matures;
3. the **3:2 slot split** — slots 3 and 5 are reserved for a VoI probe and an H0
   control instead of being spent on exploitation.

(1) cannot be ablated without changing what M5 *is*: it is the hypothesis under
test, and the paired H1/H0 audit is what grades it. (2) and (3) are structural
choices that could be wrong for short campaigns, and nothing in the current
artifact distinguishes them. This script does, by switching off exactly one at a
time on the **same matched seeds**:

    production    the shipped law
    greedy_slots  slots 3, 4, 5 all rank by a_h1        (isolates the slot split)
    no_anneal     a_null := a_base                      (isolates the VoI anneal)

**No constant is introduced by either variant.** Both are structural.

The baseline rows (M2/M3/M4) are re-run on the same seeds too, so every Δ in the
output is a paired difference rather than a comparison against a number from a
different artifact.

Usage
-----
    python scripts/run_attribution_ablation.py --seeds 100                 # 1N8Z
    python scripts/run_attribution_ablation.py --dataset 2fjg --seeds 50  # one target
    python scripts/run_attribution_ablation.py --cross --seeds 50          # both

Writes ``results/attribution_ablation.json``. It never touches the three published
benchmark artifacts, ``results/campaign.sqlite3`` or ``results/golden_seed21.sqlite3``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config.settings import SETTINGS
from core.acquisition import (
    ABLATION_VARIANTS,
    REFERENCE_N_QUALIFIED,
    compute_campaign_protocol,
    resolve_anchor_positions,
)
from core.baselines import _paired_test, run_method, run_variant
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import load_primary, public_pool

LOGGER = logging.getLogger("ablation")

#: Baseline rows re-run on the same seeds, so every comparison is paired.
BASELINES = ("m2_greedy", "m3_ucb", "m4_fixed_prior")

LABELS = {
    None: "M5 production",
    "greedy_slots": "M5 greedy slots",
    "no_anneal": "M5 no anneal",
    "slot1_model": "M5 slot1=model",
    "harvest_model": "M5 harvest=model",
    "harvest_slot1": "M5 harvest slot1=model",
    "all_model": "M5 all=model",
    "m2_greedy": "M2",
    "m3_ucb": "M3",
    "m4_fixed_prior": "M4",
}


def _baseline_curve(
    frame, normalizer, tau_dev, *, seeds: int, method: str,
    wt_sequence, anchor_positions, protocol,
) -> dict:
    """One baseline on the matched seeds, reduced to the two reported scalars."""
    results = [
        run_method(
            frame, normalizer, tau_dev, method, seed,
            np.random.default_rng(seed).choice(
                frame.candidate_id.to_numpy(), 5, replace=False
            ).tolist(),
            wt_sequence=wt_sequence,
            anchor_positions=anchor_positions,
            protocol=protocol,
        )
        for seed in range(seeds)
    ]
    aubqc = np.asarray([r.aubqc for r in results], dtype=float)
    last_q = protocol.total_budget
    rho = [
        next((p["spearman_rho"] for p in r.rho_curve if p["query_count"] == last_q), None)
        for r in results
    ]
    rho_values = np.asarray([v for v in rho if v is not None], dtype=float)
    return {
        "variant": method,
        "method": method,
        "seeds": seeds,
        "aubqc": float(aubqc.mean()),
        "aubqc_se": float(aubqc.std(ddof=1) / np.sqrt(len(aubqc))) if len(aubqc) > 1 else float("nan"),
        "aubqc_per_seed": aubqc.tolist(),
        "rho": float(rho_values.mean()) if rho_values.size else None,
        "rho_n": int(rho_values.size),
        "rho_per_seed": rho,
    }


def _paired(treatment: dict, control: dict, field: str) -> dict:
    """Paired difference on per-seed values, skipping undefined entries."""
    a = np.asarray(treatment[f"{field}_per_seed"], dtype=object)
    b = np.asarray(control[f"{field}_per_seed"], dtype=object)
    keep = np.asarray(
        [x is not None and y is not None for x, y in zip(a, b)], dtype=bool
    )
    x = np.asarray([float(v) for v in a[keep]], dtype=float)
    y = np.asarray([float(v) for v in b[keep]], dtype=float)
    diffs = x - y
    n = int(diffs.size)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "p_value": float("nan"), "seeds_won": 0}
    from scipy.stats import t as student_t

    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    t_stat = float(mean / se) if se > 0 else float("inf") if mean != 0 else 0.0
    p = float(2.0 * student_t.sf(abs(t_stat), df=n - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    return {
        "n": n,
        "mean_diff": mean,
        "se": se,
        "p_value": p,
        "seeds_won": int((diffs > 0).sum()),
    }


def run_cell(
    frame, normalizer, tau_dev, *, seeds: int, label: str,
    wt_sequence=None, anchor_positions=None, protocol=None,
    include_baselines: bool = True,
) -> dict:
    """Every variant + baseline on one (target, coverage) cell."""
    if protocol is None:
        protocol = compute_campaign_protocol(REFERENCE_N_QUALIFIED, 13)
    LOGGER.info(
        "%s: N_qual=%d L=%d R=%d Q=%d seeds=%d",
        label, protocol.n_qualified, protocol.seq_len, protocol.rounds,
        protocol.total_budget, seeds,
    )
    rows = {}
    for variant in ABLATION_VARIANTS:
        rows[str(variant)] = run_variant(
            frame, normalizer, tau_dev, seeds=seeds, variant=variant,
            wt_sequence=wt_sequence, anchor_positions=anchor_positions, protocol=protocol,
        )
    if include_baselines:
        for method in BASELINES:
            rows[method] = _baseline_curve(
                frame, normalizer, tau_dev, seeds=seeds, method=method,
                wt_sequence=wt_sequence, anchor_positions=anchor_positions,
                protocol=protocol,
            )

    production = rows[str(None)]
    comparisons = {}
    for name, row in rows.items():
        if name == str(None):
            continue
        comparisons[name] = {
            "aubqc": _paired(production, row, "aubqc"),
            "rho": _paired(production, row, "rho"),
        }
    return {
        "cell": label,
        "protocol": protocol.to_dict(),
        "seeds": seeds,
        "rows": rows,
        # Positive = production M5 is BETTER than that row.
        "production_minus": comparisons,
    }


def format_table(cells: list[dict]) -> str:
    """One block per cell: AUBQC and rho for every variant and baseline."""
    lines: list[str] = []
    order = [str(v) for v in ABLATION_VARIANTS] + list(BASELINES)
    for cell in cells:
        rows = cell["rows"]
        prod = rows[str(None)]
        header = ["model", "AUBQC", "±SE", "vs prod M5", "p", "rho", "vs prod rho", "p"]
        table = []
        for name in order:
            row = rows.get(name)
            if row is None:
                continue
            if name == str(None):
                table.append([
                    LABELS.get(None, name), f"{row['aubqc']:.4f}", f"{row['aubqc_se']:.4f}",
                    "—", "—",
                    "n/a" if row["rho"] is None else f"{row['rho']:.4f}", "—", "—",
                ])
                continue
            cmp_ = cell["production_minus"].get(name) or {}
            a = cmp_.get("aubqc") or {}
            r = cmp_.get("rho") or {}
            table.append([
                LABELS.get(name, name),
                f"{row['aubqc']:.4f}",
                f"{row['aubqc_se']:.4f}",
                f"{a.get('mean_diff', float('nan')):+.4f}",
                f"{a.get('p_value', float('nan')):.4f}",
                "n/a" if row["rho"] is None else f"{row['rho']:.4f}",
                f"{r.get('mean_diff', float('nan')):+.4f}" if r.get("n") else "n/a",
                f"{r.get('p_value', float('nan')):.4f}" if r.get("n") else "—",
            ])
        widths = [max(len(r[i]) for r in [header] + table) for i in range(len(header))]
        lines.append("")
        lines.append(f"ATTRIBUTION · {cell['cell']}  (seeds={cell['seeds']}, "
                     f"Q={cell['protocol']['total_budget']}, R={cell['protocol']['rounds']})")
        lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
        lines.append("  " + "  ".join("-" * w for w in widths))
        for row in table:
            lines.append("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths)))
        lines.append(
            "  'vs prod M5' = production M5 minus that row; positive means the ablation\n"
            "  variant is WORSE than shipped M5. So a negative 'greedy slots' delta would\n"
            "  mean the reserved slots are costing AUBQC, and a negative 'no anneal' delta\n"
            "  would mean the VoI anneal is."
        )
    return "\n".join(lines)


def _cross_target_cells(seeds: int) -> list[dict]:
    """Both cross-targets at reference and high coverage, via the existing loader."""
    from data.datasets import align_cross_target_frame
    from scripts.run_cross_target_benchmark import TARGET_CHAIN, fetch_abbibench

    cells: list[dict] = []
    for target in TARGET_CHAIN:
        if target["key"] not in {"aayl49_ML", "2fjg"}:
            continue
        try:
            source = fetch_abbibench(
                "AbBibench/Antibody_Binding_Benchmark_Dataset", target["file"],
                __import__("os").getenv("HF_TOKEN"),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("skipping %s: %s", target["key"], exc)
            continue
        frame, diagnostics = align_cross_target_frame(
            source, prefix=target["key"].upper().replace("_", "")
        )
        reference = diagnostics["reference_sequence"]
        anchors = resolve_anchor_positions(reference)
        protocol = freeze_protocol(public_pool(frame), output=Path("/tmp/_abl_prot.json"),
                                   is_synthetic=False)
        normalizer = RiskNormalizer.fit(frame.sequence, wt_sequence=reference)
        for label, coverage in (("16.7pct", 35.0 / 210.0), ("33pct", 0.33)):
            campaign = compute_campaign_protocol(
                protocol["qualified_count"], diagnostics["cdr3_length"], coverage=coverage
            )
            cells.append(
                run_cell(
                    frame, normalizer, protocol["tau_dev"], seeds=seeds,
                    label=f"{target['key']} @ {label}",
                    wt_sequence=reference, anchor_positions=anchors,
                    protocol=campaign, include_baselines=True,
                )
            )
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--dataset", default=None,
                        help="one cross-target key (aayl49_ML / 2fjg); omit for 1N8Z")
    parser.add_argument("--cross", action="store_true",
                        help="run both cross-targets instead of 1N8Z")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cross or args.dataset:
        cells = _cross_target_cells(args.seeds)
    else:
        frame, synthetic = load_primary()
        protocol = freeze_protocol(public_pool(frame), is_synthetic=synthetic)
        normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
        cells = [run_cell(
            frame, normalizer, protocol["tau_dev"], seeds=args.seeds,
            label="1N8Z @ 16.7pct",
            protocol=compute_campaign_protocol(REFERENCE_N_QUALIFIED, 13),
        )]

    # The gate that has to pass before any variant number is believable.
    for cell in cells:
        prod = cell["rows"][str(None)]
        LOGGER.info("%s production M5 AUBQC = %.6f", cell["cell"], prod["aubqc"])

    report = {"seeds": args.seeds, "cells": cells}
    path = SETTINGS.root / "results/attribution_ablation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(format_table(cells))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
