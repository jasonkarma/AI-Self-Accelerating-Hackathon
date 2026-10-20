#!/usr/bin/env python3
"""Baseline-only learnability gate. **M5 is never constructed by this script.**

Why this exists
---------------
``aayl51`` came back as a null result: every surrogate sat at ``rho < 0.06``, so
"which method ranks better" was a comparison between two near-uncorrelated
rankers. A holdout where nothing can be learned cannot distinguish a generalising
engine from a lucky one — it only shows the loop had nothing to find.

So a *learnable* companion holdout is needed, and the target must be chosen
**without looking at M5**. This script therefore runs exactly two baselines:

    M2 greedy      (pure surrogate exploitation, ``beta_0 = 0``)
    M3 UCB         (dual-surrogate UCB, the H0 null hypothesis)

and reports only their Spearman ``rho`` against true binding at the final
checkpoint. If neither baseline correlates with truth, there is no signal for M5
to exploit and the target cannot answer the generalisation question.

The gate
--------
``max(M2_rho, M3_rho) >= 0.15`` at the pre-registered 16.7 % coverage.

Why ``0.15`` and not a tuned number: it is the same order as the published
zero-shot predictors this repo already quotes for 1N8Z context — AlphaFold3
``-0.16``, ProteinMPNN ``-0.17``, ESM-2 ``-0.13`` in *magnitude*. A target whose
own exploitation baselines cannot reach that magnitude is not a ranking problem
in the sense the benchmark measures.

Why M2/M3 and not M4: M4's static loop prior is a *hand-written rule*, so its rho
measures how well that rule happens to match the landscape, not whether the
surrogate can learn. M2 and M3 are the learnability ceiling.

Overlap
-------
Every candidate is deduplicated against the three already-scored pools
(``1n8z``, ``aayl49_ML``, ``2fjg``) by exact CDRH3 string. Any sequence already
scored anywhere is **removed**, and the percentage removed is reported, because a
"new" target that is 40 % re-measured molecules is not new.

Usage
-----
    python scripts/baseline_learnability_gate.py
    python scripts/baseline_learnability_gate.py --seeds 50 --coverage 0.167
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.acquisition import compute_campaign_protocol, resolve_anchor_positions
from core.baselines import run_method
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import align_cross_target_frame, public_pool

REPO = "AbBibench/Antibody_Binding_Benchmark_Dataset"

#: Targets whose pools are already published. Any candidate sequence found in one
#: of these is removed before the gate is measured.
SCORED_TARGETS: dict[str, str] = {
    "1n8z": "binding_affinity/1n8z_benchmarking_data.csv",
    "aayl49_ML": "binding_affinity/aayl49_ML_benchmarking_data.csv",
    "2fjg": "binding_affinity/2fjg_benchmarking_data.csv",
}

#: Minimum qualified pool for the gate to be meaningful. Below this a 16.7 %
#: campaign is one or two rounds, so ``rho`` is measured on a handful of revealed
#: points and the number is noise. Lower than the lockbox's 150 because learning
#: *is* possible from a small pool — this is a different question.
MIN_NQUAL = 50

#: The gate threshold, stated once. See the module docstring for why 0.15.
RHO_GATE = 0.15

BASELINES = ("m2_greedy", "m3_ucb")


def load_pool(filename: str, *, prefix: str = "GATE"):
    from scripts.run_cross_target_benchmark import fetch_abbibench

    source = fetch_abbibench(REPO, filename, None)
    return align_cross_target_frame(source, prefix=prefix)


def scored_universe() -> tuple[set[str], dict[str, set[str]]]:
    """Every exact CDRH3 already measured in a published pool, per target."""
    per_target: dict[str, set[str]] = {}
    for key, filename in SCORED_TARGETS.items():
        frame, _ = load_pool(filename, prefix=key.upper().replace("_", ""))
        per_target[key] = set(frame.sequence.astype(str))
    union = set().union(*per_target.values()) if per_target else set()
    return union, per_target


def dedupe(frame, universe: set[str]):
    """Drop every sequence already scored. Returns ``(frame, pct_removed)``."""
    before = len(frame)
    kept = frame[~frame.sequence.astype(str).isin(universe)].reset_index(drop=True)
    removed = before - len(kept)
    pct = 0.0 if before == 0 else 100.0 * removed / before
    kept = kept.copy()
    kept["candidate_id"] = [f"{c.split('-')[0]}-{i:03d}" for i, c in enumerate(kept.candidate_id)]
    return kept, pct


def gate_one(
    frame,
    *,
    target_id: str,
    seeds: int,
    coverage: float,
    coverage_label: str,
    overlap_pct: float,
) -> dict:
    """Measure M2 and M3 ``rho`` on one target. M5 is never imported or built."""
    if len(frame) < MIN_NQUAL:
        return {
            "target_id": target_id,
            "L": int(len(frame.sequence.iloc[0])) if len(frame) else None,
            "N_qual_deduped": 0,
            "overlap_pct_removed": round(overlap_pct, 2),
            "coverage_label": coverage_label,
            "skipped": f"pool {len(frame)} < MIN_NQUAL {MIN_NQUAL} after dedupe",
        }

    reference = "".join(
        frame.sequence.astype(str).str[i].mode().iloc[0]
        for i in range(len(frame.sequence.iloc[0]))
    )
    with tempfile.TemporaryDirectory(prefix="gate-") as scratch:
        protocol = freeze_protocol(
            public_pool(frame), output=Path(scratch) / "p.json", is_synthetic=False
        )
    normalizer = RiskNormalizer.fit(frame.sequence, wt_sequence=reference)
    anchors = resolve_anchor_positions(reference)
    campaign = compute_campaign_protocol(
        protocol["qualified_count"], len(reference), coverage=coverage
    )

    rho: dict[str, float | None] = {}
    for method in BASELINES:
        values = []
        for seed in range(seeds):
            initial = np.random.default_rng(seed).choice(
                frame.candidate_id.to_numpy(), 5, replace=False
            ).tolist()
            result = run_method(
                frame, normalizer, protocol["tau_dev"], method, seed, initial,
                wt_sequence=reference, anchor_positions=anchors, protocol=campaign,
            )
            point = next(
                (p for p in reversed(result.rho_curve) if p.get("spearman_rho") is not None),
                None,
            )
            if point is not None:
                values.append(float(point["spearman_rho"]))
        rho[method] = float(np.mean(values)) if values else None

    best = max((v for v in rho.values() if v is not None), default=None)
    return {
        "target_id": target_id,
        "L": int(len(reference)),
        "N_qual_deduped": int(protocol["qualified_count"]),
        "pool_deduped": int(len(frame)),
        "overlap_pct_removed": round(overlap_pct, 2),
        "coverage_label": coverage_label,
        "coverage_pct": round(100 * campaign.coverage_pct, 3),
        "R": campaign.rounds,
        "Q": campaign.total_budget,
        "threshold_source": protocol["threshold_source"],
        "M2_rho": None if rho["m2_greedy"] is None else round(rho["m2_greedy"], 4),
        "M3_rho": None if rho["m3_ucb"] is None else round(rho["m3_ucb"], 4),
        "max_baseline_rho": None if best is None else round(best, 4),
        "passes_gate": bool(best is not None and best >= RHO_GATE),
    }


def render(rows: list[dict], *, gate: float) -> str:
    header = [
        "target_id", "L", "N_qual", "overlap%", "coverage", "M2_rho", "M3_rho",
        "max", "gate",
    ]
    table = []
    for r in rows:
        if r.get("skipped"):
            table.append([r["target_id"], str(r["L"]), "-", f"{r['overlap_pct_removed']:.2f}",
                          r["coverage_label"], "-", "-", "-", "SKIP"])
            continue
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        table.append([
            r["target_id"], str(r["L"]), str(r["N_qual_deduped"]),
            f"{r['overlap_pct_removed']:.2f}", r["coverage_label"],
            fmt(r["M2_rho"]), fmt(r["M3_rho"]), fmt(r["max_baseline_rho"]),
            "PASS" if r["passes_gate"] else "fail",
        ])
    widths = [max(len(x[i]) for x in [header] + table) for i in range(len(header))]
    lines = ["", f"BASELINE-ONLY LEARNABILITY GATE (M5 never run; threshold {gate})"]
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("  " + "  ".join("-" * w for w in widths))
    for row in table:
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    for r in rows:
        if r.get("skipped"):
            lines.append(f"\n  {r['target_id']}: {r['skipped']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--out", default="/tmp/baseline_learnability.json")
    args = parser.parse_args(argv)

    # Guard: this script must never import or construct M5. If a future refactor
    # adds it, the gate stops being blind to the thing it is supposed to protect.
    from core import baselines as _b

    assert not hasattr(_b, "_pick_m5_built"), "gate must not construct M5"

    universe, per_target = scored_universe()
    print(f"scored universe: {len(universe)} unique CDRH3 across "
          f"{list(per_target)} (per-target sizes: "
          f"{ {k: len(v) for k, v in per_target.items()} })")

    # Candidate list: every corpus target with enough *unique* CDRH3s to be a
    # search problem, minus the ones already scored (those are the universe).
    candidates = [
        ("1mlc", "binding_affinity/1mlc_benchmarking_data.csv"),
        ("4d5_her2", "binding_affinity/4d5_her2_benchmarking_data_trimmed.csv"),
        ("aayl49", "binding_affinity/aayl49_benchmarking_data.csv"),
        ("1mhp_LC", "binding_affinity/1mhp_LC_benchmarking_data.csv"),
        ("3gbn_h1", "binding_affinity/3gbn_h1_benchmarking_data.csv"),
        ("3gbn_h9", "binding_affinity/3gbn_h9_benchmarking_data.csv"),
        ("4fqi_h1", "binding_affinity/4fqi_h1_benchmarking_data.csv"),
        ("4fqi_h3", "binding_affinity/4fqi_h3_benchmarking_data.csv"),
        ("5a12_ang2", "binding_affinity/5a12_ang2_benchmarking_data.csv"),
        ("5a12_vegf", "binding_affinity/5a12_vegf_benchmarking_data.csv"),
        ("aayl50_LC", "binding_affinity/aayl50_LC_benchmarking_data.csv"),
        ("aayl51", "binding_affinity/aayl51_benchmarking_data.csv"),
        ("aayl52_LC", "binding_affinity/aayl52_LC_benchmarking_data.csv"),
        ("g6_LC", "binding_affinity/g6_LC_benchmarking_data.csv"),
    ]

    rows: list[dict] = []
    frames: dict[str, object] = {}
    for target_id, filename in candidates:
        try:
            frame, _ = load_pool(filename, prefix=target_id.upper().replace("_", ""))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "target_id": target_id, "L": None, "N_qual_deduped": 0,
                "overlap_pct_removed": 0.0, "coverage_label": "-",
                "skipped": f"align failed: {type(exc).__name__}: {exc}",
            })
            continue
        kept, pct = dedupe(frame, universe)
        frames[target_id] = (kept, pct)
        # Pre-screen on unique-count so we do not spend minutes on 1-2 sequence
        # pools that can never pass.
        if len(kept) < MIN_NQUAL:
            rows.append({
                "target_id": target_id,
                "L": int(len(kept.sequence.iloc[0])) if len(kept) else None,
                "N_qual_deduped": 0, "overlap_pct_removed": round(pct, 2),
                "coverage_label": "-",
                "skipped": f"{len(kept)} unique CDRH3 after dedupe (< {MIN_NQUAL})",
            })

    for target_id in ("1mlc", "4d5_her2", "aayl49"):
        if target_id not in frames:
            continue
        kept, pct = frames[target_id]
        for label, coverage in (("16.7pct", 35.0 / 210.0), ("33pct", 0.33)):
            rows.append(gate_one(
                kept, target_id=target_id, seeds=args.seeds,
                coverage=coverage, coverage_label=label, overlap_pct=pct,
            ))

    rows.sort(key=lambda r: (r["target_id"], r["coverage_label"]))
    report = {
        "seeds": args.seeds,
        "rho_gate": RHO_GATE,
        "min_nqual": MIN_NQUAL,
        "scored_universe_size": len(universe),
        "rows": rows,
        "note": "M5 was never constructed; only M2 and M3 were run.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(render(rows, gate=RHO_GATE))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
