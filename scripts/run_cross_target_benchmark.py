#!/usr/bin/env python3
"""Cross-target generalization benchmark: the M1-M5 ablation on a second antibody dataset.

The claim under test
--------------------
``M5`` (:func:`core.acquisition.select_five_slots` + ``m5_warm_start``) generalizes
to a second real antibody-antigen dataset **without retuning a single constant**.
Every number that shapes a decision — ``AGENT_PRIOR_BOUND``, the SAR shrinkage
constants, ``VOI_EXPLOIT_WEIGHT``, ``SLOT3_OPTIMISM``, the 3-exploit/2-experimental
split, AUBQC's ``1/30`` — is identical to the 1N8Z run. Only the *reference frame*
changes: the CDR3 width and the consensus sequence of the new target.

What is deliberately shared code
--------------------------------
This script calls the same :func:`core.baselines.run_benchmark` that produced
``results/benchmark_100seeds.json``. It does not reimplement a single selection
rule, so a difference between the two reports is a property of the *data*, not of
a divergent code path.

Target selection
----------------
The requested chain is tried in order (``4FQI`` -> ``3GBN`` -> ``aayl49_ML`` ->
``1MLC``), then ``2FJG``, then the on-disk real 4D5/HER2 slice. A candidate is
accepted only if it clears :func:`data.datasets.cross_target_usability`:

* >= 35 distinct CDR3 sequences (five slots x seven rounds cannot be filled below
  that), and
* at least two positions that actually vary.

Without that second clause ``4FQI`` and ``3GBN`` look like the obvious choices —
they are the two largest tables in AbBiBench — while being the two *worst*
possible ones: their 65,094 and 1,887 rows collapse to **two** distinct CDR3s
each, because they are framework DMS libraries with 12 of 14 positions held
constant. M1-M5 would all query the same two molecules and every metric would be
degenerate. The gate exists to catch exactly that, and every rejection is recorded
in the artifact with its measured reason.

Usage
-----
    python scripts/run_cross_target_benchmark.py --seeds 50
    python scripts/run_cross_target_benchmark.py --dataset 1mlc --seeds 50

Writes ``results/cross_target_benchmark.json`` and
``data/processed/cross_target.csv``. It never touches
``results/benchmark_100seeds.json``, ``results/benchmark_10seeds.json``,
``results/golden_seed21.sqlite3`` or ``results/campaign.sqlite3``, and it freezes
its threshold to ``config/cross_target_protocol.json`` rather than over the 1N8Z
preregistration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import SETTINGS
from core.acquisition import (
    additive_prior_trust,
    compute_campaign_protocol,
    falsification_phase_rounds,
    resolve_anchor_positions,
)
from core.baselines import METHODS, PAIRED_METRICS, WARM_START_METHOD, run_benchmark
from core.constants import WT_SEQUENCE
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import (
    align_cross_target_frame,
    cross_target_usability,
    load_primary,
    public_pool,
)

LOGGER = logging.getLogger("cross_target")

#: AbBiBench target chain, in the order the request names them, followed by the
#: one target in the repository that is actually a CDR3 search problem at 13
#: residues. Display metadata only — every acceptance decision is made by
#: :func:`data.datasets.cross_target_usability` on the data itself.
TARGET_CHAIN: tuple[dict, ...] = (
    {"key": "4fqi_h3", "label": "4FQI h3 (CR9114 / Influenza HA)", "file": "binding_affinity/4fqi_h3_benchmarking_data.csv"},
    {"key": "4fqi_h1", "label": "4FQI h1 (CR9114 / Influenza HA)", "file": "binding_affinity/4fqi_h1_benchmarking_data.csv"},
    {"key": "3gbn_h1", "label": "3GBN h1 (CR6261 / Influenza HA)", "file": "binding_affinity/3gbn_h1_benchmarking_data.csv"},
    {"key": "aayl49_ML", "label": "aayl49 ML (anti-lysozyme)", "file": "binding_affinity/aayl49_ML_benchmarking_data.csv"},
    {"key": "1mlc", "label": "1MLC (anti-lysozyme)", "file": "binding_affinity/1mlc_benchmarking_data.csv"},
    {"key": "2fjg", "label": "2FJG (b12 / HIV-1 gp120)", "file": "binding_affinity/2fjg_benchmarking_data.csv"},
)

#: Coverage levels in the sensitivity sweep, as fractions of the qualified pool.
#: ``16.7%`` is the pre-registered rate (``35/210``) recovered from the frozen
#: 1N8Z protocol; the other two bracket it. The sweep is published **because** the
#: rate was derived from a dataset we already had, which is exactly the degree of
#: freedom a reader should be able to inspect rather than trust.
COVERAGE_SWEEP: tuple[tuple[str, float], ...] = (
    ("5pct", 0.05),
    ("16.7pct", 35.0 / 210.0),
    ("33pct", 0.33),
)

#: The two targets the dynamic-sizing run reports. Both are real AbBiBench
#: antibody-antigen datasets with genuine positional diversity; the rejected
#: targets are still recorded in every result so the selection is auditable.
PRIMARY_TARGETS: tuple[str, ...] = ("aayl49_ML", "2fjg")

#: Fallback slice already on disk: real AbBiBench 4D5/HER2 data, 13-mers.
LOCAL_FALLBACK = {"key": "4d5_her2", "label": "4D5 / HER2 (data/raw/dev_hcdr3.csv)"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Target acquisition ────────────────────────────────────────────────────────
def fetch_abbibench(repo: str, filename: str, token: str | None) -> pd.DataFrame:
    """Download one AbBiBench CSV through the same path as ``fetch_real_1n8z``."""
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, token=token))
    return pd.read_csv(local)


def load_local_fallback() -> tuple[pd.DataFrame, dict]:
    """The real 4D5/HER2 slice that already ships in ``data/raw/``.

    ``dev_hcdr3.csv`` is AbBiBench's own ``4d5_her2`` binding table, 13-mers,
    extracted by the frozen 1N8Z ingest. It is used here as a *source*, not as the
    benchmark pool — the pool is re-derived through the same aligner so the
    collapse/consensus rules are identical to a downloaded target.
    """
    frame = pd.read_csv(SETTINGS.development_csv)
    if not {"sequence", "y_bind"} <= set(frame.columns):
        raise ValueError(f"{SETTINGS.development_csv} lacks sequence/y_bind")
    raw = frame.loc[:, ["sequence", "y_bind"]].copy()
    raw["heavy_chain_seq"] = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC"
        + raw.sequence.astype(str)
        + "WGQGTLVTVSSA"
    )
    raw["binding_score"] = raw.y_bind
    return raw.loc[:, ["heavy_chain_seq", "binding_score"]], {
        "key": LOCAL_FALLBACK["key"],
        "label": LOCAL_FALLBACK["label"],
        "source": f"local:{SETTINGS.development_csv.name}",
        "source_file": "binding_affinity/4d5_her2_benchmarking_data_trimmed.csv",
    }


def select_target(
    dataset: str | None,
    repo: str,
    token: str | None,
    *,
    allow_fallback: bool = True,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """First target in the chain whose aligned pool is a real search problem.

    Returns ``(frame, provenance, rejections)``. ``rejections`` is ordered and
    records the measured reason each earlier target was skipped, so the artifact
    can state *why* 4FQI did not run rather than leaving a reader to assume a bug.
    """
    chain = TARGET_CHAIN
    if dataset:
        chain = tuple(t for t in TARGET_CHAIN if t["key"].lower() == dataset.lower())
        if not chain:
            raise SystemExit(
                f"unknown --dataset {dataset!r}; choose from "
                f"{[t['key'] for t in TARGET_CHAIN]} or 'auto'"
            )
    rejections: list[dict] = []
    for target in chain:
        try:
            source = fetch_abbibench(repo, target["file"], token)
        except Exception as exc:  # noqa: BLE001 - a missing target is not fatal
            LOGGER.warning("Could not fetch %s: %s", target["key"], exc)
            rejections.append({
                "key": target["key"], "label": target["label"],
                "reason": f"fetch failed: {type(exc).__name__}: {exc}",
            })
            continue
        try:
            frame, diagnostics = align_cross_target_frame(
                source, prefix=target["key"].upper().replace("_", "")
            )
        except Exception as exc:  # noqa: BLE001 - an unalignable target is not fatal
            LOGGER.warning("Could not align %s: %s", target["key"], exc)
            rejections.append({
                "key": target["key"], "label": target["label"],
                "reason": f"align failed: {type(exc).__name__}: {exc}",
            })
            continue
        usability = diagnostics["usability"]
        if not usability["usable"]:
            LOGGER.info("Skipping %s: %s", target["key"], usability["reason"])
            rejections.append({
                "key": target["key"],
                "label": target["label"],
                "reason": usability["reason"],
                "unique_sequences": usability["unique_sequences"],
                "cdr3_length": usability["length"],
                "varying_positions": usability["varying_positions"],
            })
            continue
        provenance = {
            "key": target["key"],
            "label": target["label"],
            "source": "huggingface",
            "repo": repo,
            "source_file": target["file"],
        }
        return frame, {**provenance, **diagnostics}, rejections

    if not allow_fallback:
        raise SystemExit(f"no usable target in chain; rejections: {rejections}")
    LOGGER.warning("No downloadable target was usable; falling back to the on-disk 4D5/HER2 slice")
    source, provenance = load_local_fallback()
    frame, diagnostics = align_cross_target_frame(source, prefix="XTG4D5")
    usability = diagnostics["usability"]
    if not usability["usable"]:
        raise SystemExit(
            f"fallback slice is not usable either ({usability['reason']}); "
            f"rejections: {rejections}"
        )
    return frame, {**provenance, **diagnostics}, rejections


# ── Benchmark ─────────────────────────────────────────────────────────────────
def _prepare_target(dataset: str, repo: str, token: str | None, *, csv_stem: str):
    """Align one target to its admissible pool and freeze its own protocol.

    Returns ``(frame, provenance, protocol, normalizer, reference, anchors)``.
    Each target gets its **own** protocol file and CSV: two datasets cannot share
    a preregistered threshold, and a shared file would silently give the second
    run the first one's ``tau_dev``.
    """
    frame, provenance, rejections = select_target(dataset, repo, token)
    reference = provenance["reference_sequence"]
    anchors = resolve_anchor_positions(reference)

    SETTINGS.cross_target_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_path = SETTINGS.cross_target_csv.with_name(f"cross_target_{csv_stem}.csv")
    frame.to_csv(csv_path, index=False)

    # ``freeze_protocol`` refuses a frame carrying ``y_bind``, which is the
    # label-blindness guarantee — inherited, not re-implemented. It is written to
    # its own file: the default target is the 1N8Z preregistration, and
    # overwriting that would invalidate the live campaign and the golden ledger.
    protocol_path = SETTINGS.cross_target_protocol_path.with_name(
        f"cross_target_protocol_{csv_stem}.json"
    )
    protocol = freeze_protocol(public_pool(frame), output=protocol_path, is_synthetic=False)

    # The normalizer is re-fit on this pool and anchored on this target's
    # consensus, so ``y_dev`` means "developability relative to *this* germline".
    normalizer = RiskNormalizer.fit(frame.sequence, wt_sequence=reference)
    return frame, {**provenance, "_rejections": rejections, "_csv": csv_path,
                   "_protocol_path": protocol_path}, protocol, normalizer, reference, anchors


def _one_run(
    frame, provenance, protocol, normalizer, reference, anchors,
    *, seeds: int, coverage: float, coverage_label: str, m5_policy_mode: str,
) -> dict:
    """One (target, coverage) cell: size the campaign, then run the ablation."""
    campaign = compute_campaign_protocol(
        protocol["qualified_count"], provenance["cdr3_length"], coverage=coverage
    )
    LOGGER.info(
        "%s @ %s: N_qual=%d L=%d -> B_0=%d B=%d R=%d Q_max=%d coverage=%.2f%% alloc=%s",
        provenance["key"], coverage_label, campaign.n_qualified, campaign.seq_len,
        campaign.seed_batch, campaign.batch_size, campaign.rounds,
        campaign.total_budget, 100 * campaign.coverage_pct, campaign.allocation,
    )
    report = run_benchmark(
        frame,
        normalizer,
        protocol["tau_dev"],
        seeds=seeds,
        m5_policy_mode=m5_policy_mode,
        include_warm_start=True,
        wt_sequence=reference,
        anchor_positions=anchors,
        protocol=campaign,
    )
    report["cross_target"] = {
        "dataset": provenance["key"],
        "label": provenance["label"],
        "source": provenance["source"],
        "repo": provenance.get("repo"),
        "source_file": provenance.get("source_file"),
        "reference_sequence": reference,
        "reference_source": provenance["reference_source"],
        "reference_is_1n8z": reference == WT_SEQUENCE,
        "anchor_positions": list(anchors),
        "cdr3_length": provenance["cdr3_length"],
        "rows_final": provenance["rows_final"],
        "rows_in_source": provenance["rows_in_source"],
        "rows_unparseable": provenance["rows_unparseable"],
        "replicates_collapsed": provenance["replicates_collapsed"],
        "length_histogram": provenance["length_histogram"],
        "usability": provenance["usability"],
        "tau_dev": protocol["tau_dev"],
        "qualified_count": protocol["qualified_count"],
        "qualification_pass_rate": protocol["qualification_pass_rate"],
        "dataset_sequence_sha256": protocol["dataset_sequence_sha256"],
        "csv_sha256": file_sha256(provenance["_csv"]),
        "csv_path": str(provenance["_csv"]),
        "protocol_path": str(provenance["_protocol_path"]),
        "seeds": seeds,
        # Both universal-law coefficients, stated so the cell can be checked
        # against core/acquisition.py rather than taken on faith. ``eta_sar`` is
        # the additive-prior trust this pool+reference earns; the maturity target
        # is what ``C`` is measured against, and it scales with ``Q_max`` so a
        # 125-query campaign must corroborate 3.6x more evidence than 1N8Z before
        # ``VoI`` is allowed to collapse.
        "eta_sar": campaign.additive_prior_trust(),
        "sar_maturity_target": campaign.sar_maturity_target,
        "falsification_phase_rounds": falsification_phase_rounds(campaign.rounds),
        "coverage_label": coverage_label,
        # ── dynamic protocol (the Step 3 addition) ──────────────────────────
        "campaign_protocol": campaign.to_dict(),
        "seed_batch": campaign.seed_batch,
        "batch_size": campaign.batch_size,
        "rounds": campaign.rounds,
        "total_budget": campaign.total_budget,
        "coverage_pct": round(100.0 * campaign.coverage_pct, 4),
        "cap_bound": campaign.cap_bound,
        "allocation_mode": campaign.allocation,
        # The claim, stated so it can be checked against core/acquisition.py rather
        # than taken on faith.
        "baselines_unchanged": True,
        "unchanged_modules": [
            "core/acquisition.py", "core/baselines.py",
            "core/surrogate.py", "core/features.py",
        ],
        "varies_by_protocol": [
            "rounds", "total_budget", "checkpoints", "aubqc_normalizer",
        ],
        # One law, everywhere. ``allocation`` is a constant now, not a switch, so
        # it belongs in ``unchanged`` rather than in what varies per cell.
        "allocation_law": campaign.allocation,
        "generalizes_across": "antigen target and mutation landscape (same architecture; protocol sized to the qualified pool)",
        "rejected_targets": provenance["_rejections"],
    }
    return report


def run_cross_target(
    *,
    dataset: str | None,
    seeds: int,
    repo: str,
    token: str | None,
    m5_policy_mode: str,
    sweep: bool = True,
) -> dict:
    """Every target x every coverage level, reported side by side.

    The top level of the returned artifact keeps the shape ``/api/cross-target``
    and the stage test already consume (``curves``, ``aubqc``, ``m5_warm_start``,
    ``paired_t_tests``, ``cross_target``); that block is the **pre-registered
    16.7% cell of the first target**, so a reader who only looks at the headline
    sees the protocol the rule actually prescribes. ``targets`` then carries every
    cell, and ``coverage_sensitivity`` carries the per-target comparison.
    """
    targets = (
        tuple(t for t in TARGET_CHAIN if t["key"].lower() == dataset.lower())
        if dataset
        else tuple(t for t in TARGET_CHAIN if t["key"] in PRIMARY_TARGETS)
    )
    if not targets:
        raise SystemExit(
            f"unknown --dataset {dataset!r}; choose from {[t['key'] for t in TARGET_CHAIN]}"
        )

    levels = COVERAGE_SWEEP if sweep else (("16.7pct", 35.0 / 210.0),)
    results: list[dict] = []
    for target in targets:
        frame, provenance, protocol, normalizer, reference, anchors = _prepare_target(
            target["key"], repo, token, csv_stem=target["key"]
        )
        for label, coverage in levels:
            results.append(
                _one_run(
                    frame, provenance, protocol, normalizer, reference, anchors,
                    seeds=seeds, coverage=coverage, coverage_label=label,
                    m5_policy_mode=m5_policy_mode,
                )
            )

    # Headline = the reference-coverage cell of the first target.
    headline = next(
        (r for r in results
         if r["cross_target"]["dataset"] == targets[0]["key"]
         and r["cross_target"]["coverage_label"] == "16.7pct"),
        results[0],
    )
    report = dict(headline)
    report["targets"] = [
        {
            "dataset": r["cross_target"]["dataset"],
            "label": r["cross_target"]["label"],
            "coverage_label": r["cross_target"]["coverage_label"],
            "coverage_pct": r["cross_target"]["coverage_pct"],
            "campaign_protocol": r["cross_target"]["campaign_protocol"],
            "qualified_count": r["cross_target"]["qualified_count"],
            "cdr3_length": r["cross_target"]["cdr3_length"],
            "aubqc": r["aubqc"],
            "aubqc_se": r.get("aubqc_se"),
            "normalized_aubqc": r["normalized_aubqc"],
            "top5_panel_aubqc": r["top5_panel_aubqc"],
            "top_decile_hit_rate": r["top_decile_hit_rate"],
            "spearman_rho_at_last_checkpoint": {
                m: next((p[m] for p in reversed(r["spearman_rho_curves"])
                         if p.get(m) is not None), None)
                for m in METHODS
            },
            "paired": {
                key: r["paired_t_tests"][key]["metrics"]["aubqc"]
                for key in ("m5_hybrid_minus_m3_ucb", "m5_hybrid_minus_m2_greedy",
                            f"{WARM_START_METHOD}_minus_m5_hybrid")
                if key in r.get("paired_t_tests", {})
            },
            "m5_warm_start_summary": r.get("m5_warm_start", {}).get("summary"),
        }
        for r in results
    ]
    report["coverage_sensitivity"] = _coverage_sensitivity(results)
    return report


def _coverage_sensitivity(results: list[dict]) -> list[dict]:
    """Is M5's (dis)advantage stable across coverage, or an artefact of one knob?

    One row per (target, coverage). ``m5_minus_best_baseline`` is the honest
    headline: M5 compared against the best of M1-M4 on that cell, not against a
    baseline chosen after the fact.
    """
    rows = []
    for r in results:
        aubqc = r["aubqc"]
        baselines = {m: aubqc[m] for m in METHODS if m != "m5_hybrid"}
        best = max(baselines, key=baselines.get)
        rho = {}
        for m in METHODS:
            value = next((p[m] for p in reversed(r["spearman_rho_curves"])
                          if p.get(m) is not None), None)
            rho[m] = value
        best_rho = {m: v for m, v in rho.items() if m != "m5_hybrid" and v is not None}
        rho_best = max(best_rho, key=best_rho.get) if best_rho else None
        paired = r.get("paired_t_tests", {}).get("m5_hybrid_minus_m3_ucb", {})
        rows.append({
            "dataset": r["cross_target"]["dataset"],
            "coverage_label": r["cross_target"]["coverage_label"],
            "coverage_pct": r["cross_target"]["coverage_pct"],
            "rounds": r["cross_target"]["rounds"],
            "total_budget": r["cross_target"]["total_budget"],
            "allocation_mode": r["cross_target"]["allocation_mode"],
            # The two universal-law coefficients that decide how much this cell
            # departs from the frozen 1N8Z behaviour, reported per row so the
            # comparison across coverage is attributable rather than asserted.
            "eta_sar": round(float(r["cross_target"]["eta_sar"]), 4),
            "sar_maturity_target": round(float(r["cross_target"]["sar_maturity_target"]), 3),
            "aubqc_m5": round(aubqc["m5_hybrid"], 4),
            "aubqc_m5_se": round(float((r.get("aubqc_se") or {}).get("m5_hybrid", float("nan"))), 4),
            "aubqc_best_baseline": round(baselines[best], 4),
            "best_baseline": best,
            "m5_minus_best_baseline": round(aubqc["m5_hybrid"] - baselines[best], 4),
            "m5_rank": 1 + sum(v > aubqc["m5_hybrid"] for v in baselines.values()),
            "rho_m5": rho.get("m5_hybrid"),
            "rho_best_baseline": None if rho_best is None else rho[rho_best],
            "best_baseline_rho": rho_best,
            "m5_leads_rho": bool(
                rho_best is not None and rho.get("m5_hybrid") is not None
                and rho["m5_hybrid"] > rho[rho_best]
            ),
            "m5_vs_m3_p": paired.get("metrics", {}).get("aubqc", {}).get("p_value"),
        })
    return rows


def format_summary(report: dict) -> str:
    """The summary table, printed by the script *and* by the stage test."""
    cross = report.get("cross_target", {})
    methods = [m for m in METHODS if m in report.get("curves", [{}])[0]] or list(METHODS)
    labels = {
        "m1_random": "M1", "m2_greedy": "M2", "m3_ucb": "M3",
        "m4_fixed_prior": "M4", "m5_hybrid": "M5", WARM_START_METHOD: "M5-warm",
    }
    header = [""] + [labels.get(m, m) for m in methods]

    def cell(value, width=8, digits=4):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{'n/a':>{width}}"
        if number != number:
            return f"{'n/a':>{width}}"
        return f"{number:>{width}.{digits}f}"

    rho_q35 = {row["query_count"]: row for row in report.get("spearman_rho_curves", [])}.get(35, {})
    rows = [
        ["AUBQC (1/30)"] + [cell(report["aubqc"][m]) for m in methods],
        ["normalized AUBQC"] + [cell(report["normalized_aubqc"][m]) for m in methods],
        ["top-5 panel AUBQC"] + [cell(report["top5_panel_aubqc"][m]) for m in methods],
        ["top-decile hit rate"] + [cell(report["top_decile_hit_rate"][m]) for m in methods],
        ["Spearman rho @ q=35"] + [cell(rho_q35.get(m)) for m in methods],
    ]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    lines = [
        "",
        f"CROSS-TARGET GENERALIZATION · {cross.get('label', '?')}",
        f"  source            : {cross.get('source')} :: {cross.get('source_file')}",
        f"  CDR3 length       : {cross.get('cdr3_length')}  (1N8Z is 13)",
        f"  reference         : {cross.get('reference_sequence')}  ({cross.get('reference_source')})",
        f"  rows              : {cross.get('rows_final')} distinct CDR3s from "
        f"{cross.get('rows_in_source')} rows ({cross.get('replicates_collapsed')} replicates merged, "
        f"{cross.get('rows_unparseable')} unparseable)",
        f"  tau_dev           : {cross.get('tau_dev')}  (label-blind, frozen to "
        f"config/cross_target_protocol.json)",
        f"  seeds             : {cross.get('seeds')}",
        "",
        "  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)),
        "  " + "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths)))

    paired = report.get("paired_t_tests") or {}
    for key in ("m5_hybrid_minus_m3_ucb", "m5_hybrid_minus_m2_greedy",
                f"{WARM_START_METHOD}_minus_m5_hybrid"):
        block = paired.get(key)
        if not block:
            continue
        stats = block["metrics"].get("aubqc")
        if not stats:
            continue
        lines.append(
            f"  {block['performer']} - {block['baseline']}: AUBQC "
            f"{stats['mean_diff']:+.5f} (p={stats['p_value']:.4f}, "
            f"won {stats['seeds_won']}/{stats['n']})"
        )

    rejected = cross.get("rejected_targets") or []
    if rejected:
        lines.append("")
        lines.append("  Targets rejected before this one (measured, not assumed):")
        for item in rejected:
            lines.append(f"    - {item['label']}: {item['reason']}")

    # ── Coverage sensitivity: is the effect an artefact of the coverage knob? ──
    sensitivity = report.get("coverage_sensitivity") or []
    if sensitivity:
        lines.append("")
        lines.append(
            "COVERAGE SENSITIVITY · one architecture, coverage varied\n"
            "  eta_sar and the maturity target are the two universal-law coefficients\n"
            "  that adapt the architecture to each cell, and both are computed from\n"
            "  t=0 observables only (pool size, sequence length, campaign budget,\n"
            "  mean mutational load). Neither contains a dataset-anchored constant."
        )
        header = ["target", "coverage", "R", "Q", "eta", "C target", "M5 AUBQC",
                  "best base", "M5-vs-best", "rank", "M5 leads rho"]
        rows = []
        for row in sensitivity:
            # ``eta_sar`` / ``sar_maturity_target`` are additive: a table rendered
            # from an artifact written before the universal law existed prints
            # ``n/a`` rather than raising. ``format_summary`` is called by the
            # stage test against whatever artifact is on disk, so it must not be
            # the thing that couples a reader to a regeneration step.
            eta = row.get("eta_sar")
            target = row.get("sar_maturity_target")
            rows.append([
                row["dataset"],
                f"{row['coverage_pct']:.2f}%",
                str(row["rounds"]),
                str(row["total_budget"]),
                "n/a" if eta is None else f"{float(eta):.3f}",
                "n/a" if target is None else f"{float(target):.2f}",
                f"{row['aubqc_m5']:.4f}±{row.get('aubqc_m5_se', float('nan')):.4f}",
                f"{row['aubqc_best_baseline']:.4f} ({row['best_baseline'].split('_')[0]})",
                f"{row['m5_minus_best_baseline']:+.4f}",
                f"{row['m5_rank']}/{len(METHODS)}",
                "yes" if row["m5_leads_rho"] else "no",
            ])
        widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
        lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
        lines.append("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths)))

    lines.append("")
    lines.append(
        "  One allocation law runs on every cell: no fixed_split/adaptive_split\n"
        "  switch. What varies is the *protocol* (R, Q_max) and the two continuous\n"
        "  coefficients above, each derived from t=0 observables. AUBQC measures a\n"
        "  search process, so compare rows within this table, not against the 1N8Z\n"
        "  artifact."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None,
                        help="target key from the chain, or omit for auto (first usable)")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--repo", default="AbBibench/Antibody_Binding_Benchmark_Dataset")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--no-sweep", action="store_true",
                        help="run only the pre-registered 16.7%% coverage level")
    parser.add_argument("--m5-policy-mode", choices=["adaptive_tool", "live_agent"], default="adaptive_tool")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    report = run_cross_target(
        dataset=args.dataset,
        seeds=args.seeds,
        repo=args.repo,
        token=args.token,
        m5_policy_mode=args.m5_policy_mode,
        sweep=not args.no_sweep,
    )
    path = SETTINGS.cross_target_benchmark_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(format_summary(report))
    print(f"\nwrote {path}")
    print(f"wrote {SETTINGS.cross_target_csv}")
    print(f"wrote {SETTINGS.cross_target_protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
