#!/usr/bin/env python3
"""Step 1: blind metadata audit of the AbBiBench corpus. **No M1..M5 evaluation.**

Why this script exists
----------------------
Picking a holdout target *after* seeing how M5 scores on it is the single most
common way a benchmark becomes unfalsifiable. So the selection is split in two,
and this is only the first half:

    Step 1 (this script)   describe every untouched target by metadata alone
    Step 2 (manual)        freeze the eligibility rule, then run a10ece5 on
                           whatever the rule selects

The audited quantities are exactly:

    (target_id, antigen, modal_cdrh3_length_L, reference_consensus,
     unique_cdrh3_pool_size, N_qual, binding_iqr)

plus the diagnostics needed to check that the audited gate is the gate we think
it is (``threshold_source``, ``wt_pass_fraction``, length histogram).

Binding labels are read for **one purpose only**: the distribution statistic
``binding_iqr``, which is part of the metadata table the eligibility rule is
allowed to see. No M1..M5 number is computed, no policy is consulted, and
``core.baselines`` is never imported — asserted below, because a single stray
import would let a benchmark number leak into a pre-registration decision.

How the gate is applied
-----------------------
``N_qual`` is not re-implemented here. The script calls the frozen
:func:`core.preregistration.freeze_protocol` on ``public_pool(frame)`` — the same
label-blind function that produced ``config/frozen_protocol.json`` — and reports
the ``qualified_count`` it returns. The protocol file is written to a scratch
directory, so auditing 16 targets never touches ``config/``.

That choice is load-bearing and is also why this script reports
``threshold_source`` per target: see :func:`gate_integrity`.

Usage
-----
    python scripts/audit_lockbox_metadata.py
    python scripts/audit_lockbox_metadata.py --min-nqual 150 --exclude-her2
    python scripts/audit_lockbox_metadata.py --out /tmp/lockbox_metadata.json

Writes nothing into ``results/``, ``config/`` or ``data/`` by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.preregistration import freeze_protocol
from data.datasets import align_cross_target_frame, mode_reference, public_pool

LOGGER = logging.getLogger("lockbox_audit")

REPO = "AbBibench/Antibody_Binding_Benchmark_Dataset"
PREFIX = "binding_affinity/"

#: Antigens as documented *inside this repository*. Anything absent is reported
#: as ``unknown`` rather than guessed: an antigen label invented here would end up
#: in the pre-registration document, and a wrong one is worse than a blank.
ANTIGEN: dict[str, str] = {
    "1n8z": "HER2 (trastuzumab)",
    "4d5_her2": "HER2 (4D5)",
    "2fjg": "HIV-1 gp120 (b12)",
    "4fqi_h1": "Influenza HA (CR9114)",
    "4fqi_h3": "Influenza HA (CR9114)",
    "3gbn_h1": "Influenza HA (CR6261)",
    "aayl49_ML": "lysozyme (anti-lysozyme)",
}

#: Targets whose CDR3 is HER2-directed. Excluded from a lockbox by definition:
#: the 1N8Z engine was developed and tuned against a HER2 paratope pool, so a
#: second HER2 target is not an out-of-distribution test.
HER2_TARGETS: frozenset[str] = frozenset({"1n8z", "4d5_her2"})

#: Targets that already appear in a published or recorded artifact, and where.
#: A "lockbox" that has already been scored is not a lockbox.
ALREADY_USED: dict[str, str] = {
    "1n8z": "primary benchmark (results/benchmark_100seeds.json)",
    "aayl49_ML": "results/cross_target_benchmark.json + results/attribution_ablation.json",
    "2fjg": "results/cross_target_benchmark.json + results/attribution_ablation.json",
    "4d5_her2": "data/raw/dev_hcdr3.csv, and LOCAL_FALLBACK in run_cross_target_benchmark",
}


def list_affinity_files(repo: str, token: str | None) -> list[str]:
    """Enumerate the corpus instead of hardcoding filenames.

    ``TARGET_CHAIN`` in ``run_cross_target_benchmark`` names six files by hand, so
    a target added upstream is invisible until someone edits that tuple. Auditing
    the repository listing is what makes "we looked at every target" a checkable
    claim rather than an assumption.
    """
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(PREFIX) and f.endswith(".csv"))


def target_id_from_file(filename: str) -> str:
    stem = Path(filename).stem
    suffix = "_benchmarking_data"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def gate_integrity(protocol: dict, cdr3_length: int) -> tuple[bool, str]:
    """Is ``N_qual`` the 0.50 quantile gate we believe it is?

    :func:`core.preregistration.freeze_protocol` picks ``tau_risk`` by scoring the
    **frozen 1N8Z 13-mer wild type** ``SRWGGDGFYAMDY`` against a normalizer that
    was fit on the *target's* pool. If that pool is 11-mers, those 13-mer
    components are clipped into an 11-mer's ``[min, max]`` box, and the resulting
    ``pass_fraction`` is a meaningless number that can still land inside the
    ``[0.25, 0.65]`` window. When it does, the gate silently becomes
    ``threshold_source = "wild_type"`` on a target that has no measured wild type
    in the pool, and ``N_qual`` is no longer the documented pool-median gate.

    Returns ``(ok, note)``. ``ok=False`` means: do not pre-register this target
    until the threshold source is fixed, because the eligibility count would not
    mean what the pre-registration says it means.
    """
    source = str(protocol.get("threshold_source"))
    if source == "pool_median":
        return True, "pool_median (0.50 quantile, documented gate)"
    if cdr3_length == 13:
        return True, "wild_type (13-mer pool; WT reference is same width)"
    return (
        False,
        f"wild_type on a {cdr3_length}-mer pool: frozen 1N8Z 13-mer WT was scored "
        "against a normalizer fit on this width. N_qual is NOT the 0.50 quantile gate.",
    )


def audit_one(
    source_df,
    *,
    target_id: str,
    token_note: str | None = None,
) -> dict:
    """Align one raw AbBiBench table and describe it. Never scores a policy."""
    frame, diagnostics = align_cross_target_frame(source_df, prefix=target_id.upper())
    reference = diagnostics["reference_sequence"]
    # The gate, computed by the frozen function, written to a scratch file.
    with tempfile.TemporaryDirectory(prefix="lockbox-protocol-") as scratch:
        protocol = freeze_protocol(
            public_pool(frame),
            output=Path(scratch) / f"{target_id}.json",
            is_synthetic=False,
        )
    ok, note = gate_integrity(protocol, int(diagnostics["cdr3_length"]))
    y = frame.y_bind.astype(float).to_numpy()
    q1, q3 = (float(v) for v in np.percentile(y, [25, 75]))
    # ``mode_reference`` is a *consensus*, not an assayed germline: AbBiBench
    # tables carry no WT CDR3 row. The name in the table says so on purpose.
    consensus = reference if diagnostics["reference_source"] == "consensus" else str(reference)
    return {
        "target_id": target_id,
        "antigen": ANTIGEN.get(target_id, "unknown (not documented in this repo)"),
        "modal_cdrh3_length_L": int(diagnostics["cdr3_length"]),
        "reference_consensus": consensus,
        "reference_source": diagnostics["reference_source"],
        "unique_cdrh3_pool_size": int(diagnostics["rows_final"]),
        "N_qual": int(protocol["qualified_count"]),
        "binding_iqr": round(q3 - q1, 6),
        # ── audit diagnostics (not eligibility inputs) ──────────────────────
        "binding_median": round(float(np.median(y)), 6),
        "binding_min": round(float(y.min()), 6),
        "binding_max": round(float(y.max()), 6),
        "tau_dev": protocol["tau_dev"],
        "tau_risk": protocol["tau_risk"],
        "threshold_source": protocol["threshold_source"],
        "wt_pass_fraction": protocol["wt_pass_fraction"],
        "qualification_pass_rate": protocol["qualification_pass_rate"],
        "gate_ok": ok,
        "gate_note": note,
        "rows_in_source": int(diagnostics["rows_in_source"]),
        "rows_unparseable": int(diagnostics["rows_unparseable"]),
        "replicates_collapsed": int(diagnostics["replicates_collapsed"]),
        "length_histogram": diagnostics["length_histogram"],
        "varying_positions": diagnostics["usability"]["varying_positions"],
        "usability_reason": diagnostics["usability"]["reason"],
        "usable_as_search_problem": bool(diagnostics["usability"]["usable"]),
        "already_used": ALREADY_USED.get(target_id),
        "is_her2": target_id in HER2_TARGETS,
        "token_note": token_note,
    }


def render(rows: list[dict], *, min_nqual: int, exclude_her2: bool) -> str:
    """The blind table, plus the Step-2 eligibility verdict underneath it."""
    header = [
        "target_id", "antigen", "L", "reference(consensus)", "pool", "N_qual",
        "binding_iqr", "gate", "used?",
    ]
    table = []
    for row in rows:
        table.append([
            row["target_id"],
            row["antigen"],
            str(row["modal_cdrh3_length_L"]),
            row["reference_consensus"],
            str(row["unique_cdrh3_pool_size"]),
            str(row["N_qual"]),
            f"{row['binding_iqr']:.3f}",
            "ok" if row["gate_ok"] else "BAD-SOURCE",
            "used" if row["already_used"] else "-",
        ])
    widths = [max(len(r[i]) for r in [header] + table) for i in range(len(header))]
    lines = ["", "LOCKBOX METADATA AUDIT (no M1..M5 scores computed)"]
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("  " + "  ".join("-" * w for w in widths))
    for row in table:
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))

    lines.append("")
    lines.append(f"STEP 2 · pre-registered gate: N_qual >= {min_nqual}"
                 + (", non-HER2" if exclude_her2 else "")
                 + ", not already scored")
    for row in rows:
        verdict = []
        if row["N_qual"] < min_nqual:
            verdict.append(f"N_qual {row['N_qual']} < {min_nqual}")
        if exclude_her2 and row["is_her2"]:
            verdict.append("HER2 (in-distribution)")
        if row["already_used"]:
            verdict.append(f"already used: {row['already_used']}")
        if not row["usable_as_search_problem"]:
            verdict.append(f"not a search problem: {row['usability_reason']}")
        if not row["gate_ok"]:
            verdict.append("GATE SOURCE INVALID — do not pre-register")
        if not verdict:
            exotic = "  <-- L NOT IN {11,13}" if row["modal_cdrh3_length_L"] not in (11, 13) else ""
            lines.append(f"  ELIGIBLE  {row['target_id']:<12} "
                         f"L={row['modal_cdrh3_length_L']:<3} N_qual={row['N_qual']:<5} "
                         f"iqr={row['binding_iqr']:.3f}{exotic}")
        else:
            lines.append(f"  excluded  {row['target_id']:<12} " + "; ".join(verdict))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--min-nqual", type=int, default=150)
    parser.add_argument("--exclude-her2", action="store_true", default=False)
    parser.add_argument("--out", default="/tmp/lockbox_metadata.json")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    # The guard that makes this a *blind* audit. If a future refactor pulls a
    # policy import in transitively, the audit stops being admissible evidence.
    forbidden = [m for m in sys.modules if m.startswith("core.baselines")]
    assert not forbidden, f"audit must not import policy code: {forbidden}"

    import pandas as pd

    from scripts.run_cross_target_benchmark import fetch_abbibench

    files = list_affinity_files(args.repo, args.token)
    LOGGER.info("%s exposes %d binding_affinity CSVs", args.repo, len(files))

    rows: list[dict] = []
    failures: list[dict] = []
    for filename in files:
        target_id = target_id_from_file(filename)
        try:
            source = fetch_abbibench(args.repo, filename, args.token)
            rows.append(audit_one(source, target_id=target_id))
        except Exception as exc:  # noqa: BLE001 - one bad target must not abort the audit
            LOGGER.warning("%s: %s: %s", target_id, type(exc).__name__, exc)
            failures.append({
                "target_id": target_id, "file": filename,
                "error": f"{type(exc).__name__}: {exc}",
            })

    rows.sort(key=lambda r: (-r["N_qual"], r["target_id"]))
    report = {
        "corpus": args.repo,
        "files_enumerated": files,
        "min_nqual": args.min_nqual,
        "exclude_her2": args.exclude_her2,
        "rows": rows,
        "failures": failures,
        "notes": [
            "reference_consensus is the per-position modal residue of the pool, NOT an "
            "assayed germline; AbBiBench tables carry no WT CDR3 row.",
            "N_qual comes from core.preregistration.freeze_protocol on the label-blind "
            "pool; a BAD-SOURCE gate means threshold_source='wild_type' on a non-13-mer "
            "pool, which is not the 0.50 quantile gate.",
            "binding_iqr is reported as metadata; it is computed from that target's own "
            "labels and is not a cross-target comparable.",
            "No M1..M5 curve, AUBQC or rho was computed by this script.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(render(rows, min_nqual=args.min_nqual, exclude_her2=args.exclude_her2))
    if failures:
        print(f"\n  {len(failures)} target(s) could not be aligned; see {args.out}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
