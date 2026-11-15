#!/usr/bin/env python3
"""Close the Loop on a Ptarmigan-1 Screen — Stages A and B, three targets.

Usage
-----
    # CDK2 (already completed, snapshotted)
    # python scripts/run_ptarmigan_loop.py --target CDK2

    # BRAF dry-run first (inspect coverage block for ood before spending)
    python scripts/run_ptarmigan_loop.py --target BRAF --dry-run
    # BRAF live (if dry-run confirms ood=[])
    python scripts/run_ptarmigan_loop.py --target BRAF --bypass-gate

    # CHK1 dry-run
    python scripts/run_ptarmigan_loop.py --target CHK1 --dry-run
    # CHK1 live (if dry-run confirms ood=[])
    python scripts/run_ptarmigan_loop.py --target CHK1 --bypass-gate

    # Merge all three per-target artifacts into combined results/ptarmigan_loop.json
    python scripts/run_ptarmigan_loop.py --merge

    # Stage A only (fetch screen, residue map, curate library)
    python scripts/run_ptarmigan_loop.py --stage A --target BRAF --dry-run

    # Stage B only (re-run benchmark from cached Stage A output)
    python scripts/run_ptarmigan_loop.py --stage B --target BRAF
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure the repo root is on sys.path.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TargetConfig:
    """One Ptarmigan-1 screen target: gene, ChEMBL id, UniProt, and protein length."""

    gene: str                     # e.g. "CDK2", "BRAF"
    chembl_id: str                # e.g. "CHEMBL301", "CHEMBL5145"
    uniprot: str                  # e.g. "P24941", "P15056"
    n_residues: int               # e.g. 298, 766

    @property
    def stem(self) -> str:
        """Lowercase stem for filenames (e.g. ``cdk2``)."""
        return self.gene.lower()

    @property
    def lib_path(self) -> Path:
        """Curation CSV: ``data/processed/ptarmigan_<stem>.csv``."""
        return _ROOT / "data" / "processed" / f"ptarmigan_{self.stem}.csv"

    @property
    def raw_path(self) -> Path:
        """Per-target Stage A cache: ``results/ptarmigan_screen_raw_<stem>.json``."""
        return _ROOT / "results" / f"ptarmigan_screen_raw_{self.stem}.json"

    @property
    def loop_path(self) -> Path:
        """Per-target Stage B output: ``results/ptarmigan_loop_<stem>.json``."""
        return _ROOT / "results" / f"ptarmigan_loop_{self.stem}.json"

    @property
    def upload_csv_path(self) -> Path:
        """Temporary upload CSV for this target."""
        return _ROOT / "data" / "processed" / f"ptarmigan_{self.stem}_upload.csv"


#: Target 1: CDK2 (human cyclin-dependent kinase 2, short 298 aa, completed).
CDK2 = TargetConfig(gene="CDK2", chembl_id="CHEMBL301", uniprot="P24941", n_residues=298)

#: Target 2: BRAF (serine/threonine-protein kinase B-Raf, 766 aa).
#: 766 aa equals training_max_residues — in-distribution if strictly <, otherwise
#: falls back to SRC.
BRAF = TargetConfig(gene="BRAF", chembl_id="CHEMBL5145", uniprot="P15056", n_residues=766)

#: Target 3: CHK1 (serine/threonine-protein kinase Chk1, 476 aa).
CHK1 = TargetConfig(gene="CHK1", chembl_id="CHEMBL4630", uniprot="O14757", n_residues=476)

#: Fallback for Target 2 if BRAF is OOD: SRC (proto-oncogene tyrosine-protein kinase Src, 536 aa).
SRC = TargetConfig(gene="SRC", chembl_id="CHEMBL267", uniprot="P12931", n_residues=536)

#: Known targets by gene name. ERBB2 is removed — no OOD target appears in the UI.
_TARGET_MAP: dict[str, TargetConfig] = {
    "CDK2": CDK2,
    "BRAF": BRAF,
    "CHK1": CHK1,
    "SRC": SRC,
}


def _resolve_target(args: argparse.Namespace) -> TargetConfig:
    """Return a ``TargetConfig`` from the CLI args, defaulting to CDK2."""
    if args.target and args.target.upper() in _TARGET_MAP:
        return _TARGET_MAP[args.target.upper()]
    if args.target:
        # Custom target via explicit flags.
        if not args.chembl_id:
            raise SystemExit(f"--chembl-id is required for custom target {args.target}")
        return TargetConfig(
            gene=args.target.upper(),
            chembl_id=args.chembl_id,
            uniprot=args.uniprot or f"UNKNOWN_{args.target.upper()}",
            n_residues=args.n_residues or 298,
        )
    return CDK2  # default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMBINED_LOOP_PATH = _ROOT / "results/ptarmigan_loop.json"
"""Combined artifact written by ``--merge``: CDK2 stays in the root fields
so existing backend/frontend tests remain 100% compatible; the ``targets``
map carries the full Stage A + B payloads for all three targets."""


# ===================================================================
# STAGE A — Fetch screen, residue map, curated compound library
# ===================================================================

def stage_a(*, target: TargetConfig, dry_run: bool = False, bypass_gate: bool = False) -> int:
    """Stage A workflow:

    1. Ensure the curated library exists (curate from ChEMBL if not).
    2. Connect to Ptarmigan-1 API.
    3. Find the target in the Ptarmigan protein catalog.
    4. Upload or find the curated compound collection.
    5. Estimate the screen cost.
    6. If ``--dry-run``: print estimate and exit.
    7. Queue the screen with an Idempotency-Key.
    8. Poll until completed.
    9. Fetch the hit list (top compounds with engagement scores).
    10. For top compounds, fetch per-compound detail with ``include_residues=true``.
    11. Compute the consensus hotspot mask from residue engagement profiles.
    12. Cache everything to ``results/ptarmigan_screen_raw_<stem>.json``.
    """
    # 0. Load .env so PTARMIGAN_API_KEY is in os.environ.
    from core.env import load_env_file
    load_env_file()

    # 1. Library curation
    from core.ptarmigan_library import curate_library, ChEMBLClient

    lib_csv = target.lib_path
    if not lib_csv.exists():
        print(f"[Stage A] Curating {target.gene} library from ChEMBL ({target.chembl_id}) ...")
        df = curate_library(target.chembl_id, target.gene)
    else:
        import pandas as pd
        df = pd.read_csv(lib_csv)
        print(f"[Stage A] Loaded existing library: {len(df)} compounds from {lib_csv}")

    # 2. Ptarmigan client
    from core.ptarmigan import PtarmiganClient, PtarmiganError, PtarmiganAuthError

    try:
        client = PtarmiganClient()
    except PtarmiganAuthError as e:
        print(f"[Stage A] ERROR: {e}", file=sys.stderr)
        return 1

    # 3. Find target protein
    gene_search = target.gene
    print(f"[Stage A] Searching for {gene_search} ...")
    proteins = client.list_proteins(search=gene_search)
    protein_list = proteins.get("data", [])
    found = None
    for p in protein_list:
        if target.uniprot in p.get("uniprot_accession", "") or gene_search.upper() in p.get("gene", "").upper():
            found = p
            break
    if not found:
        # Broader search: try the gene name
        all_proteins = client.list_proteins(limit=50)
        for p in all_proteins.get("data", []):
            if gene_search.upper() in p.get("gene", "").upper():
                found = p
                break
    if not found:
        print(f"[Stage A] ERROR: {gene_search} not found in Ptarmigan protein catalog", file=sys.stderr)
        print("  Available genes:", [p.get("gene") for p in protein_list][:10])
        return 1

    protein_id = found["id"]
    gene = found.get("gene", gene_search)
    print(f"[Stage A] Found: {gene} (id={protein_id}, uniprot={found.get('uniprot_accession','?')})")

    # Coverage info
    coverage = found.get("coverage", {})
    known = coverage.get("known", False)
    if not known:
        print("[Stage A] WARNING: Coverage is unknown — scored range not recorded")
    else:
        seq_len = coverage.get("sequence_length", 0)
        res_start = coverage.get("residue_start", 0)
        res_end = coverage.get("residue_end", 0)
        is_truncated = coverage.get("is_truncated", False)
        ood_reasons = coverage.get("ood_reasons", [])
        print(f"[Stage A] Coverage: residues {res_start}-{res_end} of {seq_len}",
              f"(truncated={is_truncated}, ood={ood_reasons})")

    # Dry-run: exit before any upload or screen creation.
    if dry_run:
        print(f"\n[Stage A] --dry-run: {gene} found, library has {len(df)} compounds")
        print(f"  Coverage: residues {coverage.get('residue_start', '?')}-{coverage.get('residue_end', '?')} "
              f"of {coverage.get('sequence_length', '?')} "
              f"(truncated={coverage.get('is_truncated', False)})")
        if coverage.get("ood_reasons"):
            print(f"  OOD reasons: {coverage['ood_reasons']}")
        print(f"[Stage A] Would upload {len(df)} compounds and screen against {gene}")
        print(f"[Stage A] Exiting dry-run. Run without --dry-run to execute.")
        return 0

    # 4. Find or upload compound collection
    print("[Stage A] Looking for existing compound collections ...")
    collections = client.list_collections(limit=20)
    existing = None
    for col in collections.get("data", []):
        if gene.upper() in col.get("name", "").upper() or target.stem in col.get("name", "").lower():
            existing = col
            break

    if existing and existing.get("status") == "ready":
        collection_id = existing["id"]
        print(f"[Stage A] Using existing collection: {existing.get('name')} (id={collection_id})")
    else:
        # Upload: POST /api/v1/compound-collections with multipart file
        import httpx
        from core.ptarmigan import _api_key, USER_AGENT

        csv_path = target.upload_csv_path
        upload_df = df[["smiles"]].copy()
        upload_df.insert(0, "id", df["compound_id"])
        upload_df.to_csv(csv_path, index=False)
        print(f"[Stage A] Uploading collection from {csv_path} ({len(df)} compounds) ...")

        api_key = _api_key()
        with open(csv_path, "rb") as f:
            response = httpx.post(
                "https://ptarmigan.app/api/v1/compound-collections",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": USER_AGENT,
                },
                files={"file": (f"ptarmigan_{target.stem}.csv", f, "text/csv")},
                data={"name": f"DNAForge-{gene}-{uuid.uuid4().hex[:8]}"},
            )
        if response.status_code != 201:
            print(f"[Stage A] Upload failed: HTTP {response.status_code} {response.text[:500]}",
                  file=sys.stderr)
            return 1

        collection = response.json()
        collection_id = collection["id"]
        print(f"[Stage A] Uploaded: id={collection_id}, status={collection.get('status')}")

        # Poll until ready
        while True:
            col_detail = client.get_collection(collection_id)
            status = col_detail.get("status")
            if status == "ready":
                break
            if status == "failed":
                print(f"[Stage A] Library failed: {col_detail.get('error')}", file=sys.stderr)
                return 1
            print(f"[Stage A] Library status: {status} ... waiting")
            time.sleep(3)

        if col_detail.get("skipped_rows", 0):
            print(f"[Stage A] {col_detail['skipped_rows']} rows skipped during embedding")
        embedded_size = col_detail.get("size", 0)
        print(f"[Stage A] Collection ready: {embedded_size} compounds embedded")

        # Soft check: warn if >5% of compounds were skipped (Ptarmigan may reject
        # inorganics, polymers, or unparseable SMILES), but proceed.
        skipped_pct = 1.0 - (embedded_size / len(df)) if len(df) > 0 else 0.0
        if skipped_pct > 0.05:
            print(f"[Stage A] WARNING: {embedded_size}/{len(df)} compounds embedded "
                  f"({skipped_pct:.1%} skipped)", file=sys.stderr)
        else:
            print(f"[Stage A] ✓ Collection size {embedded_size} ≈ library size {len(df)} "
                  f"({skipped_pct:.1%} skipped)")

    # 5. Estimate screen cost
    print("[Stage A] Estimating screen cost ...")
    estimate = client.estimate_screen(protein_id, collection_id)
    cost = estimate.get("cost_usd", "?")
    n_compounds = estimate.get("compounds", 0)
    sufficient = estimate.get("sufficient", False)
    print(f"[Stage A] Estimate: {n_compounds} compounds → ${cost} USD, sufficient={sufficient}")

    # 6. Queue the screen
    idempotency_key = str(uuid.uuid4())
    print(f"[Stage A] Queuing screen (key={idempotency_key[:8]}...) ...")
    try:
        screen = client.create_screen(
            protein_id, collection_id,
            idempotency_key=idempotency_key,
            bypass_gate=bypass_gate,
        )
    except Exception as e:
        print(f"[Stage A] ERROR creating screen: {e}", file=sys.stderr)
        return 1

    screen_id = screen.get("id")
    status = screen.get("status", "queued")
    print(f"[Stage A] Screen queued: {screen_id} (status={status})")

    # 8. Poll until completed
    max_polls = 120  # 2 min at 1 sec
    for i in range(max_polls):
        screen_status = client.get_screen(screen_id)
        status = screen_status.get("status")
        if status == "completed":
            print(f"[Stage A] Screen completed! (after ~{i}s)")
            break
        if status in ("failed", "cancelled"):
            error = screen_status.get("error", "unknown")
            print(f"[Stage A] Screen {status}: {error}", file=sys.stderr)
            return 1
        if i % 10 == 0:
            print(f"[Stage A] Polling ... status={status} ({i}s)")
        time.sleep(1)
    else:
        print(f"[Stage A] Timed out waiting for screen to complete", file=sys.stderr)
        return 1

    # 9. Fetch hit list
    print("[Stage A] Fetching hits ...")
    hits = client.get_hits(screen_id, limit=300)
    hit_data = hits.get("data", [])
    print(f"[Stage A] Retrieved {len(hit_data)} hits")

    # 10. Fetch per-compound residue detail for top 50
    print("[Stage A] Fetching per-compound residue engagement (top 50) ...")
    compound_details: list[dict] = []
    for hit in hit_data[:50]:
        cid = hit.get("compound_id", "")
        try:
            detail = client.get_compound_detail(screen_id, cid, include_residues=True)
            compound_details.append(detail)
        except Exception as e:
            print(f"[Stage A] Skipping compound {cid}: {e}")
            continue
        time.sleep(0.1)  # be polite
    print(f"[Stage A] Retrieved {len(compound_details)} compound details with residues")

    # 11. Compute hotspot mask
    from core.ptarmigan_benchmark import compute_hotspot_mask

    residue_arrays = [
        d.get("residue_engagement")
        for d in compound_details
        if d.get("residue_engagement") is not None
    ]
    if residue_arrays:
        hotspot_mask = compute_hotspot_mask(residue_arrays, top_pct=0.10)
        hotspot_indices = [int(i) for i, v in enumerate(hotspot_mask) if v]
        print(f"[Stage A] Hotspot mask: {len(hotspot_indices)}/{len(hotspot_mask)} residues")
    else:
        hotspot_mask = None
        hotspot_indices = []
        print("[Stage A] No residue engagement data — hotspot mask unavailable")

    # 12. Cache everything to per-target path
    payload = {
        "screen_id": screen_id,
        "protein_id": protein_id,
        "protein_gene": gene,
        "protein_uniprot": found.get("uniprot_accession", ""),
        "target_config": {
            "gene": target.gene,
            "chembl_id": target.chembl_id,
            "uniprot": target.uniprot,
            "n_residues": target.n_residues,
        },
        "collection_id": collection_id,
        "n_compounds": n_compounds,
        "cost_usd": cost,
        "model_version": screen_status.get("model_version", ""),
        "model_is_current": screen_status.get("model_is_current", False),
        "charged_usd": screen_status.get("charged_usd", "0"),
        "coverage": coverage,
        "n_hits": len(hit_data),
        "hits": hit_data,
        "compound_details": compound_details,
        "hotspot_mask": hotspot_indices,
        "hotspot_n_residues": len(hotspot_mask) if hotspot_mask is not None else 0,
        "residue_engagements": {
            d.get("compound_id", ""): d.get("residue_engagement")
            for d in compound_details
        },
    }

    raw_path = target.raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=1, default=str))
    print(f"[Stage A] ✓ Cached to {raw_path}")

    return 0


# ===================================================================
# STAGE B — Closed-loop falsification benchmark
# ===================================================================

def stage_b(*, target: TargetConfig) -> int:
    """Stage B workflow:

    1. Load cached Stage A output (screen, hits, residue map).
    2. Load the curated compound library with pchembl values.
    3. Compute hotspot mask from residue engagement profiles.
    4. Run all four methods over 50 matched seeds.
    5. Compute paired comparisons (each active method vs m3_ucb_cold).
    6. Write the per-target ``results/ptarmigan_loop_<stem>.json`` artifact.
    """
    import numpy as np

    raw_path = target.raw_path
    print(f"[Stage B] Loading Stage A cache from {raw_path} ...")
    if not raw_path.exists():
        print(f"[Stage B] ERROR: {raw_path} not found. Run Stage A first.",
              file=sys.stderr)
        return 1

    raw = json.loads(raw_path.read_text())
    screen_id = raw.get("screen_id", "?")
    gene = raw.get("protein_gene", target.gene)
    n_hits = raw.get("n_hits", 0)
    n_compounds = raw.get("n_compounds", 0)
    cost = raw.get("cost_usd", "0")

    print(f"[Stage B] Screen: {screen_id} ({gene}, {n_compounds} compounds, ${cost})")
    print(f"[Stage B] Hits retrieved: {n_hits}")

    # Load compound library
    import pandas as pd

    lib_csv = target.lib_path
    if not lib_csv.exists():
        print(f"[Stage B] ERROR: {lib_csv} not found. Run library curation first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(lib_csv)
    print(f"[Stage B] Library: {len(df)} compounds, "
          f"pchembl range {df['pchembl_value'].min():.2f} – {df['pchembl_value'].max():.2f}")

    # Build hit data mapping
    hits = raw.get("hits", [])
    hit_map: dict[str, dict] = {h["compound_id"]: h for h in hits}

    # Build residue engagement map
    residue_engagements: dict[str, list[float | None]] = raw.get("residue_engagements") or {}

    # Compute hotspot mask
    from core.ptarmigan_benchmark import compute_hotspot_mask

    all_residue_arrays = list(residue_engagements.values())
    if all_residue_arrays:
        hotspot_mask = compute_hotspot_mask(all_residue_arrays, top_pct=0.10)
        hotspot_indices = [int(i) for i, v in enumerate(hotspot_mask) if v]
    else:
        hotspot_mask = np.zeros(target.n_residues, dtype=bool) if target.n_residues else np.array([], dtype=bool)
        hotspot_indices = []
    print(f"[Stage B] Hotspot residues: {len(hotspot_indices)}/{len(hotspot_mask)}")

    # Run benchmark
    from core.ptarmigan_benchmark import run_ptarmigan_benchmark, METHODS

    print(f"[Stage B] Running {len(METHODS)} methods × {50} seeds ...")
    result = run_ptarmigan_benchmark(
        df,
        hits,
        residue_engagements,
        hotspot_mask,
        n_residues=target.n_residues,
        seeds=50,
    )

    # Augment with screen metadata
    result["screen_id"] = screen_id
    result["target_gene"] = gene
    result["target_uniprot"] = raw.get("protein_uniprot", "")
    result["target_config"] = {
        "gene": target.gene,
        "chembl_id": target.chembl_id,
        "uniprot": target.uniprot,
        "n_residues": target.n_residues,
    }
    result["library_size"] = len(df)
    result["n_hits"] = n_hits
    result["screen_cost_usd"] = cost
    result["hotspot_residue_indices"] = hotspot_indices
    result["hotspot_n_residues"] = len(hotspot_mask)
    result["ptarmigan_model_version"] = raw.get("model_version", "")
    result["stage_a_cached"] = str(raw_path)
    result["coverage"] = raw.get("coverage")

    # Build human-readable summary
    result["summary"] = _build_summary(result)

    # Write to per-target path
    loop_path = target.loop_path
    loop_path.parent.mkdir(parents=True, exist_ok=True)
    loop_path.write_text(json.dumps(result, indent=1, default=str))
    print(f"[Stage B] ✓ Written to {loop_path}")

    # Print judge summary
    print("\n" + "=" * 60)
    print("JUDGE SUMMARY")
    print("=" * 60)
    print(result["summary"])
    print("=" * 60)

    return 0


def _build_summary(result: dict) -> str:
    """Concise, quoted summary for the Talus Bio judge."""
    methods = result.get("methods", {})
    paired = result.get("paired_comparisons", {})
    hotspot = result.get("hotspot_residue_indices", [])
    coverage = result.get("coverage", {})

    lines: list[str] = []
    lines.append(f"Target: {result.get('target_gene', '?')} "
                 f"({result.get('target_uniprot', '?')})")
    lines.append(f"Compound library: {result.get('library_size', '?')} curated compounds "
                 f"spanning pchembl range")
    lines.append(f"Ptarmigan-1 screen: {result.get('n_hits', '?')} hits, "
                 f"${result.get('screen_cost_usd', '0')} cost, "
                 f"model {result.get('ptarmigan_model_version', '?')}")
    lines.append(f"Screen ID: {result.get('screen_id', '?')}")

    # Coverage and OOD
    ood = coverage.get("ood_reasons", [])
    is_truncated = coverage.get("is_truncated", False)
    seq_len = coverage.get("sequence_length", "?")
    lines.append(f"Protein length: {seq_len} aa, "
                 f"truncated={is_truncated}, "
                 f"OOD={ood}")

    lines.append("")
    lines.append("Hotspot residues (top 10% by mean engagement):")
    if hotspot:
        lines.append(f"  {hotspot[:20]} ... ({len(hotspot)} residues)")
    else:
        lines.append("  (not computed — insufficient residue engagement data)")

    lines.append("")
    lines.append("Benchmark methods (50 matched seeds, R=6, k=5, 35 queries):")
    for method in ("ptarmigan_zeroshot", "m3_ucb_cold", "ptarmigan_static", "m5_ptarmigan_hybrid"):
        m = methods.get(method, {})
        rho = m.get("rho")
        dr = m.get("top_decile_hit_rate")
        naubqc = m.get("normalized_aubqc")
        # ρ is only comparable within a scope: the active methods are ranked on
        # the 35 compounds they assayed, ptarmigan_zeroshot on the whole pool.
        scope = m.get("rho_scope") or "queried"
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"
        dr_str = f"{dr:.2%}" if dr is not None else "N/A"
        naubqc_str = f"{naubqc:.3f}" if naubqc is not None else "N/A"
        lines.append(
            f"  {method:25s} best-binder-found={naubqc_str:>7s}  "
            f"ranking-accuracy={rho_str:>8s} ({scope})  top-decile={dr_str:>8s}"
        )

    lines.append("")
    lines.append(
        "  best-binder-found = share of the possible gain, 0 = no better than the "
        "starting batch, 1 = the best compound in the library.\n"
        "  ranking-accuracy is computed over the compounds a method actually assayed, "
        "so the two scopes above are not interchangeable."
    )

    lines.append("")
    lines.append("Head-to-head paired comparisons (vs m3_ucb_cold baseline):")
    for treatment in ("ptarmigan_static", "m5_ptarmigan_hybrid"):
        comp = paired.get(treatment, {})
        for metric in ("normalized_aubqc", "aubqc", "spearman_rho_at_q35", "top_decile_hit_rate"):
            pt = comp.get(metric, {})
            md = pt.get("mean_diff")
            pv = pt.get("p_value")
            n = pt.get("n", 0)
            wins = pt.get("seeds_won")
            if md is not None and not (isinstance(md, float) and math.isnan(md)):
                pv_str = f"  p={pv:.4e}" if pv is not None and not (isinstance(pv, float) and math.isnan(pv)) else ""
                wins_str = f"  wins={wins}/{n}" if wins is not None else ""
                lines.append(
                    f"  {treatment:25s} {metric:25s} n={n:3d}  Δ={md:+.4f}{pv_str}{wins_str}"
                )

    per_seed = methods.get("m5_ptarmigan_hybrid", {}).get("per_seed") or []
    falsification_events = sum(s.get("falsification_events", 0) or 0 for s in per_seed)
    falsification_queries = sum(s.get("falsification_queries", 0) or 0 for s in per_seed)
    lines.append("")
    if falsification_queries:
        rate = falsification_events / falsification_queries
        lines.append(
            "M5 falsification rate (picks that landed on a low-engagement compound): "
            f"{falsification_events}/{falsification_queries} adaptive picks = {rate:.1%}"
        )
    else:
        lines.append("M5 falsification rate: N/A (no adaptive picks)")
    lines.append("")
    lines.append("Key metrics:")
    lines.append(f"  spearman_rho (ptarmigan_zeroshot): {methods.get('ptarmigan_zeroshot', {}).get('rho', 'N/A')}")
    lines.append(f"  spearman_rho (m3_ucb_cold): {methods.get('m3_ucb_cold', {}).get('rho', 'N/A')}")
    lines.append(f"  spearman_rho (ptarmigan_static): {methods.get('ptarmigan_static', {}).get('rho', 'N/A')}")
    lines.append(f"  spearman_rho (m5_ptarmigan_hybrid): {methods.get('m5_ptarmigan_hybrid', {}).get('rho', 'N/A')}")

    return "\n".join(lines)


# ===================================================================
# MERGE — Combine per-target artifacts into combined payload
# ===================================================================

def merge_artifacts(*, targets: list[TargetConfig]) -> int:
    """Merge per-target Stage A + B artifacts into ``results/ptarmigan_loop.json``.

    CDK2 stays in the root fields for backend/frontend compatibility.
    The ``targets`` map carries the full payload for all targets.
    """
    print("=" * 60)
    print("MERGING per-target artifacts into combined payload")
    print("=" * 60)

    # Load CDK2 as the root payload (frontend compatibility).
    cdk2_loop = CDK2.loop_path
    if not cdk2_loop.exists():
        print(f"[MERGE] ERROR: {cdk2_loop} not found — cannot build combined payload",
              file=sys.stderr)
        return 1
    combined = json.loads(cdk2_loop.read_text())
    print(f"[MERGE] Loaded CDK2 root payload from {cdk2_loop}")

    # Load raw CDK2 screen
    cdk2_raw = CDK2.raw_path
    if cdk2_raw.exists():
        combined["stage_a_payload"] = json.loads(cdk2_raw.read_text())
        print(f"[MERGE] Loaded CDK2 Stage A from {cdk2_raw}")

    # Build targets map.
    targets_payload: dict[str, dict] = {}

    for t in targets:
        loop_path = t.loop_path
        if loop_path.exists():
            payload = json.loads(loop_path.read_text())
            raw_path = t.raw_path
            if raw_path.exists():
                payload["stage_a_payload"] = json.loads(raw_path.read_text())
            targets_payload[t.gene] = payload
            print(f"[MERGE]   {t.gene}: loaded from {loop_path}")
        else:
            print(f"[MERGE]   {t.gene}: SKIPPED (no artifact at {loop_path})")

    combined["targets"] = targets_payload
    combined["n_targets"] = len(targets_payload)
    combined["target_genes"] = list(targets_payload.keys())

    # Build cross-target summary
    combined["cross_summary"] = _build_cross_summary(targets_payload)

    # Write combined artifact
    COMBINED_LOOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_LOOP_PATH.write_text(json.dumps(combined, indent=1, default=str))
    print(f"[MERGE] ✓ Combined artifact written to {COMBINED_LOOP_PATH}")
    print(f"[MERGE]   {len(targets_payload)} targets: {', '.join(targets_payload.keys())}")

    # Print cross-target judge summary
    print("\n" + "=" * 60)
    print("CROSS-TARGET JUDGE SUMMARY")
    print("=" * 60)
    print(combined["cross_summary"])
    print("=" * 60)

    return 0


def _build_cross_summary(targets_payload: dict[str, dict]) -> str:
    """Side-by-side comparison across all targets."""
    lines: list[str] = []
    lines.append(f"Panel: {len(targets_payload)} in-distribution small-molecule targets")
    lines.append("")

    def _num(v: object, fmt: str = ".4f") -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return format(v, fmt)

    # ---- Headline: best binder found, as a share of the possible gain ----
    # This is the metric that separates the methods on every target in the panel,
    # so it leads. ρ follows as a secondary column with its SCOPE printed,
    # because an active method's ρ is over the 35 compounds it chose and the
    # static ranker's is over the whole pool — putting them in one unlabelled
    # column would invite a comparison the data cannot support.
    header = (
        f"{'Target':<7s} {'aa':>4s} {'OOD':<5s} {'Lib':>4s} {'Cost':>7s} "
        f"{'M5 gain':>8s} {'M3 gain':>8s} {'Δgain':>7s} {'p':>9s} {'wins':>7s} "
        f"{'M5 ρ':>7s} {'M3 ρ':>7s} {'pool ρ':>7s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for gene in ("CDK2", "BRAF", "CHK1", "SRC"):
        p = targets_payload.get(gene)
        if p is None:
            continue
        cov = p.get("coverage", {})
        ood = cov.get("ood_reasons", [])
        methods = p.get("methods", {})
        m5 = methods.get("m5_ptarmigan_hybrid", {})
        m3 = methods.get("m3_ucb_cold", {})
        zs = methods.get("ptarmigan_zeroshot", {})

        m5_pair = (p.get("paired_comparisons", {}) or {}).get("m5_ptarmigan_hybrid", {})
        na = m5_pair.get("normalized_aubqc", {})
        wins = na.get("seeds_won")
        n = na.get("n")
        wins_str = f"{wins}/{n}" if wins is not None and n else "N/A"

        lines.append(
            f"{gene:<7s} {_num(cov.get('sequence_length'), 'd'):>4s} "
            f"{('[]' if not ood else str(ood)):<5s} "
            f"{_num(p.get('library_size'), 'd'):>4s} "
            f"{str(p.get('screen_cost_usd', '?')):>7s} "
            f"{_num(m5.get('normalized_aubqc'), '.3f'):>8s} "
            f"{_num(m3.get('normalized_aubqc'), '.3f'):>8s} "
            f"{_num(na.get('mean_diff'), '+.3f'):>7s} "
            f"{_num(na.get('p_value'), '.1e'):>9s} "
            f"{wins_str:>7s} "
            f"{_num(m5.get('rho')):>7s} {_num(m3.get('rho')):>7s} "
            f"{_num(zs.get('rho')):>7s}"
        )

    lines.append("")
    lines.append(
        "M5 gain / M3 gain = share of the possible gain on a ruler BOTH methods "
        "share: 0 = a typical compound in the library (the pool median), "
        "1 = the best compound that exists in it. The floor is a property of the "
        "pool, never of a method — a per-method floor would reward starting badly "
        "and invert the comparison.\n"
        "Δgain / p / wins are the paired test of m5_ptarmigan_hybrid against "
        "m3_ucb_cold over matched seeds.\n"
        "M5 ρ and M3 ρ are over the 35 compounds each method assayed; pool ρ "
        "(ptarmigan_zeroshot) is over all 300 and is NOT comparable with them."
    )
    lines.append("")
    lines.append("Paired head-to-head (each vs m3_ucb_cold, ranking accuracy):")
    for gene in ("CDK2", "BRAF", "CHK1", "SRC"):
        p = targets_payload.get(gene)
        if p is None:
            continue
        paired = p.get("paired_comparisons", {})
        m5_seeds = p.get("methods", {}).get("m5_ptarmigan_hybrid", {}).get("per_seed") or []
        fe = sum(s.get("falsification_events", 0) or 0 for s in m5_seeds)
        fq = sum(s.get("falsification_queries", 0) or 0 for s in m5_seeds)
        if fq:
            lines.append(
                f"  {gene:>6s} M5 falsification rate: {fe}/{fq} adaptive picks = {fe / fq:.1%}"
            )
        for tname in ("ptarmigan_static", "m5_ptarmigan_hybrid"):
            comp = paired.get(tname, {})
            sp = comp.get("spearman_rho_at_q35", {})
            md = sp.get("mean_diff")
            pv = sp.get("p_value")
            n = sp.get("n", 0)
            if md is not None and not (isinstance(md, float) and math.isnan(md)):
                pv_str = f"p={pv:.4e}" if pv is not None and not (isinstance(pv, float) and math.isnan(pv)) else "p=N/A"
                lines.append(
                    f"  {gene:>6s} {tname:25s} n={n:3d}  Δ={md:+.4f}  {pv_str}"
                )

    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Close the Loop on a Ptarmigan-1 Screen — Stages A and B (three-target panel)",
    )
    p.add_argument(
        "--stage", choices=("A", "B", "AB"), default="AB",
        help="Stage(s) to run (default: AB, both).  Stage A requires an API key "
        "and will contact ptarmigan.app.  Stage B runs from cached files.",
    )
    p.add_argument(
        "--target", default="CDK2",
        help="Target gene name.  One of CDK2 (default), BRAF, CHK1, SRC, or a custom name "
        "with --chembl-id.  The per-target output goes to "
        "results/ptarmigan_loop_<stem>.json.  Run --merge to combine.",
    )
    p.add_argument(
        "--chembl-id", default=None,
        help="ChEMBL target id (e.g. CHEMBL301 for CDK2).  Required for custom targets.",
    )
    p.add_argument(
        "--uniprot", default=None,
        help="UniProt accession (e.g. P24941 for CDK2).  Optional for custom targets.",
    )
    p.add_argument(
        "--n-residues", type=int, default=None,
        help="Protein length in residues (e.g. 298 for CDK2).  Optional for custom targets.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Stage A only: fetch balance and estimate, print cost, and EXIT 0 "
        "without POSTing the screen.  Ignores --bypass-gate.",
    )
    p.add_argument(
        "--bypass-gate", action="store_true",
        help="Skip the $15 hard credit gate (only use if user explicitly confirmed)",
    )
    p.add_argument(
        "--merge", action="store_true",
        help="Merge all per-target artifacts into one combined results/ptarmigan_loop.json. "
        "CDK2 stays in the root fields for compatibility; targets map carries full payloads.",
    )
    p.add_argument(
        "--targets", nargs="+", default=None,
        help="Target gene names for --merge or sequential run (e.g. --targets CDK2 BRAF CHK1)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Merge mode
    if args.merge:
        tnames = args.targets or ["CDK2", "BRAF", "CHK1"]
        targets = []
        for name in tnames:
            if name.upper() in _TARGET_MAP:
                targets.append(_TARGET_MAP[name.upper()])
            else:
                print(f"[MERGE] Unknown target: {name}", file=sys.stderr)
                return 1
        return merge_artifacts(targets=targets)

    # Normal mode: single target
    target = _resolve_target(args)

    print(f"Target: {target.gene} (ChEMBL {target.chembl_id}, UniProt {target.uniprot}, "
          f"{target.n_residues} aa)")

    if args.stage in ("A", "AB"):
        exit_code = stage_a(target=target, dry_run=args.dry_run, bypass_gate=args.bypass_gate)
        if exit_code != 0:
            return exit_code
        if args.dry_run:
            return 0

    if args.stage in ("B", "AB"):
        exit_code = stage_b(target=target)
        if exit_code != 0:
            return exit_code

    return 0


if __name__ == "__main__":
    sys.exit(main())