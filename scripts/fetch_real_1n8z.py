#!/usr/bin/env python3
"""Atomically fetch official AbBiBench CSVs without datasets/dill serialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from config.settings import SETTINGS
from core.preregistration import freeze_protocol
from data.datasets import prepare_abbibench_frame, public_pool

LOGGER = logging.getLogger("fetch_real_1n8z")
PRIMARY_FILE = "binding_affinity/1n8z_benchmarking_data.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_csv(repo: str, filename: str, token: str | None) -> pd.DataFrame:
    LOGGER.info("Downloading dataset file: %s", filename)
    local = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, token=token))
    frame = pd.read_csv(local)
    LOGGER.info("Loaded %s: rows=%d columns=%s cache_sha256=%s", filename, len(frame),
                frame.columns.tolist(), file_sha256(local))
    return frame


def fetch_and_prepare(repo: str, token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = HfApi(token=token).list_repo_files(repo_id=repo, repo_type="dataset")
    affinity_files = sorted(path for path in files if path.startswith("binding_affinity/") and path.endswith(".csv"))
    LOGGER.info("Repository %s exposes %d binding CSV files", repo, len(affinity_files))
    if PRIMARY_FILE not in affinity_files:
        raise RuntimeError(f"Official 1N8Z file missing: {PRIMARY_FILE}")

    primary_source = download_csv(repo, PRIMARY_FILE, token)
    primary = prepare_abbibench_frame(primary_source, prefix="CAND", expected_rows=419)
    duplicates = int(primary["sequence"].duplicated(keep=False).sum())
    LOGGER.info("Validated 1N8Z: rows=%d unique_hcdr3=%d duplicate_rows=%d length_counts=%s",
                len(primary), primary.sequence.nunique(), duplicates,
                primary.sequence.str.len().value_counts().to_dict())
    LOGGER.info("Declared WT is a reference (measured WT rows=%d); no binding label will be fabricated",
                int((primary.sequence == "SRWGGDGFYAMDY").sum()))

    development_parts = []
    for filename in affinity_files:
        if filename == PRIMARY_FILE:
            continue
        source = download_csv(repo, filename, token)
        try:
            part = prepare_abbibench_frame(source, prefix="TMP", length=13)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", filename, exc)
            continue
        if not part.empty:
            part["source_file"] = filename
            development_parts.append(part)
    if not development_parts:
        raise RuntimeError("No non-1N8Z length-13 development rows were found")
    development = pd.concat(development_parts, ignore_index=True).head(300).copy()
    development["candidate_id"] = [f"DEV-{i:03d}" for i in range(len(development))]
    LOGGER.info("Prepared separate development slice: rows=%d source_files=%d", len(development),
                development.source_file.nunique())
    return primary, development


def atomic_install(primary: pd.DataFrame, development: pd.DataFrame) -> dict:
    SETTINGS.primary_csv.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.protocol_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SETTINGS.root, prefix=".real-data-stage-") as directory:
        stage = Path(directory)
        primary_tmp, development_tmp = stage / "1n8z_hcdr3.csv", stage / "dev_hcdr3.csv"
        protocol_tmp = stage / "frozen_protocol.json"
        primary.to_csv(primary_tmp, index=False)
        development.to_csv(development_tmp, index=False)
        protocol = freeze_protocol(public_pool(primary), output=protocol_tmp, is_synthetic=False)
        LOGGER.info("Staged files: primary_sha256=%s development_sha256=%s sequence_sha256=%s tau_dev=%.8f qualification_pass_rate=%.8f qualified_count=%d threshold_percentile=%.2f",
                    file_sha256(primary_tmp), file_sha256(development_tmp),
                    protocol["dataset_sequence_sha256"], protocol["tau_dev"],
                    protocol["qualification_pass_rate"], protocol["qualified_count"],
                    protocol["threshold_percentile"])
        # All validation and threshold computation succeeded; only now replace live files.
        old_protocol = None
        if SETTINGS.protocol_path.exists():
            try:
                old_protocol = json.loads(SETTINGS.protocol_path.read_text())
            except (json.JSONDecodeError, OSError):
                LOGGER.warning("Existing protocol is unreadable; downstream artifacts will be archived")
        os.replace(primary_tmp, SETTINGS.primary_csv)
        os.replace(development_tmp, SETTINGS.development_csv)
        os.replace(protocol_tmp, SETTINGS.protocol_path)
        if old_protocol and old_protocol.get("dataset_sequence_sha256") != protocol["dataset_sequence_sha256"]:
            old_hash = str(old_protocol.get("dataset_sequence_sha256", "unknown"))[:12]
            for artifact in (SETTINGS.ledger_path, SETTINGS.benchmark_path):
                if artifact.exists():
                    archived = artifact.with_name(f"{artifact.name}.pre-real-{old_hash}")
                    os.replace(artifact, archived)
                    LOGGER.info("Archived incompatible prior artifact: %s -> %s", artifact, archived)
    LOGGER.info("Atomic install complete: %s, %s, %s", SETTINGS.primary_csv,
                SETTINGS.development_csv, SETTINGS.protocol_path)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="AbBibench/Antibody_Binding_Benchmark_Dataset")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    primary, development = fetch_and_prepare(args.dataset, args.token)
    protocol = atomic_install(primary, development)
    print(json.dumps({"rows": len(primary), "development_rows": len(development),
                      "is_synthetic": protocol["is_synthetic"], "tau_dev": protocol["tau_dev"],
                      "qualification_pass_rate": protocol["qualification_pass_rate"],
                      "qualified_count": protocol["qualified_count"],
                      "threshold_percentile": protocol["threshold_percentile"],
                      "dataset_sequence_sha256": protocol["dataset_sequence_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
