"""Strict dataset boundary and deterministic synthetic fallback."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import SETTINGS
from core.constants import AMINO_ACIDS, WT_SEQUENCE
from core.features import validate_sequence

REQUIRED_COLUMNS = ("candidate_id", "sequence", "y_bind")


def dataset_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.loc[:, REQUIRED_COLUMNS].sort_values("candidate_id").to_csv(index=False, float_format="%.8f")
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_dataset(frame: pd.DataFrame, *, expected_rows: int | None = None,
                     require_wt: bool = True, require_unique_sequences: bool = True) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["sequence"] = result["sequence"].map(validate_sequence)
    result["y_bind"] = pd.to_numeric(result["y_bind"], errors="raise")
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(result)}")
    if result["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    if require_unique_sequences and result["sequence"].duplicated().any():
        raise ValueError("Sequences must be unique")
    if require_wt and WT_SEQUENCE not in set(result["sequence"]):
        raise ValueError(f"Wild type {WT_SEQUENCE} is absent")
    return result.reset_index(drop=True)


def extract_hcdr3(heavy_chain_sequence: str) -> str:
    """Extract the residues between the conserved YYC and WGQG framework motifs."""
    sequence = str(heavy_chain_sequence).strip().upper()
    matches = re.findall(r"YYC([A-Z]+?)WGQG", sequence)
    if len(matches) != 1:
        raise ValueError(f"Expected one YYC...WGQG HCDR3 context, found {len(matches)}")
    return matches[0]


def prepare_abbibench_frame(source: pd.DataFrame, *, prefix: str, expected_rows: int | None = None,
                            length: int = 13) -> pd.DataFrame:
    """Convert an official AbBiBench table to the internal oracle schema."""
    required = {"heavy_chain_seq", "binding_score"}
    if not required <= set(source.columns):
        raise ValueError(f"AbBiBench table missing columns: {sorted(required - set(source.columns))}")
    extracted = source["heavy_chain_seq"].map(extract_hcdr3)
    keep = extracted.str.len().eq(length)
    result = pd.DataFrame({
        "candidate_id": [f"{prefix}-{i:03d}" for i in range(int(keep.sum()))],
        "sequence": extracted[keep].to_numpy(),
        "y_bind": pd.to_numeric(source.loc[keep, "binding_score"], errors="raise").to_numpy(),
    })
    result["source_is_real"] = True
    validated = validate_dataset(result, expected_rows=expected_rows, require_wt=False,
                                 require_unique_sequences=False)
    validated["source_is_real"] = True
    return validated


def generate_fixture(n: int = 419, seed: int = 180, *, prefix: str = "CAND") -> pd.DataFrame:
    """Synthetic benchmark fixture. Never represented as experimental evidence."""
    rng = np.random.default_rng(seed)
    sequences = [WT_SEQUENCE]
    while len(sequences) < n:
        chars = list(WT_SEQUENCE)
        count = int(rng.choice([1, 2, 3, 4], p=[.36, .34, .22, .08]))
        for pos in rng.choice(13, count, replace=False):
            choices = [a for a in AMINO_ACIDS if a != chars[pos]]
            chars[pos] = str(rng.choice(choices))
        sequence = "".join(chars)
        if sequence not in sequences:
            sequences.append(sequence)
    # Smooth hidden landscape plus deterministic epistasis/noise; known only to fixture generator.
    anchors = {2: "W", 7: "F", 8: "Y", 10: "M", 12: "Y"}
    values = []
    for sequence in sequences:
        mutations = sum(a != b for a, b in zip(sequence, WT_SEQUENCE))
        anchor_loss = sum(sequence[p] != aa for p, aa in anchors.items())
        polar_bonus = sum(sequence[p] in "STNQ" for p in (0, 5, 7))
        epistasis = 0.35 if sequence[5] in "ENQ" and sequence[7] in "ST" else 0.0
        stable_noise = np.random.default_rng(int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16)).normal(0, .16)
        values.append(8.85 - .18 * mutations - .55 * anchor_loss + .13 * polar_bonus + epistasis + stable_noise)
    ids = ["WT"] + [f"{prefix}-{i:03d}" for i in range(1, n)]
    return pd.DataFrame({"candidate_id": ids, "sequence": sequences, "y_bind": np.round(values, 5)})


def write_default_fixtures(primary_path: Path = SETTINGS.primary_csv, development_path: Path = SETTINGS.development_csv) -> None:
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    generate_fixture().to_csv(primary_path, index=False)
    generate_fixture(n=180, seed=991, prefix="DEV").to_csv(development_path, index=False)


def load_primary(path: Path = SETTINGS.primary_csv, *, generate_if_missing: bool = True) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(path)
        write_default_fixtures(path, SETTINGS.development_csv)
        synthetic = True
    else:
        synthetic = "source_is_real" not in pd.read_csv(path, nrows=1).columns
    raw = pd.read_csv(path)
    return validate_dataset(raw, expected_rows=SETTINGS.pool_size, require_wt=synthetic,
                            require_unique_sequences=synthetic), synthetic


def public_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a view that structurally cannot expose hidden binding labels."""
    return frame.loc[:, ["candidate_id", "sequence"]].copy()
