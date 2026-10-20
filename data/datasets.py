"""Strict dataset boundary and deterministic synthetic fallback."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import SETTINGS
from core.constants import AMINO_ACIDS, SEQUENCE_LENGTH, WT_SEQUENCE
from core.features import validate_sequence

REQUIRED_COLUMNS = ("candidate_id", "sequence", "y_bind")


def dataset_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.loc[:, REQUIRED_COLUMNS].sort_values("candidate_id").to_csv(index=False, float_format="%.8f")
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_dataset(frame: pd.DataFrame, *, expected_rows: int | None = None,
                     require_wt: bool = True, require_unique_sequences: bool = True,
                     length: int = SEQUENCE_LENGTH) -> pd.DataFrame:
    """Strict schema validation at an explicit CDR3 ``length``.

    ``length`` defaults to :data:`core.constants.SEQUENCE_LENGTH` (13), so the
    1N8Z boundary is unchanged: a 12- or 14-mer in the primary pool is still
    rejected. A cross-target pool passes its own width instead of being forced
    into 1N8Z's.
    """
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["sequence"] = result["sequence"].map(lambda s: validate_sequence(s, length=length))
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


#: Canonical IMGT J-segment start, and the pattern the 1N8Z ingest has always
#: used. Applied **first** because it is exact.
_HCDR3_STRICT = re.compile(r"YYC([A-Z]+?)WGQG")

#: Fallback for heavy chains whose J-segment does not read ``WGQG``: 3GBN reads
#: ``...MDVWGKGTTV``, so a strict-only match finds *zero* CDR3s in that entire
#: target and the dataset looks unusable when the regex is merely too narrow.
#: ``W G . G`` covers the J-segment's invariant Trp-Gly pair plus the two variable
#: positions around them (``WGQG``, ``WGKG``, ``WGRG``).
#:
#: This is a **fallback, not a replacement**, and that distinction is load-bearing.
#: The non-greedy group in the loose pattern will happily terminate at a spurious
#: early ``W G x G``: run loose-first over 1N8Z and 16 of its 419 rows come back as
#: 2-4 residue fragments instead of 13-mers. Strict-first keeps every one of 1N8Z's
#: 419 rows exactly as the frozen load did, and still recovers 3GBN's 1,887.
_HCDR3_LOOSE = re.compile(r"YYC([A-Z]+?)WG.G")

#: Public alias kept for callers that only need "the" motif.
_HCDR3_PATTERN = _HCDR3_STRICT


def extract_hcdr3(heavy_chain_sequence: str) -> str:
    """Extract the residues between the conserved YYC and the J-segment ``WGxG``.

    Used by the 1N8Z ingest and by every cross-target slice. The motif is the
    IMGT CDR3 definition: it opens at the conserved Cys that begins the loop
    (``...YYC``) and closes at the J-segment's invariant Trp-Gly pair.

    Strict ``WGQG`` is tried first and the looser ``WG.G`` only if it finds
    nothing, so the frozen 1N8Z parse is byte-identical (see
    :data:`_HCDR3_LOOSE`).
    """
    sequence = str(heavy_chain_sequence).strip().upper()
    for pattern in (_HCDR3_STRICT, _HCDR3_LOOSE):
        matches = pattern.findall(sequence)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Expected one YYC...WG.G HCDR3 context, found {len(matches)}")
    raise ValueError("Expected one YYC...WG.G HCDR3 context, found 0")


def safe_extract_hcdr3(heavy_chain_sequence: str) -> str | None:
    """``extract_hcdr3`` that returns ``None`` instead of raising.

    Real AbBiBench tables always contain rows whose heavy chain carries no single
    YYC...WGxG context (measured: 2FJG 133/2223, 1MLC 93/1229). Mapping the
    raising version over a column aborts the entire target on the first such
    row — which is how a perfectly usable 2,109-row dataset gets reported as
    "no CDR3 could be extracted". Dropping the unparseable rows is the correct
    behaviour: they are rows the extractor genuinely cannot interpret, and the
    count is recorded in the diagnostics.
    """
    try:
        return extract_hcdr3(heavy_chain_sequence)
    except (ValueError, TypeError):
        return None


#: Minimum number of distinct CDR3 sequences a cross-target slice must hold to be
#: a search problem at all. Five is the batch size: below it the selector cannot
#: even fill one round, so no method can do anything but query the whole pool.
MIN_CROSS_TARGET_SEQUENCES = 35


def cross_target_usability(sequences: Iterable[str]) -> dict:
    """Is this slice a *search problem*, and what does it look like?

    A dataset can be perfectly real and still be untestable: 4FQI's 65k rows
    collapse to **two** distinct CDR3 sequences (12 of its 14 positions are 100%
    conserved — it is a framework DMS library, not a CDR3 library), so M1-M5 all
    query the same two molecules and every metric is degenerate. Rejecting on
    row count alone would accept exactly that.

    The gate therefore measures *positional diversity*, which is what the
    surrogate and the SAR table actually learn from:

    * ``unique_sequences`` >= :data:`MIN_CROSS_TARGET_SEQUENCES`;
    * at least two positions that actually vary.

    Returns the diagnostics either way, so the reason a target was skipped is
    recorded rather than guessed at.
    """
    unique = sorted({str(s) for s in sequences})
    if not unique:
        return {
            "usable": False, "unique_sequences": 0, "length": 0,
            "varying_positions": [], "reason": "no sequences",
        }
    lengths = {len(s) for s in unique}
    length = max(lengths)
    varying = [
        position
        for position in range(length)
        if len({s[position] for s in unique if len(s) > position}) > 1
    ]
    if len(lengths) != 1:
        reason = f"ragged CDR3 lengths {sorted(lengths)}; a positional model needs one width"
        usable = False
    elif len(unique) < MIN_CROSS_TARGET_SEQUENCES:
        reason = (
            f"only {len(unique)} distinct CDR3 sequences (< {MIN_CROSS_TARGET_SEQUENCES}); "
            "no search problem to benchmark"
        )
        usable = False
    elif len(varying) < 2:
        reason = (
            f"only {len(varying)} varying position(s) of {length}; the library is "
            "framework-conserved and offers no positional signal"
        )
        usable = False
    else:
        reason = (
            f"{len(unique)} distinct CDR3s, {len(varying)}/{length} positions vary"
        )
        usable = True
    return {
        "usable": usable,
        "unique_sequences": len(unique),
        "length": length,
        "varying_positions": [p + 1 for p in varying],
        "reason": reason,
    }


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
                                 require_unique_sequences=False, length=int(length))
    validated["source_is_real"] = True
    return validated


def mode_reference(sequences: Iterable[str]) -> str:
    """The consensus CDR3 of a slice: per-position modal residue.

    Used as the cross-target *reference frame*. It is derived only from the
    sequences (available at ``t=0``, SPEC §2) and never from ``y_bind``, so using
    it cannot leak a label. It is also the only defensible reference for a
    dataset whose measured wild type was never assayed in that slice — the same
    situation 1N8Z's own ingest is in (README: the export contains no measured
    ``SRWGGDGFYAMDY`` row).
    """
    unique = [str(s) for s in sequences if str(s)]
    if not unique:
        raise ValueError("Cannot derive a consensus from an empty slice")
    width = len(unique[0])
    return "".join(
        pd.Series([s[position] for s in unique]).mode().iloc[0]
        for position in range(width)
    )


def collapse_replicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per distinct CDR3, at the mean of its measured replicates.

    AbBiBench reports every assay of every mutant, so the same CDR3 appears many
    times with different readings. Keeping them all would let a method "discover"
    the same molecule repeatedly (inflating the top-5 panel with one sequence) and
    would hand the surrogate contradictory labels for one input. Collapsing is a
    preprocessing decision, not a model change, and the number of merged rows is
    returned so the artifact can state it.
    """
    if frame.empty:
        return frame.copy(), 0
    grouped = (
        frame.groupby("sequence", as_index=False)
        .agg(y_bind=("y_bind", "mean"), n_replicates=("y_bind", "size"))
        .sort_values(["y_bind", "sequence"], ascending=[False, True])
        .reset_index(drop=True)
    )
    merged = int(len(frame) - len(grouped))
    return grouped, merged


def align_cross_target_frame(
    source: pd.DataFrame,
    *,
    prefix: str = "XTG",
    reference: str | None = None,
    length: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Turn a raw AbBiBench table into a cross-target pool in this repo's schema.

    Returns ``(frame, diagnostics)``. ``frame`` carries the same three columns the
    oracle path requires (``candidate_id``, ``sequence``, ``y_bind``) plus
    ``source_is_real``; ``diagnostics`` records how it was built so the artifact
    can state, rather than imply, what ran.

    Three decisions, all of which change the numbers and therefore must be
    written down:

    1. **No padding, no truncation.** ``length`` selects the CDR3 width to keep
       (default: the modal width of the slice). A slice whose CDR3s are 14-mers
       is either kept whole at 14 or rejected — never chopped to 13, because a
       chopped position was never assayed and the SAR table would attribute an
       effect to a residue the experiment did not measure.
    2. **Replicates collapse** to one row per sequence at the mean reading.
    3. **The reference is the consensus**, not 1N8Z's wild type. Scoring a 9-mer
       against ``SRWGGDGFYAMDY`` compares it to a sequence it does not share a
       coordinate system with; see ``core.acquisition.resolve_wt_sequence``.
    """
    required = {"heavy_chain_seq", "binding_score"}
    if not required <= set(source.columns):
        raise ValueError(f"AbBiBench table missing columns: {sorted(required - set(source.columns))}")
    extracted = source["heavy_chain_seq"].map(safe_extract_hcdr3)
    widths = extracted.str.len().dropna()
    rows_unparseable = int(extracted.isna().sum())
    if widths.empty:
        raise ValueError("No YYC...WG.G CDR3 could be extracted from this table")
    if length is None:
        length = int(widths.mode().iloc[0])
    keep = extracted.str.len().eq(length)
    kept = pd.DataFrame({
        "sequence": extracted[keep].to_numpy(),
        "y_bind": pd.to_numeric(source.loc[keep, "binding_score"], errors="raise").to_numpy(),
    })
    collapsed, merged = collapse_replicates(kept)
    reference_sequence = reference if reference is not None else mode_reference(collapsed.sequence)
    if len(reference_sequence) != length:
        raise ValueError(
            f"reference length {len(reference_sequence)} != slice length {length}"
        )
    collapsed["candidate_id"] = [f"{prefix}-{index:03d}" for index in range(len(collapsed))]
    collapsed["source_is_real"] = True
    validated = validate_dataset(
        collapsed,
        require_wt=False,
        require_unique_sequences=True,
        length=int(length),
    )
    validated["source_is_real"] = True
    if "n_replicates" in collapsed.columns:
        validated["n_replicates"] = collapsed["n_replicates"].to_numpy()
    diagnostics = {
        "rows_in_source": int(len(source)),
        "rows_unparseable": rows_unparseable,
        "rows_extracted": int(widths.size),
        "cdr3_length": int(length),
        "length_histogram": {int(k): int(v) for k, v in widths.value_counts().sort_index().items()},
        "rows_after_length_filter": int(keep.sum()),
        "replicates_collapsed": merged,
        "rows_final": int(len(validated)),
        "reference_sequence": str(reference_sequence),
        "reference_source": "consensus" if reference is None else "provided",
        "usability": cross_target_usability(validated.sequence),
    }
    return validated, diagnostics


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
