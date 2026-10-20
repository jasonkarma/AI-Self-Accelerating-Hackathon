"""Leakage-free positional sequence encoding, width-agnostic.

The encoding is ``length × (20 one-hot + 3 physchem)``, so a 13-residue HCDR3
gives the frozen 299 dimensions and a 9- or 11-residue CDR3 gives
``9 × 23 = 207`` / ``11 × 23 = 253``. The width is **inferred from the data**, not
from a constant, so every existing call site (`select_five_slots`, the benchmark,
the live campaign) keeps working unchanged and 1N8Z's feature matrix is
bit-identical: for a 13-mer the reduction below is the same concatenation it
always was.

Why the width has to be inferred rather than threaded: `encode_sequences` is
called with candidates from *one* pool in every path, and inferring keeps the
surrogate's interface (`X` of shape ``(n, dim)``) unparameterized. A caller that
mixes lengths is a bug, so it is rejected rather than silently padded.
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .constants import AA_TO_INDEX, CHARGE, HYDROPATHY, SEQUENCE_LENGTH, VOLUME

#: Amino-acid one-hot width and per-residue physicochemical block. Kept as named
#: constants because :func:`feature_dim` is the only place that multiplies them.
ONE_HOT_WIDTH = 20
PHYSCHEM_WIDTH = 3

#: The frozen 1N8Z width (``13 × 23``). Retained for callers and tests that
#: reference it by name; it is *not* the width of every layer any more.
FEATURE_DIM = SEQUENCE_LENGTH * (ONE_HOT_WIDTH + PHYSCHEM_WIDTH)


def feature_dim(length: int = SEQUENCE_LENGTH) -> int:
    """Feature width for a CDR3 of ``length`` residues."""
    return int(length) * (ONE_HOT_WIDTH + PHYSCHEM_WIDTH)


def validate_sequence(sequence: str, *, length: int = SEQUENCE_LENGTH) -> str:
    """Uppercase and validate one sequence at an explicit residue ``length``.

    ``length`` defaults to :data:`core.constants.SEQUENCE_LENGTH`, so every
    existing caller keeps the strict 13-residue dataset boundary it had. A
    cross-target path passes its own length instead of padding or truncating a
    sequence the assay never measured.
    """
    sequence = str(sequence).strip().upper()
    if len(sequence) != length:
        raise ValueError(f"Expected length {length}, got {len(sequence)}: {sequence!r}")
    invalid = sorted(set(sequence) - set(AA_TO_INDEX))
    if invalid:
        raise ValueError(f"Non-canonical amino acids: {invalid}")
    return sequence


def encode_sequence(sequence: str, *, length: int | None = None) -> np.ndarray:
    """One positional encoding; ``length`` defaults to the sequence's own length."""
    if length is None:
        length = len(str(sequence).strip())
    sequence = validate_sequence(sequence, length=length)
    one_hot = np.zeros((length, ONE_HOT_WIDTH), dtype=np.float32)
    physchem = np.zeros((length, PHYSCHEM_WIDTH), dtype=np.float32)
    for pos, aa in enumerate(sequence):
        one_hot[pos, AA_TO_INDEX[aa]] = 1.0
        physchem[pos] = (HYDROPATHY[aa], CHARGE[aa], VOLUME[aa])
    encoded = np.concatenate((one_hot.ravel(), physchem.ravel()))
    assert encoded.shape == (feature_dim(length),)
    return encoded


def encode_sequences(sequences: Iterable[str]) -> np.ndarray:
    """Encode a homogeneous pool; the width follows the pool's residue length.

    An empty input keeps the frozen 299-wide shape so a degenerate caller gets a
    correctly-typed empty matrix rather than a shape the surrogate cannot fit.
    """
    rows = list(sequences)
    if not rows:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    lengths = {len(str(sequence).strip()) for sequence in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"encode_sequences requires a single CDR3 length; got {sorted(lengths)}. "
            "Mixed-length pools cannot share one positional feature space."
        )
    length = lengths.pop()
    return np.vstack([encode_sequence(s, length=length) for s in rows])
