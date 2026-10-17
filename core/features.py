"""Leakage-free 299-dimensional positional sequence encoding."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .constants import AA_TO_INDEX, CHARGE, HYDROPATHY, SEQUENCE_LENGTH, VOLUME

FEATURE_DIM = 299


def validate_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper()
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected length {SEQUENCE_LENGTH}, got {len(sequence)}: {sequence!r}")
    invalid = sorted(set(sequence) - set(AA_TO_INDEX))
    if invalid:
        raise ValueError(f"Non-canonical amino acids: {invalid}")
    return sequence


def encode_sequence(sequence: str) -> np.ndarray:
    sequence = validate_sequence(sequence)
    one_hot = np.zeros((SEQUENCE_LENGTH, 20), dtype=np.float32)
    physchem = np.zeros((SEQUENCE_LENGTH, 3), dtype=np.float32)
    for pos, aa in enumerate(sequence):
        one_hot[pos, AA_TO_INDEX[aa]] = 1.0
        physchem[pos] = (HYDROPATHY[aa], CHARGE[aa], VOLUME[aa])
    encoded = np.concatenate((one_hot.ravel(), physchem.ravel()))
    assert encoded.shape == (FEATURE_DIM,)
    return encoded


def encode_sequences(sequences: Iterable[str]) -> np.ndarray:
    rows = [encode_sequence(s) for s in sequences]
    return np.vstack(rows) if rows else np.empty((0, FEATURE_DIM), dtype=np.float32)
