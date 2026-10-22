"""Frozen fast directional developability-risk proxy."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import pandas as pd

from .constants import CHARGE, HYDROPATHY, RISK_COMPONENTS, RISK_WEIGHTS, WT_SEQUENCE
from .features import validate_sequence


def raw_components(
    sequence: str,
    *,
    wt_sequence: str | None = None,
    length: int | None = None,
) -> dict[str, float]:
    """Six raw developability components for one sequence.

    ``wt_sequence`` is the reference the liability *excess* is measured against
    (``liability = max(0, this - reference)``). It defaults to the frozen 1N8Z
    wild type, so a 1N8Z call is unchanged; a cross-target call passes its own
    reference, because charging a 9-mer for liabilities its own germline already
    carries would measure the target, not the mutation.

    ``length`` is inferred from the sequence when omitted, which makes this
    width-agnostic without stopping the existing 13-residue dataset boundary
    from being enforced downstream by ``validate_dataset``.
    """
    reference = WT_SEQUENCE if wt_sequence is None else str(wt_sequence)
    if length is None:
        length = len(str(sequence).strip())
    s = validate_sequence(sequence, length=length)
    hydro = np.asarray([HYDROPATHY[a] for a in s])
    charges = np.asarray([CHARGE[a] for a in s])
    same_sign = sum(1 for a, b in zip(charges[:-1], charges[1:]) if a and b and np.sign(a) == np.sign(b))
    liability = s.count("C") + sum(s.count(m) for m in ("NG", "NS", "DG", "DS"))
    wt_liability = reference.count("C") + sum(
        reference.count(m) for m in ("NG", "NS", "DG", "DS")
    )
    # The ``window`` term is the worst 5-residue hydropathy window. Its scan range
    # must be the number of *valid* windows, ``len(s) - 4``, not the literal 9 that
    # was correct only at the frozen 13-residue width:
    #
    #   L=13 -> range(9), 9 windows of exactly width 5   (unchanged, bit-for-bit)
    #   L=9  -> range(5); the old range(9) reached i=8, where hydro[8:13] is a
    #           *1-element* slice, so a single hydrophobic residue won the max
    #           outright (measured 4.500 vs the correct 2.340)
    #   L=20 -> range(16); the old range(9) never looked at positions 13-19
    #
    # ``max(1, ...)`` guards a reference shorter than the window: there is no
    # valid 5-mer, so the whole-sequence mean is the honest answer rather than an
    # empty ``max()``. At L=13 this expression is exactly 9, which is why the
    # 1N8Z path is unchanged.
    window_span = max(1, len(s) - 4)
    return {
        "hydro": float(hydro.mean()),
        "window": float(max(hydro[i:i + 5].mean() for i in range(window_span))),
        "aromatic": float(sum(a in "FWY" for a in s) / len(s)),
        "charge": float(abs(charges.sum()) / len(s)),
        "cluster": float(same_sign),
        "liability": float(max(0, liability - wt_liability)),
    }


@dataclass(frozen=True)
class RiskNormalizer:
    minima: dict[str, float]
    maxima: dict[str, float]
    #: Reference CDR3 the liability *excess* is measured against. ``None`` means
    #: the frozen 1N8Z wild type, so every existing protocol — and every artifact
    #: serialized before this field existed — keeps its exact meaning.
    reference: str | None = None

    @classmethod
    def fit(
        cls, sequences: Iterable[str], *, wt_sequence: str | None = None
    ) -> "RiskNormalizer":
        frame = pd.DataFrame(
            raw_components(s, wt_sequence=wt_sequence) for s in sequences
        )
        if frame.empty:
            raise ValueError("Cannot fit risk normalizer to an empty pool")
        return cls(
            frame.min().to_dict(), frame.max().to_dict(), reference=wt_sequence
        )

    def transform_components(self, components: dict[str, float]) -> dict[str, float]:
        result = {}
        for name in RISK_COMPONENTS:
            span = self.maxima[name] - self.minima[name]
            result[name] = 0.0 if span == 0 else float(np.clip((components[name] - self.minima[name]) / span, 0, 1))
        return result

    def score(self, sequence: str) -> dict[str, object]:
        raw = raw_components(sequence, wt_sequence=self.reference)
        normalized = self.transform_components(raw)
        risk = float(np.dot([normalized[n] for n in RISK_COMPONENTS], RISK_WEIGHTS))
        return {"raw": raw, "normalized": normalized, "risk": risk, "y_dev": 1.0 - risk}

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"minima": self.minima, "maxima": self.maxima}

    @classmethod
    def from_dict(cls, value: dict[str, dict[str, float]]) -> "RiskNormalizer":
        # ``reference`` is deliberately *not* read from the serialized protocol:
        # the frozen 1N8Z protocol predates the field, and a cross-target run
        # re-fits its own normalizer rather than reinterpreting that artifact.
        return cls(value["minima"], value["maxima"])
