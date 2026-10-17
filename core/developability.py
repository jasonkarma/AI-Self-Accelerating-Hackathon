"""Frozen fast directional developability-risk proxy."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import pandas as pd

from .constants import CHARGE, HYDROPATHY, RISK_COMPONENTS, RISK_WEIGHTS, WT_SEQUENCE
from .features import validate_sequence


def raw_components(sequence: str) -> dict[str, float]:
    s = validate_sequence(sequence)
    hydro = np.asarray([HYDROPATHY[a] for a in s])
    charges = np.asarray([CHARGE[a] for a in s])
    same_sign = sum(1 for a, b in zip(charges[:-1], charges[1:]) if a and b and np.sign(a) == np.sign(b))
    liability = s.count("C") + sum(s.count(m) for m in ("NG", "NS", "DG", "DS"))
    wt_liability = WT_SEQUENCE.count("C") + sum(WT_SEQUENCE.count(m) for m in ("NG", "NS", "DG", "DS"))
    return {
        "hydro": float(hydro.mean()),
        "window": float(max(hydro[i:i + 5].mean() for i in range(9))),
        "aromatic": float(sum(a in "FWY" for a in s) / len(s)),
        "charge": float(abs(charges.sum()) / len(s)),
        "cluster": float(same_sign),
        "liability": float(max(0, liability - wt_liability)),
    }


@dataclass(frozen=True)
class RiskNormalizer:
    minima: dict[str, float]
    maxima: dict[str, float]

    @classmethod
    def fit(cls, sequences: Iterable[str]) -> "RiskNormalizer":
        frame = pd.DataFrame(raw_components(s) for s in sequences)
        if frame.empty:
            raise ValueError("Cannot fit risk normalizer to an empty pool")
        return cls(frame.min().to_dict(), frame.max().to_dict())

    def transform_components(self, components: dict[str, float]) -> dict[str, float]:
        result = {}
        for name in RISK_COMPONENTS:
            span = self.maxima[name] - self.minima[name]
            result[name] = 0.0 if span == 0 else float(np.clip((components[name] - self.minima[name]) / span, 0, 1))
        return result

    def score(self, sequence: str) -> dict[str, object]:
        raw = raw_components(sequence)
        normalized = self.transform_components(raw)
        risk = float(np.dot([normalized[n] for n in RISK_COMPONENTS], RISK_WEIGHTS))
        return {"raw": raw, "normalized": normalized, "risk": risk, "y_dev": 1.0 - risk}

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"minima": self.minima, "maxima": self.maxima}

    @classmethod
    def from_dict(cls, value: dict[str, dict[str, float]]) -> "RiskNormalizer":
        return cls(value["minima"], value["maxima"])
