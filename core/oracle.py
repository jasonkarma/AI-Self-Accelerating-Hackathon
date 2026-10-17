"""Only authorized access path to hidden binding and objective labels."""
from __future__ import annotations

from collections.abc import Sequence
import pandas as pd

from .developability import RiskNormalizer
from .ledger import ImmutableLedger


class _Oracle:
    objective: str

    def __init__(self, private_dataset: pd.DataFrame, ledger: ImmutableLedger):
        self._data = private_dataset.set_index("candidate_id", drop=False)
        self.ledger = ledger

    def _value(self, row: pd.Series) -> float:
        raise NotImplementedError

    def query(self, candidate_ids: Sequence[str], *, round_number: int, actor: str = "campaign",
              batch_tag: str = "batch") -> list[dict]:
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Duplicate candidate IDs in one oracle query")
        reveals = []
        for candidate_id in candidate_ids:
            if candidate_id not in self._data.index:
                raise KeyError(candidate_id)
            cached = self.ledger.find_reveal(candidate_id, self.objective)
            if cached:
                reveals.append({**cached, "cached": True})
                continue
            value = self._value(self._data.loc[candidate_id])
            payload = {"objective": self.objective, "value": value, "sequence": self._data.loc[candidate_id, "sequence"]}
            self.ledger.append("oracle_reveal", payload, actor=actor, candidate_id=candidate_id,
                               round_number=round_number, batch_tag=batch_tag)
            reveals.append({"candidate_id": candidate_id, **payload, "cached": False})
        return reveals


class BindingOracle(_Oracle):
    objective = "y_bind"

    def _value(self, row: pd.Series) -> float:
        return float(row["y_bind"])


class DevelopabilityOracle(_Oracle):
    objective = "y_dev"

    def __init__(self, private_dataset: pd.DataFrame, ledger: ImmutableLedger, normalizer: RiskNormalizer):
        super().__init__(private_dataset, ledger)
        self.normalizer = normalizer

    def _value(self, row: pd.Series) -> float:
        return float(self.normalizer.score(row["sequence"])["y_dev"])
