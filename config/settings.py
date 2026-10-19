"""Frozen campaign settings. Environment variables only select runtime behavior."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _representative_seed() -> int:
    """The benchmark seed whose Round-0 batch is a *median* cold start.

    Round 0 is five uniformly random draws from the pool, so its best qualified
    binder — ``y*(5)`` — is pure seed luck and ranges from ``6.2`` to ``8.97``
    across the 100 benchmark seeds. Reporting a single live campaign against the
    *seed median* is therefore not a like-for-like comparison: a seed that starts
    at ``6.2`` has far more headroom than one that starts at ``8.6``, and AUBQC
    rewards exactly that headroom.

    This constant is chosen by a criterion that never looks at the outcome: the
    seed whose Round-0 score is closest to the 100-seed median ``y*(5) = 7.705``.
    That is seed 21, at ``7.706`` — a difference of ``0.001`` binding units, i.e.
    the most representative cold start available. Picking the seed *after* seeing
    which one flatters M5 would be the single most common way to fake a benchmark
    result, so it is deliberately not done: the same-seed comparison this seed
    enables is what makes the run interpretable, and it is reported with the
    caveat that benchmarking is a *distribution*, not one draw.
    """
    return int(os.getenv("RANDOM_SEED", "21"))


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    primary_csv: Path = ROOT / "data/raw/1n8z_hcdr3.csv"
    development_csv: Path = ROOT / "data/raw/dev_hcdr3.csv"
    protocol_path: Path = ROOT / "config/frozen_protocol.json"
    ledger_path: Path = ROOT / "results/campaign.sqlite3"
    #: Frozen copy of the verified Seed-21 run (6 rounds, ``8.580`` best /
    #: ``1.963`` AUBQC). Stage recovery reads *from* this file and never writes to
    #: it, so a bad live round can never cost the demo its campaign.
    golden_ledger_path: Path = ROOT / "results/golden_seed21.sqlite3"
    benchmark_path: Path = ROOT / "results/benchmark_10seeds.json"
    pool_size: int = 419
    initial_queries: int = 5
    rounds: int = 6
    batch_size: int = 5
    assay_cost_usd: int = 1500
    m5_policy_mode: str = os.getenv("M5_POLICY_MODE", "adaptive_tool")
    hitl: bool = os.getenv("HITL", "0").lower() in {"1", "true", "yes"}
    #: Seed for the Round-0 batch, and therefore the benchmark seed this live
    #: campaign is the same-seed twin of. See :func:`_representative_seed`.
    random_seed: int = _representative_seed()

    @property
    def total_budget(self) -> int:
        return self.initial_queries + self.rounds * self.batch_size


SETTINGS = Settings()
