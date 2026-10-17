"""Frozen campaign settings. Environment variables only select runtime behavior."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    primary_csv: Path = ROOT / "data/raw/1n8z_hcdr3.csv"
    development_csv: Path = ROOT / "data/raw/dev_hcdr3.csv"
    protocol_path: Path = ROOT / "config/frozen_protocol.json"
    ledger_path: Path = ROOT / "results/campaign.sqlite3"
    benchmark_path: Path = ROOT / "results/benchmark_10seeds.json"
    pool_size: int = 419
    initial_queries: int = 5
    rounds: int = 6
    batch_size: int = 5
    assay_cost_usd: int = 1500
    m5_policy_mode: str = os.getenv("M5_POLICY_MODE", "adaptive_tool")
    hitl: bool = os.getenv("HITL", "0").lower() in {"1", "true", "yes"}

    @property
    def total_budget(self) -> int:
        return self.initial_queries + self.rounds * self.batch_size


SETTINGS = Settings()
