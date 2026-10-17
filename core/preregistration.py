"""Label-blind frozen threshold preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import SETTINGS
from .constants import WT_SEQUENCE
from .developability import RiskNormalizer


def sequence_hash(pool: pd.DataFrame) -> str:
    payload = pool[["candidate_id", "sequence"]].sort_values("candidate_id").to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_protocol(pool: pd.DataFrame, *, output: Path = SETTINGS.protocol_path, is_synthetic: bool) -> dict:
    # Deliberately rejects rather than ignores labels: proves this step is label blind.
    if "y_bind" in pool.columns:
        raise ValueError("freeze_protocol accepts only candidate_id and sequence; binding labels are forbidden")
    normalizer = RiskNormalizer.fit(pool["sequence"])
    risks = np.asarray([normalizer.score(s)["risk"] for s in pool["sequence"]])
    wt_risk = normalizer.score(WT_SEQUENCE)["risk"]
    pass_fraction = float(np.mean(risks <= wt_risk))
    if 0.25 <= pass_fraction <= 0.65:
        tau_risk, source = wt_risk, "wild_type"
        threshold_percentile = pass_fraction
    else:
        tau_risk, source = float(np.median(risks)), "pool_median"
        threshold_percentile = 0.50
    qualified_mask = risks <= tau_risk
    qualified_count = int(np.sum(qualified_mask))
    qualification_pass_rate = float(np.mean(qualified_mask))
    protocol = {
        "version": "1.2-post-kickoff",
        "dataset_sequence_sha256": sequence_hash(pool),
        "pool_size": int(len(pool)),
        "wild_type": WT_SEQUENCE,
        "wt_risk": wt_risk,
        "wt_pass_fraction": pass_fraction,
        "qualification_pass_rate": qualification_pass_rate,
        "qualified_count": qualified_count,
        "threshold_percentile": threshold_percentile,
        "tau_risk": tau_risk,
        "tau_dev": 1.0 - tau_risk,
        "threshold_source": source,
        "is_synthetic": bool(is_synthetic),
        "normalizer": normalizer.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if output.exists():
        current = json.loads(output.read_text())
        if current == protocol:
            return current
    output.write_text(serialized)
    return protocol


def load_protocol(path: Path = SETTINGS.protocol_path) -> dict:
    if not path.exists():
        raise FileNotFoundError("Frozen protocol is missing; run scripts/freeze_benchmark.py")
    return json.loads(path.read_text())
