#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import write_default_fixtures, load_primary, public_pool
from core.preregistration import freeze_protocol

if __name__ == "__main__":
    write_default_fixtures()
    frame, _ = load_primary()
    protocol = freeze_protocol(public_pool(frame), is_synthetic=True)
    print(f"Generated {len(frame)} synthetic rows; tau_dev={protocol['tau_dev']:.6f}")
