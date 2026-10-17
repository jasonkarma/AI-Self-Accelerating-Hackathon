#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import load_primary, public_pool
from core.preregistration import freeze_protocol

if __name__ == "__main__":
    frame, synthetic = load_primary()
    protocol = freeze_protocol(public_pool(frame), is_synthetic=synthetic)
    print(protocol)
