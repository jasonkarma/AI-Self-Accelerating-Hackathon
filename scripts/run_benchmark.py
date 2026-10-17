#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import SETTINGS
from core.baselines import run_benchmark
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol
from data.datasets import load_primary, public_pool

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--seeds",type=int,default=10); parser.add_argument("--m5-policy-mode",choices=["adaptive_tool","live_agent"],default="adaptive_tool"); args=parser.parse_args()
    frame, synthetic=load_primary(); protocol=freeze_protocol(public_pool(frame),is_synthetic=synthetic); normalizer=RiskNormalizer.from_dict(protocol["normalizer"])
    result=run_benchmark(frame,normalizer,protocol["tau_dev"],seeds=args.seeds,m5_policy_mode=args.m5_policy_mode)
    path=SETTINGS.root/f"results/benchmark_{args.seeds}seeds.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(result,indent=2)+"\n")
    if args.seeds==10: SETTINGS.benchmark_path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["aubqc"],indent=2)); print(path)
