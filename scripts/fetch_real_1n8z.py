#!/usr/bin/env python3
"""Fetch AbBiBench and write validated 1N8Z + separate prompt-development slices."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from datasets import load_dataset
from config.settings import SETTINGS
from core.constants import WT_SEQUENCE
from data.datasets import validate_dataset, public_pool
from core.preregistration import freeze_protocol

def find_column(columns, terms):
    for col in columns:
        if any(t in col.lower() for t in terms): return col
    raise KeyError(f"Could not find any of {terms} in {columns}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default="AbBibench/Antibody_Binding_Benchmark_Dataset"); args=parser.parse_args()
    dataset=load_dataset(args.dataset)
    frame=pd.concat([split.to_pandas() for split in dataset.values()],ignore_index=True)
    structure=find_column(frame.columns,["pdb","structure"]); sequence=find_column(frame.columns,["hcdr3","h3_sequence","cdrh3"]); affinity=find_column(frame.columns,["-log","affinity","binding"])
    primary=frame[frame[structure].astype(str).str.upper().eq("1N8Z")].copy(); primary=primary[primary[sequence].astype(str).str.len()==13]
    primary=pd.DataFrame({"candidate_id":["WT" if s==WT_SEQUENCE else f"CAND-{i:03d}" for i,s in enumerate(primary[sequence])],"sequence":primary[sequence].str.upper(),"y_bind":primary[affinity],"source_is_real":True})
    primary=validate_dataset(primary,expected_rows=419); primary["source_is_real"]=True; SETTINGS.primary_csv.parent.mkdir(parents=True,exist_ok=True); primary.to_csv(SETTINGS.primary_csv,index=False)
    dev=frame[~frame[structure].astype(str).str.upper().eq("1N8Z")].head(300); pd.DataFrame({"candidate_id":[f"DEV-{i:03d}" for i in range(len(dev))],"sequence":dev[sequence],"y_bind":dev[affinity],"source_is_real":True}).to_csv(SETTINGS.development_csv,index=False)
    freeze_protocol(public_pool(primary),is_synthetic=False); print(f"Wrote {len(primary)} real 1N8Z rows")
