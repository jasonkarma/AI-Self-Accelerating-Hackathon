"""Five matched 35-query benchmark methods."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .acquisition import fixed_expert_prior
from .developability import RiskNormalizer
from .features import encode_sequences
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions

METHODS = ("m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")


class AdaptiveToolPolicy:
    mode = "adaptive_tool"
    def score(self, sequences: pd.Series, revealed: pd.DataFrame) -> np.ndarray:
        best = revealed.sort_values("y_bind", ascending=False).head(max(1, min(5, len(revealed))))
        priors: dict[int, str] = {}
        for p in range(13):
            priors[p] = best.sequence.str[p].mode().iloc[0]
        return np.asarray([np.clip(sum(s[p] == aa for p, aa in priors.items()) / 13 * .15, -.15, .15) for s in sequences])


class LiveAgentPolicy(AdaptiveToolPolicy):
    mode = "live_agent"


@dataclass
class BenchmarkResult:
    method: str
    seed: int
    initial_ids: list[str]
    curve: list[dict]
    aubqc: float


def run_method(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, method: str, seed: int,
               initial_ids: list[str], m5_policy: AdaptiveToolPolicy | LiveAgentPolicy | None = None) -> BenchmarkResult:
    rng = np.random.default_rng(seed + METHODS.index(method) * 10_000)
    work = frame.copy()
    scored = [normalizer.score(s) for s in work.sequence]
    work["y_dev"] = [s["y_dev"] for s in scored]
    X = encode_sequences(work.sequence)
    id_to_idx = {cid: i for i, cid in enumerate(work.candidate_id)}
    queried = list(initial_ids)
    curve = []
    policy = m5_policy or AdaptiveToolPolicy()
    for checkpoint in range(5, 36, 5):
        revealed = work[work.candidate_id.isin(queried)]
        qualified = revealed[revealed.y_dev >= tau_dev]
        curve.append({"query_count": checkpoint, "best_qualified_binding": float(qualified.y_bind.max()) if len(qualified) else float("nan")})
        if checkpoint == 35:
            break
        remaining = work[~work.candidate_id.isin(queried)].copy()
        if method == "m1_random":
            picks = rng.choice(remaining.candidate_id.to_numpy(), 5, replace=False).tolist()
        else:
            train_idx = np.asarray([id_to_idx[x] for x in queried]); pred_idx = np.asarray([id_to_idx[x] for x in remaining.candidate_id])
            bs = BindingSurrogate(seed, n_jobs=-1).fit(X[train_idx], work.iloc[train_idx].y_bind.to_numpy())
            ds = DevelopabilitySurrogate(seed + 1, n_jobs=-1).fit(X[train_idx], work.iloc[train_idx].y_dev.to_numpy())
            mb, sb = bs.predict(X[pred_idx]); md, sd = ds.predict(X[pred_idx])
            linked = link_predictions(mb, sb, md, sd, tau_dev, beta_0=0 if method == "m2_greedy" else 1)
            remaining["score"] = linked.a_base
            if method == "m4_fixed_prior":
                remaining["score"] += [fixed_expert_prior(s) for s in remaining.sequence]
            elif method == "m5_hybrid":
                remaining["score"] += policy.score(remaining.sequence, revealed)
                # Quota: two UCB, max joint sigma, hypothesis, falsification.
                remaining["sigma_joint"] = linked.sigma_joint
                two = remaining.nlargest(2, "score").candidate_id.tolist()
                rest = remaining[~remaining.candidate_id.isin(two)]
                third = rest.nlargest(1, "sigma_joint").candidate_id.tolist()
                rest = rest[~rest.candidate_id.isin(third)]
                fourth = rest.nlargest(1, "score").candidate_id.tolist()
                rest = rest[~rest.candidate_id.isin(fourth)]
                fifth = rest.nlargest(1, "sigma_joint").candidate_id.tolist()
                picks = two + third + fourth + fifth
            if method != "m5_hybrid":
                picks = remaining.sort_values(["score", "candidate_id"], ascending=[False, True]).head(5).candidate_id.tolist()
        queried.extend(picks)
    values = [c["best_qualified_binding"] for c in curve if np.isfinite(c["best_qualified_binding"])]
    return BenchmarkResult(method, seed, initial_ids, curve, float(sum(values) / 30.0))


def run_benchmark(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, *, seeds: int = 10,
                  m5_policy_mode: str = "adaptive_tool") -> dict:
    results = []
    policy = LiveAgentPolicy() if m5_policy_mode == "live_agent" else AdaptiveToolPolicy()
    for seed in range(seeds):
        initial = np.random.default_rng(seed).choice(frame.candidate_id.to_numpy(), 5, replace=False).tolist()
        results.extend(run_method(frame, normalizer, tau_dev, m, seed, initial, policy) for m in METHODS)
    curves = []
    for q in range(5, 36, 5):
        row = {"query_count": q}
        for method in METHODS:
            vals = [next(p["best_qualified_binding"] for p in r.curve if p["query_count"] == q) for r in results if r.method == method]
            row[method] = round(float(np.nanmedian(vals)), 3)
        curves.append(row)
    return {"m5_policy_mode": policy.mode, "seeds": seeds, "curves": curves,
            "aubqc": {m: float(np.mean([r.aubqc for r in results if r.method == m])) for m in METHODS},
            "matched_initial_seeds": {str(s): next(r.initial_ids for r in results if r.seed == s) for s in range(seeds)}}
