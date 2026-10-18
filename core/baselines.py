"""Five matched 35-query benchmark methods.

Every method ranks and selects from the same **qualified** pool: ``y_dev`` is
sequence-only information available at ``t=0`` (SPEC §2), so spending a slot on a
candidate that cannot pass ``tau_dev`` is strictly wasted — it can never move
``best_qualified_binding`` or AUBQC.

M5 does not re-implement the campaign's logic: it calls the same
:func:`core.acquisition.select_five_slots` portfolio the live run uses, with the
same surrogate residuals and the same falsified-position feedback, so a
benchmark number and a live number cannot diverge.

Four self-learning metrics accompany AUBQC, because "best single molecule found"
is a weak claim about a learning loop:

* ``normalized_aubqc`` — the same area, but ``y*(q)`` is rescaled between the
  seed score ``y*(5)`` (0.0) and the best qualified binder in the pool (1.0), so
  the number reads directly as a fraction of the available headroom.
* ``top5_panel_*`` — mean binding of the best five qualified candidates found so
  far, which measures *portfolio depth* instead of one lucky draw and does not
  punish exploration slots that surface non-binders.
* ``spearman_rho_*`` — rank correlation between each method's scoring model and
  the true binding of the qualified *unqueried* pool. This is the number that is
  comparable to AbBiBench zero-shot baselines (AF3 ``-0.16``,
  ProteinMPNN ``-0.17``, ESM-2 ``-0.13`` on 1N8Z), whereas AUBQC only measures a
  search process.
* ``top_decile_hit_rate`` — fraction of the 30 post-seed queries that land in the
  top decile of all qualified binders in the pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .acquisition import (
    SLOT_COUNT,
    adaptive_prior_score,
    fixed_expert_prior,
    positional_priors_from_revealed,
    qualified_pool,
    sar_effect_table,
    select_five_slots,
)
from .developability import RiskNormalizer
from .features import encode_sequences
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions
from .tool_arsenal import _tested_positions

METHODS = ("m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")


class AdaptiveToolPolicy:
    """Deterministic stand-in for the council when no LLM is in the loop.

    Emits ``Δ_agent`` through the same function the live path uses - summed
    positional priors derived from the best revealed sequences plus the
    empirical-Bayes SAR term - and honours the falsified-position feedback, so
    the benchmark's M5 self-corrects exactly like a live campaign would.
    """

    mode = "adaptive_tool"

    def __init__(self) -> None:
        self.last_priors: list = []
        self.last_sar: dict = {}
        self.falsified_positions: set[int] = set()

    def prepare(
        self,
        revealed: pd.DataFrame,
        tau_dev: float,
        *,
        residuals: dict[str, float] | None = None,
    ) -> None:
        """Precompute the priors and SAR table for one decision."""
        self.last_priors = positional_priors_from_revealed(
            revealed, falsified_positions=self.falsified_positions
        )
        self.last_sar = sar_effect_table(revealed, tau_dev=tau_dev, residuals=residuals)

    def score(self, sequences: pd.Series, revealed: pd.DataFrame) -> np.ndarray:
        """``Δ_agent`` for each sequence."""
        self.prepare(revealed, tau_dev=float("nan"))
        return np.asarray([
            adaptive_prior_score(
                s,
                priors=self.last_priors,
                sar_effects=self.last_sar,
                falsified_positions=self.falsified_positions,
            )
            for s in sequences
        ])


class LiveAgentPolicy(AdaptiveToolPolicy):
    mode = "live_agent"


@dataclass
class BenchmarkResult:
    method: str
    seed: int
    initial_ids: list[str]
    curve: list[dict]
    aubqc: float
    #: ids queried per round, in slot order (diagnostics only).
    picks: list[list[str]] = field(default_factory=list)
    #: ``y*(q)`` rescaled to ``[0, 1]`` between the seed score and the pool ceiling.
    normalized_curve: list[dict] = field(default_factory=list)
    #: Area under the normalized curve, i.e. ``AUBQC`` rescaled to ``[0, 1]``.
    normalized_aubqc: float = float("nan")
    #: Mean binding of the best five qualified candidates found by ``q``.
    panel_curve: list[dict] = field(default_factory=list)
    panel_aubqc: float = float("nan")
    #: ``{query_count, spearman_rho}``; ``rho`` is ``None`` for model-free M1.
    rho_curve: list[dict] = field(default_factory=list)
    #: Post-seed queries that landed in the pool's top decile, and queries spent.
    top_decile_hits: int = 0
    top_decile_queries: int = 0

    @property
    def top_decile_hit_rate(self) -> float:
        """Fraction of post-seed queries that discovered a top-decile binder."""
        if not self.top_decile_queries:
            return float("nan")
        return self.top_decile_hits / self.top_decile_queries


def _spearman_or_none(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    """Spearman ``rho`` between a scoring model and true binding, or ``None``.

    ``None`` rather than ``0.0`` when it is undefined: a constant score (M1 has no
    model at all, and an early pooled surrogate can be degenerate on a tiny
    training set) is *missing information*, not evidence of no correlation, and
    averaging it in as zero would understate every method equally.
    """
    x = np.asarray(prediction, dtype=float)
    y = np.asarray(truth, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if np.isfinite(rho) else None


def _pick_m5(
    remaining: pd.DataFrame,
    revealed: pd.DataFrame,
    tau_dev: float,
    *,
    residuals: dict[str, float],
    queries_spent: int,
    total_budget: int,
    falsified_positions: set[int] | frozenset[int],
    priors: list | None = None,
    batch_size: int = SLOT_COUNT,
) -> list[str]:
    """M5's five slots, delegated to the single shared portfolio.

    ``select_five_slots`` takes an optional ``RoundDecisionRecord`` and derives
    priors itself when there is none; the benchmark has no planner, so it passes
    its own ``AdaptiveToolPolicy`` priors through a minimal stand-in record so
    both paths converge on the identical ``Δ_agent``.
    """
    decision = None
    if priors:
        from .schemas import RoundDecisionRecord

        decision = RoundDecisionRecord(
            round_number=1,
            phase="Cold-Start Tuition (Exploration)",
            tools_invoked=[],
            hypothesis_id="HYP-BENCH-01",
            hypothesis_statement="Benchmark policy priors.",
            previous_hypothesis_verdict="N/A (Round 1)",
            supporting_candidate_ids=[],
            agent_disagreement_summary="Deterministic policy; no council.",
            confidence_rating="Low",
            positional_priors=priors,
            slot3_uncertainty_target_id=str(remaining.candidate_id.iloc[0]),
            slot4_hypothesis_target_id=str(remaining.candidate_id.iloc[1]),
            slot5_falsification_target_id=str(remaining.candidate_id.iloc[2]),
        )
    chosen = select_five_slots(
        remaining,
        decision,
        ledger=revealed,
        tau_dev=tau_dev,
        sar_effects=sar_effect_table(revealed, tau_dev=tau_dev, residuals=residuals),
        residuals=residuals,
        falsified_positions=falsified_positions,
        queries_spent=queries_spent,
        total_budget=total_budget,
        batch_size=batch_size,
    )
    return [str(slot["candidate_id"]) for slot in chosen]


def run_method(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, method: str, seed: int,
               initial_ids: list[str], m5_policy: AdaptiveToolPolicy | LiveAgentPolicy | None = None) -> BenchmarkResult:
    rng = np.random.default_rng(seed + METHODS.index(method) * 10_000)
    work = frame.copy()
    scored = [normalizer.score(s) for s in work.sequence]
    work["y_dev"] = [s["y_dev"] for s in scored]
    X = encode_sequences(work.sequence)
    id_to_idx = {cid: i for i, cid in enumerate(work.candidate_id)}
    queried = list(initial_ids)
    total_budget = len(initial_ids) + 6 * SLOT_COUNT
    curve, panel_curve, rho_curve, picks = [], [], [], []
    policy = m5_policy or AdaptiveToolPolicy()
    falsified_positions: set[int] = set()
    #: (slot 4 id, slot 5 id, the round's declared priors) for the batch that was
    #: just designed, so the paired H1/H0 audit can run once it is measured.
    pending_audit: tuple[str, str, list] | None = None

    # Evaluation-only reference points. Both are computed from labels the method
    # never sees during design: the pool ceiling is a reporting normaliser, and
    # the top-decile threshold is a hit definition. Neither steers a pick.
    pool_qualified = work[work.y_dev >= tau_dev]
    pool_ceiling = float(pool_qualified.y_bind.max()) if len(pool_qualified) else float("nan")
    decile_threshold = float(np.quantile(pool_qualified.y_bind, 0.90)) if len(pool_qualified) else float("inf")
    post_seed_hits = 0
    post_seed_queries = 0
    seed_binding: float | None = None

    for checkpoint in range(SLOT_COUNT, total_budget + 1, SLOT_COUNT):
        revealed = work[work.candidate_id.isin(queried)]
        qualified = revealed[revealed.y_dev >= tau_dev]
        best = float(qualified.y_bind.max()) if len(qualified) else float("nan")
        curve.append({"query_count": checkpoint, "best_qualified_binding": best})
        # Panel depth: the best five qualified molecules found so far, not one
        # lucky maximum. Exploration slots that surface non-binders simply do not
        # enter the panel, so this metric cannot punish exploration.
        panel_curve.append({
            "query_count": checkpoint,
            "top5_panel_binding": float(qualified.y_bind.nlargest(SLOT_COUNT).mean()) if len(qualified) else float("nan"),
        })
        if seed_binding is None:
            seed_binding = best

        # Paired H1/H0 audit of the previous batch (mirrors the live campaign's
        # ``audit_past_hypotheses``): if the null arm beat the hypothesis arm,
        # invert the positions that hypothesis was built on. This is the
        # benchmark's self-correction path and needs no extra queries.
        if pending_audit is not None:
            h1_id, h0_id, prior_list = pending_audit
            measured = dict(zip(revealed.candidate_id.astype(str), revealed.y_bind.astype(float)))
            if h1_id in measured and h0_id in measured and measured[h0_id] > measured[h1_id]:
                # Invert only the rules the Slot 4 candidate actually *exercised*,
                # via the same helper the live audit uses. Inverting the whole
                # declared list blacklists most of the CDR3 within three rounds,
                # and it would also make benchmark M5 diverge from live M5 — the
                # split this module exists to prevent.
                sequences = dict(zip(work.candidate_id.astype(str), work.sequence.astype(str)))
                falsified_positions |= _tested_positions(
                    {"positional_priors": [prior.model_dump() for prior in prior_list]},
                    sequences.get(h1_id),
                )
            pending_audit = None

        remaining = work[~work.candidate_id.isin(queried)].copy()

        if method == "m1_random":
            # No scoring model at all, so the rank correlation is undefined by
            # construction rather than zero: reporting 0.0 would be a claim.
            rho_curve.append({"query_count": checkpoint, "spearman_rho": None})
            if checkpoint >= total_budget:
                break
            round_picks = rng.choice(
                qualified_pool(remaining, tau_dev).candidate_id.to_numpy(), SLOT_COUNT, replace=False
            ).tolist()
        else:
            # Fit on the whole remaining pool, *then* gate: the gate must change
            # which candidates are eligible, never which ones the surrogate saw.
            train_idx = np.asarray([id_to_idx[x] for x in queried])
            pred_idx = np.asarray([id_to_idx[x] for x in remaining.candidate_id])
            bs = BindingSurrogate(seed, n_jobs=-1).fit(X[train_idx], work.iloc[train_idx].y_bind.to_numpy())
            ds = DevelopabilitySurrogate(seed + 1, n_jobs=-1).fit(X[train_idx], work.iloc[train_idx].y_dev.to_numpy())
            mb, sb = bs.predict(X[pred_idx]); md, sd = ds.predict(X[pred_idx])
            linked = link_predictions(mb, sb, md, sd, tau_dev, beta_0=0 if method == "m2_greedy" else 1)
            remaining["mu_bind"] = mb
            remaining["sigma_bind"] = sb
            remaining["sigma_joint"] = linked.sigma_joint
            remaining["a_base"] = linked.a_base
            pool = qualified_pool(remaining, tau_dev)

            if method == "m5_hybrid":
                # Residuals need a prediction for each *revealed* candidate. Those
                # rows are no longer in ``remaining``, so predict them explicitly
                # rather than looking them up in ``mu_map`` (which would be empty
                # for exactly the rows we need). The surrogate was fitted on a
                # subset of these rows, so the residual is in-sample for those and
                # honest for the rest — the SAR question ("which substitution
                # surprised the model?") is still the right target, and it is the
                # only residual available without refitting a held-out model each
                # round.
                train_seq_idx = np.asarray([id_to_idx[x] for x in revealed.candidate_id])
                mu_revealed, _ = bs.predict(X[train_seq_idx])
                residuals = {
                    str(cid): float(y) - float(mu)
                    for cid, y, mu in zip(revealed.candidate_id, revealed.y_bind, mu_revealed)
                }
                policy.prepare(revealed, tau_dev, residuals=residuals)
                pool = pool.assign(score=pool.a_base + np.asarray([
                    adaptive_prior_score(
                        s,
                        priors=policy.last_priors,
                        sar_effects=policy.last_sar,
                        falsified_positions=falsified_positions,
                    )
                    for s in pool.sequence
                ]))
            elif method == "m4_fixed_prior":
                pool = pool.copy()
                pool["score"] = pool.a_base + [fixed_expert_prior(s) for s in pool.sequence]
            else:
                pool = pool.assign(score=pool.a_base)

            # The method's own scoring model against the truth. This is the one
            # metric comparable to zero-shot affinity predictors (AF3 -0.16,
            # ProteinMPNN -0.17, ESM-2 -0.13 on 1N8Z).
            rho_curve.append({
                "query_count": checkpoint,
                "spearman_rho": _spearman_or_none(pool.score.to_numpy(), pool.y_bind.to_numpy()),
            })
            if checkpoint >= total_budget:
                break

            if method == "m5_hybrid":
                round_picks = _pick_m5(
                    pool, revealed, tau_dev,
                    residuals=residuals,
                    queries_spent=len(queried),
                    total_budget=total_budget,
                    falsified_positions=falsified_positions,
                    priors=policy.last_priors,
                    batch_size=SLOT_COUNT,
                )
                pending_audit = (
                    round_picks[3],
                    round_picks[4],
                    list(policy.last_priors),
                ) if len(round_picks) == SLOT_COUNT else None
            else:
                round_picks = pool.sort_values(
                    ["score", "candidate_id"], ascending=[False, True]
                ).candidate_id.head(SLOT_COUNT).tolist()

        picks.append(round_picks)
        measured_now = dict(zip(work.candidate_id.astype(str), work.y_bind.astype(float)))
        post_seed_hits += sum(measured_now.get(str(cid), -np.inf) >= decile_threshold for cid in round_picks)
        post_seed_queries += len(round_picks)
        queried.extend(round_picks)

    values = [c["best_qualified_binding"] for c in curve if np.isfinite(c["best_qualified_binding"])]
    # ``normalized`` rescales the discovery curve between the seed score (0.0) and
    # the best qualified binder available anywhere in the pool (1.0), so the area
    # under it reads directly as a fraction of the achievable headroom.
    span = pool_ceiling - seed_binding if seed_binding is not None and np.isfinite(pool_ceiling) else float("nan")
    normalized_curve = []
    for point in curve:
        value = point["best_qualified_binding"]
        normalized = (
            (value - seed_binding) / span
            if np.isfinite(value) and np.isfinite(span) and span > 0
            else float("nan")
        )
        # Clamp into [0, 1]: a method can never beat the pool ceiling, and a
        # dataset with a degenerate span would otherwise emit infinities.
        normalized_curve.append({
            "query_count": point["query_count"],
            "normalized": float(np.clip(normalized, 0.0, 1.0)) if np.isfinite(normalized) else float("nan"),
        })
    normalized_values = [p["normalized"] for p in normalized_curve if np.isfinite(p["normalized"])]
    panel_values = [p["top5_panel_binding"] for p in panel_curve if np.isfinite(p["top5_panel_binding"])]
    return BenchmarkResult(
        method=method,
        seed=seed,
        initial_ids=initial_ids,
        curve=curve,
        aubqc=float(sum(values) / 30.0),
        picks=picks,
        normalized_curve=normalized_curve,
        normalized_aubqc=float(sum(normalized_values) / 30.0),
        panel_curve=panel_curve,
        panel_aubqc=float(sum(panel_values) / 30.0),
        rho_curve=rho_curve,
        top_decile_hits=int(post_seed_hits),
        top_decile_queries=int(post_seed_queries),
    )


def _median_curve(results: list[BenchmarkResult], series: str, value_key: str, *, digits: int) -> list[dict]:
    """Median across seeds of one per-checkpoint series, one column per method."""
    curve = []
    for q in range(5, 36, 5):
        row: dict = {"query_count": q}
        for name in METHODS:
            values = [
                next(point[value_key] for point in getattr(result, series) if point["query_count"] == q)
                for result in results
                if result.method == name
            ]
            row[name] = round(float(np.nanmedian(values)), digits)
        curve.append(row)
    return curve


def _mean_rho(results: list[BenchmarkResult], method: str, q: int) -> float | None:
    """Seed-mean Spearman at one checkpoint, skipping undefined correlations.

    ``None`` (model-free M1, or an undefined constant score) is dropped rather
    than counted as ``0.0``: it is absent evidence, not evidence of absence.
    """
    values = []
    for result in results:
        if result.method != method:
            continue
        point = next((p for p in result.rho_curve if p["query_count"] == q), None)
        rho = None if point is None else point["spearman_rho"]
        if rho is not None and np.isfinite(rho):
            values.append(float(rho))
    return float(np.mean(values)) if values else None


def run_benchmark(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, *, seeds: int = 10,
                  m5_policy_mode: str = "adaptive_tool") -> dict:
    results = []
    policy = LiveAgentPolicy() if m5_policy_mode == "live_agent" else AdaptiveToolPolicy()
    for seed in range(seeds):
        initial = np.random.default_rng(seed).choice(frame.candidate_id.to_numpy(), 5, replace=False).tolist()
        results.extend(run_method(frame, normalizer, tau_dev, m, seed, initial, policy) for m in METHODS)
    curves = _median_curve(results, "curve", "best_qualified_binding", digits=3)
    top5_panel_curves = _median_curve(results, "panel_curve", "top5_panel_binding", digits=3)

    # Spearman is reported as a seed-mean per checkpoint (not a median of
    # medians) because the quantity of interest is the model's *expected* rank
    # correlation, and per-seed values are already scalar summaries.
    spearman_rho_curves = []
    for q in range(5, 36, 5):
        row: dict = {"query_count": q}
        for name in METHODS:
            rho = _mean_rho(results, name, q)
            row[name] = None if rho is None else round(rho, 4)
        spearman_rho_curves.append(row)

    hit_rates = {}
    for name in METHODS:
        rows = [r for r in results if r.method == name and r.top_decile_queries]
        hit_rates[name] = (
            float(sum(r.top_decile_hits for r in rows) / sum(r.top_decile_queries for r in rows))
            if rows else float("nan")
        )

    # ``method`` here must be the comprehension's own variable. Referencing the
    # leftover loop variable from the curve loop above silently reported M5's
    # AUBQC under every method name.
    return {
        "m5_policy_mode": policy.mode,
        "seeds": seeds,
        "curves": curves,
        "aubqc": {m: float(np.mean([r.aubqc for r in results if r.method == m])) for m in METHODS},
        # ``y*(q)`` rescaled 0-1 between the seed checkpoint and the pool ceiling.
        # The area keeps the preregistered 1/30 scaling, so its ceiling is
        # 6/30 = 0.2 for a method that jumps to the pool ceiling on round one;
        # reported as a clean fraction of headroom, not as 100.
        "normalized_curves": _median_curve(results, "normalized_curve", "normalized", digits=4),
        "normalized_aubqc": {
            m: float(np.mean([r.normalized_aubqc for r in results if r.method == m])) for m in METHODS
        },
        "top5_panel_curves": top5_panel_curves,
        "top5_panel_aubqc": {
            m: float(np.mean([r.panel_aubqc for r in results if r.method == m])) for m in METHODS
        },
        "spearman_rho_curves": spearman_rho_curves,
        "top_decile_hit_rate": hit_rates,
        "top_decile_definition": "y_bind >= 90th percentile of the developability-qualified pool",
        "matched_initial_seeds": {
            str(s): next(r.initial_ids for r in results if r.seed == s) for s in range(seeds)
        },
    }
