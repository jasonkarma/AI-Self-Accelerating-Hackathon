"""Five matched 35-query benchmark methods.

Every method ranks and selects from the same **qualified** pool: ``y_dev`` is
sequence-only information available at ``t=0`` (SPEC §2), so spending a slot on a
candidate that cannot pass ``tau_dev`` is strictly wasted — it can never move
``best_qualified_binding`` or AUBQC.

M5 does not re-implement the campaign's logic: it calls the same
:func:`core.acquisition.select_five_slots` portfolio the live run uses, with the
same surrogate residuals and the same falsified-position feedback, so a
benchmark number and a live number cannot diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

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
    curve, picks = [], []
    policy = m5_policy or AdaptiveToolPolicy()
    falsified_positions: set[int] = set()
    #: (slot 4 id, slot 5 id, positions of the round's priors) for the batch that
    #: was just designed, so the paired H1/H0 audit can run once it is measured.
    pending_audit: tuple[str, str, set[int]] | None = None

    for checkpoint in range(SLOT_COUNT, total_budget + 1, SLOT_COUNT):
        revealed = work[work.candidate_id.isin(queried)]
        qualified = revealed[revealed.y_dev >= tau_dev]
        curve.append({"query_count": checkpoint, "best_qualified_binding": float(qualified.y_bind.max()) if len(qualified) else float("nan")})

        # Paired H1/H0 audit of the previous batch (mirrors the live campaign's
        # ``audit_past_hypotheses``): if the null arm beat the hypothesis arm,
        # invert the positions that hypothesis was built on. This is the
        # benchmark's self-correction path and needs no extra queries.
        if pending_audit is not None:
            h1_id, h0_id, positions = pending_audit
            measured = dict(zip(revealed.candidate_id.astype(str), revealed.y_bind.astype(float)))
            if h1_id in measured and h0_id in measured and measured[h0_id] > measured[h1_id]:
                falsified_positions |= positions
            pending_audit = None

        if checkpoint >= total_budget:
            break
        remaining = work[~work.candidate_id.isin(queried)].copy()
        if method == "m1_random":
            round_picks = rng.choice(
                qualified_pool(remaining, tau_dev).candidate_id.to_numpy(), SLOT_COUNT, replace=False
            ).tolist()
            picks.append(round_picks)
            queried.extend(round_picks)
            continue

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
            # rather than looking them up in ``mu_map`` (which would be empty for
            # exactly the rows we need). The surrogate was fitted on a subset of
            # these rows, so the residual is in-sample for those and honest for
            # the rest — the SAR question ("which substitution surprised the
            # model?") is still the right target, and it is the only residual
            # available without refitting a held-out model every round.
            train_seq_idx = np.asarray([id_to_idx[x] for x in revealed.candidate_id])
            mu_revealed, _ = bs.predict(X[train_seq_idx])
            residuals = {
                str(cid): float(y) - float(mu)
                for cid, y, mu in zip(revealed.candidate_id, revealed.y_bind, mu_revealed)
            }
            policy.prepare(revealed, tau_dev, residuals=residuals)
            round_picks = _pick_m5(
                pool, revealed, tau_dev,
                residuals=residuals,
                queries_spent=len(queried),
                total_budget=total_budget,
                falsified_positions=falsified_positions,
                priors=policy.last_priors,
                batch_size=SLOT_COUNT,
            )
            picks.append(round_picks)
            queried.extend(round_picks)
            pending_audit = (
                round_picks[3],
                round_picks[4],
                {prior.position for prior in policy.last_priors},
            ) if len(round_picks) == SLOT_COUNT else None
            continue

        if method == "m4_fixed_prior":
            pool = pool.copy()
            pool["score"] = pool.a_base + [fixed_expert_prior(s) for s in pool.sequence]
        else:
            pool = pool.assign(score=pool.a_base)
        round_picks = pool.sort_values(["score", "candidate_id"], ascending=[False, True]).candidate_id.head(SLOT_COUNT).tolist()
        picks.append(round_picks)
        queried.extend(round_picks)

    values = [c["best_qualified_binding"] for c in curve if np.isfinite(c["best_qualified_binding"])]
    return BenchmarkResult(method, seed, initial_ids, curve, float(sum(values) / 30.0), picks)


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
    # ``method`` here must be the comprehension's own variable. Referencing the
    # leftover loop variable from the curve loop above silently reported M5's
    # AUBQC under every method name.
    return {"m5_policy_mode": policy.mode, "seeds": seeds, "curves": curves,
            "aubqc": {
                m: float(np.mean([r.aubqc for r in results if r.method == m]))
                for m in METHODS
            },
            "matched_initial_seeds": {str(s): next(r.initial_ids for r in results if r.seed == s) for s in range(seeds)}}
