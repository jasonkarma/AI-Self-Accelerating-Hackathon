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
from .constants import ANCHOR_POSITIONS, CHECKPOINT_QUERIES, WT_SEQUENCE
from .developability import RiskNormalizer, raw_components
from .features import encode_sequences
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions
from .tool_arsenal import (
    H1_ARM_KEY,
    H1_EXPLOIT_SLOT_IDS,
    _tested_positions,
    audit_past_hypotheses,
)

#: The preregistered ablation. Every member starts from the *same* five matched
#: random seeds, so a difference between two rows is attributable to post-seed
#: feedback learning rather than to a luckier seed batch.
METHODS = ("m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")

#: Published zero-shot affinity predictors on 1N8Z, as Spearman ``rho`` against
#: measured binding (AbBiBench). They are the only external reference points on
#: this panel, and they answer a different question from AUBQC: *how well does a
#: model rank the pool without ever querying it*, versus *how well does a search
#: process spend a budget*. Reported side by side, never merged.
ZERO_SHOT_REFERENCES: dict[str, float] = {
    "alphafold3": -0.16,
    "proteinmpnn": -0.17,
    "esm2": -0.13,
}

#: Key of the live trajectory inside ``same_seed_comparison``. It is *not* a
#: method and is never averaged into the seed medians; it is the recorded campaign,
#: shown on its own seed next to the five matched baselines.
LIVE_LABEL = "live_m5"

#: Optional sixth row: identical to ``m5_hybrid`` except that Round 0's five
#: queries are chosen by the cold-start prior instead of at random. Reported
#: separately so it can never contaminate the matched-seed M1-M5 comparison.
WARM_START_METHOD = "m5_warm_start"


def _is_m5(method: str) -> bool:
    return method in {"m5_hybrid", WARM_START_METHOD}


def _method_index(method: str) -> int:
    """RNG stream index for a method, treating the warm start as M5."""
    return METHODS.index(method if method in METHODS else "m5_hybrid")


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


#: Liability motifs the developability proxy itself charges for. A cold start
#: that seeds the campaign with a deamidation/isomerisation motif or an unpaired
#: cysteine teaches the surrogate about molecules that could never ship.
LIABILITY_MOTIFS: tuple[str, ...] = ("NG", "NS", "DG", "DS")


def warm_start_ids(
    frame: pd.DataFrame,
    normalizer: RiskNormalizer,
    tau_dev: float,
    *,
    k: int = SLOT_COUNT,
    seed: int = 0,
) -> list[str]:
    """Round-0 queries chosen by the Virtual Lab's cold-start prior.

    Used **only** by the optional ``m5_warm_start`` ablation. M1-M5 keep their
    five matched random seeds so the primary ablation isolates post-seed
    learning; this row exists to show what the loop buys *before* it has any
    labels, without letting that advantage leak into the matched comparison.

    The rule uses sequence-only information available at ``t=0`` (SPEC §2) and
    never touches ``y_bind``:

    1. **Qualified** — ``y_dev >= tau_dev``. An unqualified seed query cannot
       contribute to ``best_qualified_binding``.
    2. **Liability-free** — no unpaired cysteine and none of
       :data:`LIABILITY_MOTIFS`, so the seed batch does not spend its first
       assays on molecules that fail developability on inspection.
    3. **Informative** — at least two non-WT substitutions, otherwise a seed
       row carries almost no positional SAR information.
    4. **Diverse and anchor-covering** — greedy maximin Hamming selection over
       the remaining eligible pool, so the five seeds probe five different
       regions of the 260-cell substitution space instead of five neighbours.
       One slot is reserved for an anchor-conserving scaffold when one exists,
       because the paired audit needs a candidate whose rule set is empty.
    """
    scored = [
        {
            "candidate_id": str(cid),
            "sequence": str(sequence),
            "y_dev": float(normalizer.score(str(sequence))["y_dev"]),
        }
        for cid, sequence in zip(frame.candidate_id, frame.sequence)
    ]
    eligible = [
        row
        for row in scored
        if row["y_dev"] >= tau_dev
        and "C" not in row["sequence"]
        and not any(motif in row["sequence"] for motif in LIABILITY_MOTIFS)
        and sum(a != b for a, b in zip(row["sequence"], WT_SEQUENCE)) >= 2
    ]
    if len(eligible) < k:
        # Fall back to the qualified pool, then to the raw frame, rather than
        # under-filling the seed batch.
        eligible = [row for row in scored if row["y_dev"] >= tau_dev] or scored
    if not eligible:
        return []

    def mutations(row: dict) -> set[int]:
        return {p for p in range(13) if row["sequence"][p] != WT_SEQUENCE[p]}

    def hamming(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b))

    # Deterministic base order. Deliberately *not* sorted by ``y_dev`` descending:
    # ``y_dev`` is a developability proxy, so maximising it selects the most
    # developable molecules, which on this landscape are among the weakest
    # binders. A cold start has no binding labels, so the only defensible
    # criterion is information — spread the five queries across the substitution
    # space and let the loop do the exploiting.
    eligible.sort(key=lambda row: row["candidate_id"])
    anchors = set(ANCHOR_POSITIONS)

    # Slot 1: an anchor-conserving scaffold when one exists, so the batch contains
    # a candidate whose positional rule set is empty (the clean H0 reference).
    # Among scaffolds, take the one carrying the *most* non-anchor substitutions:
    # equally valid as an H0 reference, strictly more positional information per
    # assay. Selecting by ``candidate_id`` instead would hand the slot to whichever
    # id happens to sort first — an irrelevant criterion that on this pool happens
    # to coincide with the best binder, which would flatter the ablation for a
    # reason the rule never earned.
    scaffold = max(
        (row for row in eligible if not (mutations(row) & anchors)),
        key=lambda row: (len(mutations(row)), row["candidate_id"]),
        default=None,
    )
    if scaffold is not None:
        picked = [scaffold]
    else:
        # No anchor-conserving qualified molecule: anchor the batch on the
        # candidate furthest from the pool centroid in mutation space instead.
        picked = [
            max(
                eligible,
                key=lambda row: (len(mutations(row)), row["candidate_id"]),
            )
        ]

    # Remaining slots: greedy maximin on Hamming distance, preferring candidates
    # that reach an anchor position the batch has not yet probed, then those with
    # more non-WT substitutions (more SAR information per assay).
    while len(picked) < min(k, len(eligible)):
        covered = set().union(*(mutations(row) & anchors for row in picked))
        best = max(
            (row for row in eligible if row not in picked),
            key=lambda row: (
                bool((mutations(row) & anchors) - covered),
                min(hamming(row["sequence"], chosen["sequence"]) for chosen in picked),
                len(mutations(row)),
                row["candidate_id"],
            ),
        )
        picked.append(best)

    return [row["candidate_id"] for row in picked]


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
) -> list[dict]:
    """M5's five slots, delegated to the single shared portfolio.

    ``select_five_slots`` takes an optional ``RoundDecisionRecord`` and derives
    priors itself when there is none; the benchmark has no planner, so it passes
    its own ``AdaptiveToolPolicy`` priors through a minimal stand-in record so
    both paths converge on the identical ``Δ_agent``.

    Returns the **full slot records**, not just ids. The paired H1/H0 audit needs
    the exploitation arm (slots 1, 2, 4 with ``Δ_agent >= 0``) and each pick's
    pre-query ``mu_bind`` — the residual gate that decides which positions may be
    inverted is ``y_bind - mu_bind``, and a bare id list cannot express it. The
    live campaign stamps exactly the same payload in
    ``Campaign._annotate_decision``, so benchmark M5 and live M5 grade their
    rounds through one code path.
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
    return select_five_slots(
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


def run_method(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, method: str, seed: int,
               initial_ids: list[str], m5_policy: AdaptiveToolPolicy | LiveAgentPolicy | None = None,
               *, seed_floor: float | None = None) -> BenchmarkResult:
    """Run one method on one seed and return its full discovery record.

    ``seed_floor`` overrides the baseline of the **normalized** curve. The
    default is the method's own first checkpoint, which is the right choice for
    M1-M5 because they share the same five initial queries. The cold-start
    ablation seeds itself differently, so normalizing it against its own
    (stronger) round-0 score would shrink the headroom it is measured against and
    report a *lower* fraction of achievable gain for a strictly better start. It
    therefore passes the seed's matched random ``y*(5)`` instead, so every row on
    a given seed is expressed against the same ``[0%, 100%]`` ruler.
    """
    rng = np.random.default_rng(seed + _method_index(method) * 10_000)
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
    #: (the batch's five slot records, the round's declared priors) for the batch
    #: that was just designed, so the paired H1/H0 audit can run once it is
    #: measured. The full records carry each pick's ``Δ_agent`` (to build the H1
    #: arm) and ``mu_bind`` (for the residual gate).
    pending_audit: tuple[list[dict], list] | None = None

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

        # Paired H1/H0 audit of the previous batch. This is **not** re-implemented
        # here: the decision payload is assembled exactly as the live campaign
        # assembles it (measured slots 1/2/4 as the H1 arm, slot 5 as the H0
        # control, plus each pick's pre-query ``mu_bind``) and handed to
        # :func:`core.tool_arsenal.audit_past_hypotheses`. Two consequences, both
        # deliberate:
        #
        # * benchmark M5 and live M5 grade their rounds through one code path, so
        #   a fix to the audit cannot land in one and not the other; and
        # * the residual gate lives in the audit, so the benchmark no longer
        #   inverts a position whose measure/surrogate residual was positive.
        if pending_audit is not None:
            slot_records, prior_list = pending_audit
            arm_ids = [
                str(slot["candidate_id"])
                for slot in slot_records
                if int(slot.get("slot_id", 0)) in H1_EXPLOIT_SLOT_IDS
                and float(slot.get("delta_agent", 0.0) or 0.0) >= 0.0
            ]
            audit = audit_past_hypotheses(
                revealed,
                [{
                    "hypothesis_id": f"HYP-BENCH-R{checkpoint // SLOT_COUNT}",
                    "round_number": None,
                    "slot4_measured_id": str(slot_records[3]["candidate_id"]),
                    "slot5_measured_id": str(slot_records[4]["candidate_id"]),
                    H1_ARM_KEY: arm_ids,
                    "positional_priors": [prior.model_dump() for prior in prior_list],
                    "pre_query_expectation": {
                        str(slot["candidate_id"]): float(slot["mu_bind"])
                        for slot in slot_records
                    },
                }],
            )
            falsified_positions |= set(audit.get("falsified_positions") or ())
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

            if _is_m5(method):
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
                delta_agents = np.asarray([
                    adaptive_prior_score(
                        s,
                        priors=policy.last_priors,
                        sar_effects=policy.last_sar,
                        falsified_positions=falsified_positions,
                    )
                    for s in pool.sequence
                ])
                # M5's *world model* for the rank-correlation metric is the
                # binding mean plus the agent prior, **not** ``A_base``.
                # ``A_base`` carries ``beta_0 = 1.0`` tree variance and a
                # ``0.35 · (mu_d + sigma_d)`` developability margin, neither of
                # which is a binding prediction — scoring M5 with it measured
                # something other than what M5 actually believes about affinity
                # and depressed its rho for a scale reason, not a science one.
                # ``Δ_agent`` is bounded to ±0.06 in *log10 Kd*, so it is
                # expressed in ``mu_bind`` units by multiplying by the pool's
                # ``mu_bind`` spread before the two are added.
                delta_span = float(np.ptp(pool.mu_bind.to_numpy())) if len(pool) > 1 else 0.0
                pool = pool.assign(score=pool.mu_bind + delta_agents * delta_span)
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

            if _is_m5(method):
                slot_records = _pick_m5(
                    pool, revealed, tau_dev,
                    residuals=residuals,
                    queries_spent=len(queried),
                    total_budget=total_budget,
                    falsified_positions=falsified_positions,
                    priors=policy.last_priors,
                    batch_size=SLOT_COUNT,
                )
                round_picks = [str(slot["candidate_id"]) for slot in slot_records]
                # Keep the full records, not just the ids: the audit needs each
                # pick's ``Δ_agent`` (to build the H1 arm) and ``mu_bind`` (for the
                # residual gate).
                pending_audit = (
                    list(slot_records),
                    list(policy.last_priors),
                ) if len(slot_records) == SLOT_COUNT else None
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
    # ``seed_floor`` lets the cold-start ablation share the matched random
    # baseline as its 0% mark instead of using its own stronger round-0 score.
    floor = seed_binding if seed_floor is None else float(seed_floor)
    span = pool_ceiling - floor if floor is not None and np.isfinite(pool_ceiling) else float("nan")
    normalized_curve = []
    for point in curve:
        value = point["best_qualified_binding"]
        normalized = (
            (value - floor) / span
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


def _median_curve(
    results: list[BenchmarkResult],
    series: str,
    value_key: str,
    *,
    digits: int,
    methods: tuple[str, ...] | None = None,
) -> list[dict]:
    """Median across seeds of one per-checkpoint series, one column per method."""
    names = METHODS if methods is None else methods
    curve = []
    for q in range(5, 36, 5):
        row: dict = {"query_count": q}
        for name in names:
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


def _summary_for(results: list[BenchmarkResult]) -> dict:
    """The four self-learning metrics for one method subset."""
    hit_rows = [r for r in results if r.top_decile_queries]
    return {
        "aubqc": float(np.mean([r.aubqc for r in results])) if results else float("nan"),
        "normalized_aubqc": (
            float(np.mean([r.normalized_aubqc for r in results])) if results else float("nan")
        ),
        "top5_panel_aubqc": (
            float(np.mean([r.panel_aubqc for r in results])) if results else float("nan")
        ),
        "top_decile_hit_rate": (
            float(sum(r.top_decile_hits for r in hit_rows) / sum(r.top_decile_queries for r in hit_rows))
            if hit_rows
            else float("nan")
        ),
        "spearman_rho_at_q35": _mean_rho(results, results[0].method, 35) if results else None,
    }


def _paired_test(
    results: list[BenchmarkResult], treatment: str, control: str, metric: str
) -> dict:
    """Paired t-test of ``treatment - control`` across matched seeds.

    Paired, not pooled: every seed reuses the *same* five initial queries for all
    methods, so the only variation removed by pairing is the seed's own luck —
    which is precisely the nuisance term. A pooled test would be dominated by
    seed difficulty rather than by method quality.
    """
    from scipy.stats import t as student_t

    def per_seed(method: str) -> dict[int, float]:
        values: dict[int, float] = {}
        for result in results:
            if result.method != method:
                continue
            if metric == "spearman_rho_at_q35":
                point = next((p for p in result.rho_curve if p["query_count"] == 35), None)
                rho = None if point is None else point["spearman_rho"]
                if rho is not None and np.isfinite(rho):
                    values[result.seed] = float(rho)
            elif metric == "top_decile_hit_rate":
                if result.top_decile_queries:
                    values[result.seed] = result.top_decile_hits / result.top_decile_queries
            else:
                values[result.seed] = float(getattr(result, metric))
        return values

    treated, controlled = per_seed(treatment), per_seed(control)
    seeds = sorted(set(treated) & set(controlled))
    diffs = np.asarray([treated[s] - controlled[s] for s in seeds], dtype=float)
    n = int(diffs.size)
    if n < 2:
        return {
            "metric": metric, "n": n, "mean_diff": float("nan"), "se": float("nan"),
            "t_stat": float("nan"), "p_value": float("nan"), "seeds_won": 0,
        }
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    t_stat = float(mean / se) if se > 0 else float("inf") if mean != 0 else 0.0
    # Two-sided p from the exact t distribution; ``inf`` SE-degenerate cases are
    # reported with p = 0 when the effect is strictly non-zero.
    if se > 0:
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    else:
        p_value = 1.0 if mean == 0 else 0.0
    return {
        "metric": metric,
        "n": n,
        "mean_diff": mean,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "seeds_won": int(sum(d > 0 for d in diffs)),
    }


#: Metrics reported in the paired comparison table, in reading order.
PAIRED_METRICS: tuple[str, ...] = (
    "aubqc",
    "normalized_aubqc",
    "panel_aubqc",
    "top_decile_hit_rate",
    "spearman_rho_at_q35",
)


def _curve_summary(curve: list[dict], aubqc: float) -> dict:
    """``q=5`` / ``q=35`` / AUBQC triple for one curve, carrying ``y*(q)`` forward."""
    recorded = {
        int(point["query_count"]): float(point["best_qualified_binding"])
        for point in curve
        if point.get("best_qualified_binding") is not None
    }

    def at(checkpoint: int) -> float:
        if not recorded:
            return float("nan")
        value = recorded[sorted(recorded)[0]]
        for count in sorted(recorded):
            if count <= checkpoint:
                value = recorded[count]
        return value

    return {"q": 5, "q5": at(5), "q35": at(35), "aubqc": float(aubqc)}


def same_seed_comparison(
    frame: pd.DataFrame,
    normalizer: RiskNormalizer,
    tau_dev: float,
    *,
    seed: int,
    initial_ids: list[str] | None = None,
    live_curve: list[dict] | None = None,
    live_aubqc: float | None = None,
    m5_policy_mode: str = "adaptive_tool",
) -> dict:
    """The live campaign against M1-M5 **on its own seed**, at ``q=5 / q=35 / AUBQC``.

    A 100-seed median answers "does the design work on average?" and cannot
    attribute one campaign's result: a seed whose five random probes happen to
    contain no qualified binder starts from a lower ``y*(5)`` and therefore has
    more headroom. Re-running the five matched baselines on the identical five
    initial queries costs a few hundred milliseconds and makes the comparison
    like-for-like instead of seed-median-against-one-draw.

    The live row is labelled :data:`LIVE_LABEL` and is **not** a benchmark method:
    its curve is passed in rather than re-derived, because the point of the row is
    what the recorded campaign actually did. It is never folded into a seed median.

    Returns ``{"seed", "methods", "trajectory", "summary", "ranking", "live"}``:
    ``trajectory`` is the per-checkpoint comparison the UI plots on one axis, and
    ``summary``/``ranking`` answer ``q=5``, ``q=35`` and AUBQC per method.
    """
    if initial_ids is None:
        initial_ids = (
            np.random.default_rng(seed)
            .choice(frame.candidate_id.to_numpy(), SLOT_COUNT, replace=False)
            .tolist()
        )
    initial_ids = [str(candidate_id) for candidate_id in initial_ids]
    results = {
        method: run_method(
            frame, normalizer, tau_dev, method, int(seed), initial_ids,
            LiveAgentPolicy() if m5_policy_mode == "live_agent" else AdaptiveToolPolicy(),
        )
        for method in METHODS
    }
    summary = {
        method: _curve_summary(result.curve, result.aubqc)
        for method, result in results.items()
    }

    live_curve = list(live_curve or [])
    live_aubqc = float("nan") if live_aubqc is None else float(live_aubqc)
    if live_curve:
        summary[LIVE_LABEL] = _curve_summary(live_curve, live_aubqc)
    live_points = {int(point["query_count"]): point.get("best_qualified_binding") for point in live_curve}

    trajectory = []
    for checkpoint in CHECKPOINT_QUERIES:
        row: dict = {"query_count": checkpoint}
        for method, result in results.items():
            row[method] = next(
                (point["best_qualified_binding"] for point in result.curve
                 if point["query_count"] == checkpoint),
                None,
            )
        # The live campaign only has a point at a checkpoint it has actually spent
        # through; ``None`` is the honest value before that, and it keeps the plot
        # from drawing a live line into rounds that never happened.
        row[LIVE_LABEL] = live_points.get(checkpoint)
        trajectory.append(row)

    ranked = sorted(
        ((name, block["aubqc"]) for name, block in summary.items() if name != LIVE_LABEL),
        key=lambda item: item[1],
        reverse=True,
    )
    best_baseline_name, best_baseline_aubqc = ranked[0] if ranked else (None, float("nan"))
    gap = live_aubqc - best_baseline_aubqc
    return {
        "seed": int(seed),
        "initial_ids": initial_ids,
        "methods": list(METHODS),
        "trajectory": trajectory,
        "summary": summary,
        "ranking": [
            {"rank": index + 1, "method": name, "aubqc": float(value)}
            for index, (name, value) in enumerate(ranked)
        ],
        "live": {
            "label": LIVE_LABEL,
            "q5": summary.get(LIVE_LABEL, {}).get("q5"),
            "q35": summary.get(LIVE_LABEL, {}).get("q35"),
            "aubqc": None if np.isnan(live_aubqc) else live_aubqc,
            "best_baseline": best_baseline_name,
            "best_baseline_aubqc": None if np.isnan(best_baseline_aubqc) else best_baseline_aubqc,
            "aubqc_gap_vs_best_baseline": None if np.isnan(gap) else round(gap, 4),
        },
    }


def _seed_floor(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float,
                initial_ids: list[str]) -> float:
    """Best qualified binding among a seed's five initial queries.

    This is the ``q=5`` checkpoint — and therefore the 0% mark of the normalized
    curve — that every method on that seed shares. Computed once here so the
    cold-start ablation can be normalised against the *matched random* baseline
    rather than against its own stronger start.
    """
    sequences = dict(zip(frame.candidate_id.astype(str), frame.sequence.astype(str)))
    bindings = dict(zip(frame.candidate_id.astype(str), frame.y_bind.astype(float)))
    qualified = [
        bindings[cid]
        for cid in initial_ids
        if normalizer.score(sequences[cid])["y_dev"] >= tau_dev and np.isfinite(bindings[cid])
    ]
    return float(max(qualified)) if qualified else float("nan")


def run_benchmark(frame: pd.DataFrame, normalizer: RiskNormalizer, tau_dev: float, *, seeds: int = 10,
                  m5_policy_mode: str = "adaptive_tool", include_warm_start: bool = True) -> dict:
    results = []
    warm_results: list[BenchmarkResult] = []
    policy = LiveAgentPolicy() if m5_policy_mode == "live_agent" else AdaptiveToolPolicy()
    for seed in range(seeds):
        initial = np.random.default_rng(seed).choice(frame.candidate_id.to_numpy(), 5, replace=False).tolist()
        # The primary ablation: all five methods share these exact five seeds.
        results.extend(run_method(frame, normalizer, tau_dev, m, seed, initial, policy) for m in METHODS)
        if include_warm_start:
            # The optional sixth row re-seeds its own Round 0 from the cold-start
            # prior. It is *not* part of the matched M1-M5 comparison, and its
            # normalized curve is measured from the *matched random* floor so every
            # row on this seed reads against the same [0%, 100%] ruler.
            floor = _seed_floor(frame, normalizer, tau_dev, initial)
            warm_results.append(
                run_method(
                    frame, normalizer, tau_dev, WARM_START_METHOD, seed,
                    warm_start_ids(frame, normalizer, tau_dev, seed=seed),
                    LiveAgentPolicy() if m5_policy_mode == "live_agent" else AdaptiveToolPolicy(),
                    seed_floor=floor,
                )
            )
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

    paired = {
        f"{performer}_minus_{baseline}": {
            "performer": performer,
            "baseline": baseline,
            "metrics": {
                metric: _paired_test(results, performer, baseline, metric)
                for metric in PAIRED_METRICS
            },
        }
        for performer, baseline in (("m5_hybrid", "m3_ucb"), ("m5_hybrid", "m2_greedy"))
    }
    if warm_results:
        paired[f"{WARM_START_METHOD}_minus_m5_hybrid"] = {
            "performer": WARM_START_METHOD,
            "baseline": "m5_hybrid",
            "metrics": {
                metric: _paired_test(warm_results + results, WARM_START_METHOD, "m5_hybrid", metric)
                for metric in PAIRED_METRICS
            },
        }

    # ``method`` here must be the comprehension's own variable. Referencing the
    # leftover loop variable from the curve loop above silently reported M5's
    # AUBQC under every method name.
    report = {
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
        "paired_t_tests": paired,
        # Published zero-shot affinity predictors on 1N8Z (AbBiBench), as Spearman
        # rho against measured binding. They measure a *model's ranking of the
        # pool it never queried*, which is the one axis comparable to our
        # ``spearman_rho_curves`` — AUBQC measures a search process instead. Shipped
        # in the artifact so the API can never serve a benchmark that omits them.
        "zero_shot_references": dict(ZERO_SHOT_REFERENCES),
        "zero_shot_definition": (
            "Spearman rho of a zero-shot affinity predictor's ranking of the 1N8Z "
            "HCDR3 pool against measured binding (AbBiBench); compare with "
            "spearman_rho_curves, not with AUBQC."
        ),
    }
    if warm_results:
        # The optional cold-start ablation lives under its own key. It is never
        # merged into ``aubqc``/``curves`` so the matched M1-M5 comparison stays
        # strictly seed-for-seed.
        warm_curve = _median_curve(
            warm_results, "curve", "best_qualified_binding", digits=3,
            methods=(WARM_START_METHOD,),
        )
        warm_panel = _median_curve(
            warm_results, "panel_curve", "top5_panel_binding", digits=3,
            methods=(WARM_START_METHOD,),
        )
        report["m5_warm_start"] = {
            "definition": (
                "m5_hybrid with Round 0 (q=5) chosen by the Virtual Lab cold-start "
                "prior instead of five matched random draws: qualified "
                "(y_dev >= tau_dev), liability-free, >=2 non-WT substitutions, and "
                "greedy maximin sequence diversity with anchor coverage."
            ),
            "curves": warm_curve,
            "top5_panel_curves": warm_panel,
            "summary": _summary_for(warm_results),
            "initial_ids": {
                str(r.seed): r.initial_ids for r in warm_results
            },
        }
    return report
