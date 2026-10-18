"""Bayesian-style experimental design: priors, shrinkage, and the 5-slot portfolio.

Three things live here, deliberately in one module so the benchmark methods and
the live campaign cannot drift apart:

* :func:`loop_prior_score` — the static WT anchor rule shared by M4 and M5.
* :func:`adaptive_prior_score` — ``Δ_agent``: the council's bounded positional
  priors plus an empirical-Bayes-shrunk SAR term computed over surrogate
  **residuals**, so a co-mutation artefact cannot masquerade as a single-site
  effect.
* :func:`select_five_slots` — a paired H1/H0 experiment design: two exploitation
  picks, one optimistic multi-beta exploration pick that anneals with budget, one
  hypothesis-testing pick, and one null-control pick.

Nothing here is calibrated against a particular seed set, and there is no rank
window or "top-N frontier" cutoff: slot 3 and slot 5 are scored by their own
objectives over the *whole* qualified pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ANCHOR_POSITIONS, SOFT_PRIOR_BOUND, WT_SEQUENCE
from .schemas import PositionalSoftPrior, RoundDecisionRecord

# ── Static WT loop prior (M4, shared with M5) ────────────────────────────────
ANCHOR_CONSERVATION_BONUS = 0.15
AROMATIC_SLACK = 3
AROMATIC_PENALTY = 0.05

#: ``loop_prior_score`` is an additive bonus on ``A_base``, which is itself
#: unbounded, so the rule is clipped into the same band as ``Δ_agent``. The two
#: priors are then composable: the anchor rule shifts the ordering, the council
#: re-ranks within it, and neither can drown out the surrogate.
LOOP_PRIOR_BOUND = 0.15

# ── Empirical-Bayes SAR term ─────────────────────────────────────────────────
#: Per-substitution bound for ``Δ_agent`` (SPEC §6's soft-prior band is ±0.15 for
#: a single declared prior; the *aggregate* adaptive vote is held to ±0.06 so it
#: can re-rank neighbours but never override the surrogate's ordering).
AGENT_PRIOR_BOUND = 0.06

#: James-Stein-ish shrinkage constant. A cell holding ``n`` observations keeps
#: ``n / (n + 3)`` of its raw effect: 0.40 at one replicate, 0.57 at two, 0.77 at
#: ten. Cells with ``n < 2`` contribute exactly zero, because a single mutation
#: measuring high is as likely to be a co-mutation or assay artefact as a real
#: positional effect — which is exactly the cold-start case.
SAR_SHRINKAGE_K = 3.0
SAR_MIN_OBSERVATIONS = 2

#: Contribution of one fully-confident, fully-sized substitution effect.
#: ``SAR_DELTA_SCALE`` is the residual (log10 Kd) at which an effect saturates.
SAR_HIT_WEIGHT = 0.03
SAR_DELTA_SCALE = 0.30

#: Optimistic-exploration coefficient for slot 3. Decays as ``1 - spent/budget``,
#: so early rounds behave like a UCB and the last round is pure exploitation
#: without any hand-written round switch.
SLOT3_OPTIMISM = 0.50

SLOT_COUNT = 5


# ── Shared WT loop prior ─────────────────────────────────────────────────────
def loop_prior_score(sequence: str) -> float:
    """WT anchor-conservation loop prior: the rule set shared by M4 **and** M5.

    Conserving the four structural anchors (``W3``, ``Y11``, ``D12``, ``Y13``)
    earns a bounded bonus; carrying more than three aromatics is charged an
    aggregation-liability penalty. M4 adds exactly this term to ``A_base``; M5
    keeps it and layers ``Δ_agent`` on top, so the hybrid inherits the expert
    rule instead of replacing it.
    """
    conserved = sum(sequence[p - 1] == WT_SEQUENCE[p - 1] for p in ANCHOR_POSITIONS) / len(ANCHOR_POSITIONS)
    aromatic_excess = max(0, sum(residue in "FWY" for residue in sequence) - AROMATIC_SLACK)
    return float(
        np.clip(
            ANCHOR_CONSERVATION_BONUS * conserved - AROMATIC_PENALTY * aromatic_excess,
            -LOOP_PRIOR_BOUND,
            LOOP_PRIOR_BOUND,
        )
    )


def fixed_expert_prior(sequence: str) -> float:
    """M4's static prior. The benchmark scores M4 with exactly this term."""
    return loop_prior_score(sequence)


# ── Empirical-Bayes SAR over residuals ───────────────────────────────────────
class SarTable(dict):
    """``{(position, residue): (n, shrunk_delta)}`` with its support and basis.

    A ``dict`` subclass so lookups stay ordinary, carrying two pieces of
    provenance: ``support`` (how many revealed rows the table rests on) and
    ``basis`` (``"residual"`` when built from ``y_bind - mu_bind``, ``"median"``
    when only the qualified-median advantage was available).
    """

    def __init__(self, *args, support: int = 0, basis: str = "median", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support = int(support)
        self.basis = basis


def _shrink(n: int) -> float:
    """``n / (n + K)``, or 0 below :data:`SAR_MIN_OBSERVATIONS`."""
    if n < SAR_MIN_OBSERVATIONS:
        return 0.0
    return n / (n + SAR_SHRINKAGE_K)


def sar_effect_table(
    ledger_df: pd.DataFrame | None,
    *,
    tau_dev: float | None = None,
    residuals: dict[str, float] | None = None,
    min_n: int = 1,
) -> SarTable:
    """Per-position/substitution effect, shrunk by its own sample size.

    The effect of a substitution is measured on the **surrogate residual**
    (``y_bind - mu_bind``) whenever one is available for a revealed candidate,
    and on the advantage over the qualified revealed median otherwise. Residuos
    are the right target because the surrogate already absorbs the additive
    multi-mutation structure: a candidate that measures high *and* was predicted
    high tells us nothing about its individual mutations, whereas one that
    surprises the model does. That is what separates a real positional effect
    from co-mutation confounding.

    Only revealed rows are ever read, and only the ``(position, residue)`` pairs
    that differ from the wild type are counted. ``n < 2`` cells are omitted
    entirely, and every retained effect is multiplied by ``n / (n + 3)``.
    """
    if ledger_df is None or getattr(ledger_df, "empty", True):
        return SarTable()
    if not {"sequence", "y_bind"} <= set(ledger_df.columns):
        return SarTable()
    frame = ledger_df.loc[ledger_df.y_bind.notna()].copy()
    if tau_dev is not None and "y_dev" in frame.columns:
        # An unqualified high binder teaches the wrong substitution: it can never
        # contribute to best_qualified_binding, so it must not steer the prior.
        qualified = frame.loc[frame.y_dev >= tau_dev]
        if len(qualified):
            frame = qualified
    if frame.empty:
        return SarTable()

    basis = "median"
    if residuals:
        values = frame.candidate_id.astype(str).map(residuals)
        if values.notna().mean() > 0.5:
            signal = values.astype(float)
            basis = "residual"
        else:
            signal = frame.y_bind.astype(float) - float(frame.y_bind.median())
    else:
        signal = frame.y_bind.astype(float) - float(frame.y_bind.median())

    frame = frame.assign(_signal=signal.to_numpy())
    effects = SarTable(support=len(frame), basis=basis)
    sequences = frame.sequence.astype(str)
    for position in range(13):
        column = sequences.str[position]
        mutated = column != WT_SEQUENCE[position]
        if not mutated.any():
            continue
        for residue, group in frame.loc[mutated].groupby(column[mutated]):
            n = int(len(group))
            if n < max(min_n, SAR_MIN_OBSERVATIONS):
                continue
            raw = float(group._signal.mean())
            # Bound the *saturated* contribution, then shrink toward zero.
            magnitude = min(1.0, abs(raw) / SAR_DELTA_SCALE)
            effects[(position + 1, str(residue))] = (
                n,
                float(np.sign(raw)) * magnitude * _shrink(n),
            )
    return effects


def _sar_total(sequence: str, effects: dict | None) -> float:
    """Summed contribution of a candidate's non-WT substitutions."""
    if not effects:
        return 0.0
    total = 0.0
    for position in range(13):
        residue = sequence[position]
        if residue == WT_SEQUENCE[position]:
            continue
        stat = effects.get((position + 1, residue))
        if stat is None:
            continue
        total += SAR_HIT_WEIGHT * float(stat[1])
    return total


def sar_prior_score(
    sequence: str,
    effects: dict | None,
    *,
    bound: float = AGENT_PRIOR_BOUND,
) -> float:
    """Empirical-SAR contribution of one candidate, bounded."""
    return float(np.clip(_sar_total(sequence, effects), -bound, bound))


# ── Adaptive agent prior (Δ_agent) ───────────────────────────────────────────
def _soft_prior_total(
    sequence: str,
    priors: list[PositionalSoftPrior],
    falsified_positions: set[int] | frozenset[int] = frozenset(),
) -> float:
    """Summed positional-prior contribution, with falsified rules inverted.

    Summed, not averaged: averaging over thirteen positions diluted one
    confident rule to ``weight_delta / 13``, which made the council's vote
    invisible next to an ``A_base`` of order one.

    A position whose rule was graded ``FALSIFIED`` in a previous round is
    **inverted**: the system already knows that residue does not help, so a
    candidate carrying it should be pushed down rather than merely unboosted.
    """
    total = 0.0
    for prior in priors:
        residue = sequence[prior.position - 1]
        sign = -1.0 if prior.position in falsified_positions else 1.0
        if residue in prior.preferred_residues:
            total += sign * abs(prior.weight_delta)
        elif residue in prior.discouraged_residues:
            total -= sign * abs(prior.weight_delta)
    return total


def _saturate(value: float, bound: float) -> float:
    """Smoothly bound ``value`` to ``[-bound, bound]``.

    ``tanh`` rather than a hard clip: a hard clip maps several strong hits onto
    exactly ``+bound`` and stops distinguishing them, which leaves the selector
    tie-breaking on candidate id. ``tanh`` is monotone, so the ordering of
    saturated candidates is preserved.
    """
    return float(bound * np.tanh(value / bound))


def soft_prior_score(sequence: str, priors: list[PositionalSoftPrior]) -> float:
    """The council's declared positional priors alone, bounded."""
    return _saturate(_soft_prior_total(sequence, priors), SOFT_PRIOR_BOUND)


def adaptive_prior_score(
    sequence: str,
    *,
    priors: tuple[PositionalSoftPrior, ...] | list[PositionalSoftPrior] = (),
    sar_effects: dict | None = None,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
    bound: float = AGENT_PRIOR_BOUND,
) -> float:
    """``Δ_agent``: summed soft priors **plus** the shrunk empirical SAR term.

    Both terms are evidence about the same candidate, so they are summed before
    saturation. ``A_H1 = A_null + Δ_agent`` therefore strictly augments M4's
    ordering rather than replacing it, and ``|Δ_agent| ≤ 0.06`` keeps the
    surrogate in charge of the ordering.
    """
    raw = _soft_prior_total(sequence, list(priors), falsified_positions) + _sar_total(sequence, sar_effects)
    return _saturate(raw, bound)


def positional_priors_from_revealed(
    revealed: pd.DataFrame,
    *,
    top_k: int = 5,
    weight_delta: float = 0.04,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
) -> list[PositionalSoftPrior]:
    """Derive soft priors from the best revealed sequences.

    The deterministic stand-in for a council memo (``adaptive_tool`` mode) and
    the offline fallback. Positions already graded ``FALSIFIED`` are skipped, so
    the fallback self-corrects on the same evidence the live planner sees.
    """
    if revealed is None or getattr(revealed, "empty", True):
        return []
    if not {"sequence", "y_bind"} <= set(revealed.columns):
        return []
    best = revealed.sort_values("y_bind", ascending=False).head(max(1, min(top_k, len(revealed))))
    priors: list[PositionalSoftPrior] = []
    for position in range(13):
        if position + 1 in falsified_positions:
            continue
        residue = best.sequence.str[position].mode().iloc[0]
        if residue == WT_SEQUENCE[position]:
            continue
        priors.append(
            PositionalSoftPrior(
                position=position + 1,
                preferred_residues=[str(residue)],
                discouraged_residues=[],
                weight_delta=weight_delta,
            )
        )
    return priors


# ── Qualification gate ───────────────────────────────────────────────────────
def qualified_pool(frame: pd.DataFrame, tau_dev: float | None) -> pd.DataFrame:
    """Restrict a candidate frame to ``y_dev >= tau_dev`` when that leaves a batch.

    ``y_dev`` is derived from sequence-only proxies and is therefore known at
    ``t=0`` (SPEC §2), so gating on it leaks no label. A slot spent on a
    candidate that cannot qualify is wasted: only qualified candidates can ever
    move ``best_qualified_binding`` or AUBQC. When fewer than five qualified
    candidates remain, the unfiltered frame is returned rather than
    under-filling the batch.
    """
    if tau_dev is None or "y_dev" not in frame.columns:
        return frame
    qualified = frame.loc[frame.y_dev >= tau_dev]
    return qualified if len(qualified) >= SLOT_COUNT else frame


# ── The 5-slot portfolio ─────────────────────────────────────────────────────
def select_five_slots(
    unqueried: pd.DataFrame,
    decision: RoundDecisionRecord | None = None,
    *,
    ledger: pd.DataFrame | None = None,
    tau_dev: float | None = None,
    sar_effects: dict | None = None,
    residuals: dict[str, float] | None = None,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
    queries_spent: int | None = None,
    total_budget: int | None = None,
) -> list[dict]:
    # ``queries_spent`` / ``total_budget`` drive slot 3's annealing. When a
    # caller omits them, derive ``queries_spent`` from the revealed ledger (the
    # only defensible source) and take the budget from settings. Falling back to
    # "0 spent" would leave a late round maximally exploratory.
    """Qualified 5-slot portfolio as a paired H1/H0 experiment (SPEC §6).

    Two acquisition scores are defined over the qualified unqueried pool::

        A_null(s) = A_base(s) + loop_prior_score(s)
        A_H1(s)   = A_null(s) + Δ_agent(s)

    ``A_null`` is the shared M4 objective (surrogate + WT anchor rule); ``A_H1``
    is M5's, which adds the council's bounded adaptive prior. The batch then
    spends its five slots as:

    1-2  top 2 by ``A_H1`` — core exploitation.
    3    ``A_null + (1 - spent/budget) · 0.50 · sigma_joint`` maximised over the
         whole pool — an *optimistic* UCB whose exploration weight anneals to
         zero as budget is spent, so late rounds are pure exploitation. Raw
         ``max sigma_joint`` is deliberately not used: unconstrained, it picks
         the least-informed candidate in the pool regardless of whether that
         candidate could ever win.
    4    top ``A_H1`` among ``Δ_agent > 0`` — the H1 positive arm, i.e. a
         candidate the live hypothesis actually favours. Falls back to the
         unconstrained ``A_H1`` argmax if the prior boosts nobody (round 1).
    5    top ``A_null`` among ``Δ_agent <= 0`` — the H0 null-control arm. This is
         the paired counterfactual: if the agents are wrong, slot 5 still spends
         on the best null-frontier candidate, so a bad hypothesis cannot starve
         the batch of exploitative picks.

    Because slot 4 and slot 5 are selected on the *same* batch from opposite
    signs of ``Δ_agent``, every round yields a direct H1-vs-H0 comparison for
    :func:`core.tool_arsenal.audit_past_hypotheses` to grade.
    """
    if len(unqueried) < SLOT_COUNT:
        raise ValueError("At least five unqueried candidates are required")
    if queries_spent is None or total_budget is None:
        from config.settings import SETTINGS  # local import keeps core importable alone

        if total_budget is None:
            total_budget = SETTINGS.total_budget
        if queries_spent is None:
            # The revealed ledger is the only defensible source: assuming "0
            # spent" would leave every late round maximally exploratory, which is
            # the opposite of annealing.
            revealed = 0 if ledger is None else int(len(ledger))
            queries_spent = revealed or int(SETTINGS.initial_queries)

    frame = unqueried.copy()
    if sar_effects is None:
        sar_effects = sar_effect_table(ledger, tau_dev=tau_dev, residuals=residuals)
    priors = (
        list(decision.positional_priors)
        if decision
        else positional_priors_from_revealed(ledger, falsified_positions=falsified_positions)
    )

    frame["loop_prior"] = [loop_prior_score(s) for s in frame.sequence]
    frame["delta_agent"] = [
        adaptive_prior_score(
            s, priors=priors, sar_effects=sar_effects, falsified_positions=falsified_positions
        )
        for s in frame.sequence
    ]
    frame["a_null"] = frame.a_base + frame.loop_prior
    frame["a_h1"] = frame.a_null + frame.delta_agent
    pool = qualified_pool(frame, tau_dev)

    # Annealing exploration weight: 1.0 at the start, 0.0 once budget is spent.
    remaining_budget = max(0, int(total_budget) - int(queries_spent))
    exploration = (remaining_budget / float(total_budget)) if total_budget else 0.0
    pool = pool.assign(optimistic=pool.a_null + exploration * SLOT3_OPTIMISM * pool.sigma_joint)

    selected: list[dict] = []

    def take(row: pd.Series, slot_type: str) -> None:
        selected.append({"slot_id": len(selected) + 1, "slot_type": slot_type, **row.to_dict()})

    def remaining() -> pd.DataFrame:
        used = {row["candidate_id"] for row in selected}
        return pool.loc[~pool.candidate_id.isin(used)]

    def best(frame_slice: pd.DataFrame, column: str) -> pd.Series | None:
        if frame_slice.empty:
            return None
        return frame_slice.sort_values([column, "candidate_id"], ascending=[False, True]).iloc[0]

    # Slots 1-2: core exploitation on A_H1.
    for _, row in pool.sort_values(["a_h1", "candidate_id"], ascending=[False, True]).head(2).iterrows():
        take(row, "Dual-Surrogate UCB")

    # Slot 3: optimistic multi-beta UCB, annealed by remaining budget.
    rest = remaining()
    take(best(rest, "optimistic"), "Max Joint Uncertainty")

    # Slot 4: H1 arm. The planner's declared target can only *tie-break*; it can
    # never force a candidate the acquisition score does not rank.
    rest = remaining()
    supported = rest.loc[rest.delta_agent > 0]
    hypothesis = best(supported, "a_h1") if not supported.empty else best(rest, "a_h1")
    take(hypothesis, "Agent Hypothesis Test")

    # Slot 5: H0 null-control arm, on the unmodified null objective.
    rest = remaining()
    control = rest.loc[rest.delta_agent <= 0]
    probe = best(control, "a_null") if not control.empty else best(rest, "a_null")
    take(probe, "Skeptic Falsification Probe")

    picked = [row["candidate_id"] for row in selected]
    if len(set(picked)) != len(picked):
        raise RuntimeError("Slot selection produced duplicate candidates")
    return selected
