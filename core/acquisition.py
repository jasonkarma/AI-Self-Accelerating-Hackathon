"""Bayesian-style experimental design: priors, shrinkage, and the 5-slot portfolio.

Three things live here, deliberately in one module so the benchmark methods and
the live campaign cannot drift apart:

* :func:`loop_prior_score` — the static WT anchor rule shared by M4 and M5.
* :func:`adaptive_prior_score` — ``Δ_agent``: the council's bounded positional
  priors plus an empirical-Bayes-shrunk SAR term computed over surrogate
  **residuals**, so a co-mutation artefact cannot masquerade as a single-site
  effect.
* :func:`select_five_slots` — a paired H1/H0 experiment design: two exploitation
  picks, one Value-of-Information exploration pick, one hypothesis-testing pick,
  and one *dynamic* H0 / payoff-harvest pick.

Nothing here is calibrated against a particular seed set, and there is no rank
window or "top-N frontier" cutoff: slot 3 and slot 5 are scored by their own
objectives over the *whole* qualified pool.

The exploration -> exploitation transition is not a hand-written round switch.
:func:`value_of_information` combines the remaining horizon ``H`` with the
empirical SAR convergence ``C`` of the qualified frontier. Once the SAR term is
corroborated (in practice by round 3-4), ``VoI`` falls below
:data:`VOI_EXPLOIT_THRESHOLD` and the campaign shifts into all-in exploitation
for the remaining rounds instead of boycotting proven SAR mutations.
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

#: Optimistic-exploration coefficient for slot 3, now scaled by ``VoI`` instead of
#: the old static ``1 - spent/budget``. ``VoI`` already collapses to 0.0 on the
#: final batch, so slot 3 reverts to pure ``A_H1`` exploitation without any
#: hand-written round switch.
SLOT3_OPTIMISM = 0.50

SLOT_COUNT = 5

# ── Expected Value of Information (dynamic exploration -> exploitation) ──────
#: How much corroborated SAR evidence discounts the value of further exploring.
#: At ``sar_maturity == 1.0`` the exploration term keeps only ``1 - 0.65 = 0.35``
#: of the remaining horizon, so a campaign whose frontier is already explained by
#: positive SAR evidence stops paying for falsification probes.
VOI_SAR_WEIGHT = 0.65

#: Decision boundary for slot 5. Above it the batch still needs a strict
#: ``Δ_agent <= 0`` null control (unknown territory, hypotheses unproven); at or
#: below it the campaign is in payoff-harvest mode and slot 5 competes on
#: ``A_null`` across the whole qualified pool instead of boycotting proven SAR.
VOI_EXPLOIT_THRESHOLD = 0.35

#: Stable slot labels used by the API and campaign reconciliation. Slot 5 keeps
#: its historical label in harvest mode: its *allocation rule* changes, while it
#: remains the null-objective side of the paired audit.
SLOT3_LABEL = "Max Joint Uncertainty"
SLOT4_LABEL = "Agent Hypothesis Test"
SLOT5_FALSIFY_LABEL = "Skeptic Falsification Probe"


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


# ── Expected Value of Information ────────────────────────────────────────────
def horizon_factor(
    queries_spent: int,
    *,
    total_budget: int,
    batch_size: int = SLOT_COUNT,
) -> float:
    """Remaining-horizon factor ``H`` in ``[0, 1]``.

    ``H`` asks how many batches are still funded *after the one being designed*::

        future_queries = max(0, total_budget - (queries_spent + batch_size))
        denom          = max(1, total_budget - 2 * batch_size)
        H              = min(1.0, future_queries / denom)

    It is ``1.0`` on the first batch after the seed (one full horizon left) and
    strictly ``0.0`` on the final batch, where nothing after this one can ever be
    measured — so no experiment designed now can inform a later decision. The
    ``denom`` is the number of queries that remain after the first two batches,
    which normalises ``H`` to the actual campaign length instead of hard-coding
    "6 rounds".
    """
    future_queries = max(0, int(total_budget) - (int(queries_spent) + int(batch_size)))
    denom = max(1, int(total_budget) - 2 * int(batch_size))
    return float(min(1.0, future_queries / denom))


def sar_maturity_score(delta_agent) -> float:
    """Empirical SAR convergence ``C`` in ``[0, 1]``.

    ``C`` is how strongly the qualified frontier is *already* corroborated by
    positive SAR evidence::

        C = min(1, max(0, max(Δ_agent)) / (0.5 · AGENT_PRIOR_BOUND))

    Half the adaptive prior bound is the "established" bar: once some unqueried
    qualified candidate sits at half the maximum the prior can contribute, the
    substitution that carries it is no longer a live question — continuing to
    spend a slot falsifying it buys nothing.
    """
    values = np.asarray(list(delta_agent), dtype=float) if len(delta_agent) else np.empty(0)
    values = values[np.isfinite(values)]
    max_pos_delta = max(0.0, float(np.max(values))) if values.size else 0.0
    return float(min(1.0, max_pos_delta / (0.5 * AGENT_PRIOR_BOUND)))


def value_of_information(horizon: float, sar_maturity: float) -> float:
    """``VoI = H · (1 - 0.65 · C)``.

    Horizon says *how long* new information could still matter; ``C`` says *how
    much* is still unresolved. A campaign that has corroborated its SAR term is
    discounted to 35% of its remaining horizon, and a campaign on its final batch
    is worth exactly ``0.0`` regardless of ``C`` — nothing can be re-decided after
    it.
    """
    h = float(np.clip(horizon, 0.0, 1.0))
    c = float(np.clip(sar_maturity, 0.0, 1.0))
    return float(max(0.0, min(1.0, h * (1.0 - VOI_SAR_WEIGHT * c))))


def voi_context(
    delta_agent,
    *,
    queries_spent: int,
    total_budget: int,
    batch_size: int = SLOT_COUNT,
) -> dict:
    """``(H, C, VoI)`` plus the harvest-mode flag, for one batch.

    ``harvest`` is ``VoI <= VOI_EXPLOIT_THRESHOLD`` (rounded to 9 dp so a float
    artefact of exactly 0.35 cannot flip the branch). In harvest mode the batch
    no longer pays for a strict null control: slot 5 competes on ``A_null`` across
    the whole qualified pool.
    """
    horizon = horizon_factor(queries_spent, total_budget=total_budget, batch_size=batch_size)
    maturity = sar_maturity_score(delta_agent)
    voi = value_of_information(horizon, maturity)
    return {
        "horizon": horizon,
        "sar_maturity": maturity,
        "voi": voi,
        "harvest": bool(round(voi, 9) <= VOI_EXPLOIT_THRESHOLD),
    }


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
    batch_size: int = SLOT_COUNT,
) -> list[dict]:
    """Qualified 5-slot portfolio as a paired H1/H0 experiment (SPEC §6).

    Two acquisition scores are defined over the qualified unqueried pool::

        A_null(s) = A_base(s) + loop_prior_score(s)
        A_H1(s)   = A_null(s) + Δ_agent(s)

    ``A_null`` is the shared M4 objective (surrogate + WT anchor rule); ``A_H1``
    is M5's, which adds the council's bounded adaptive prior. One Expected Value
    of Information number, :func:`voi_context`, decides how much the batch is
    still allowed to explore::

        VoI = H · (1 - 0.65 · C)

    The five slots are then:

    1-2  top 2 by ``A_H1`` — core exploitation.
    3    ``A_H1 + 0.50 · VoI · sigma_joint`` maximised over the whole pool. As
         ``VoI -> 0`` (SAR already corroborated, or ``future_queries == 0`` on the
         final batch) the bonus vanishes and slot 3 *is* a pure ``A_H1``
         exploitation slot, without a hand-written round switch. Raw
         ``max sigma_joint`` is deliberately not used: unconstrained, it picks
         the least-informed candidate in the pool regardless of whether that
         candidate could ever win.
    4    top ``A_H1`` among ``Δ_agent > 0`` — the H1 positive arm, i.e. a
         candidate the live hypothesis actually favours. Falls back to the
         unconstrained ``A_H1`` argmax if the prior boosts nobody (round 1).
    5    the dynamic H0 / payoff-harvest arm, on the unmodified null objective:

         * ``VoI > 0.35`` (early, unconverged: the hypotheses still need strict
           falsification) — top ``A_null`` among ``Δ_agent <= 0``.
         * ``VoI <= 0.35`` (SAR corroborated by round 3-4, or the final harvest
           rounds) — top ``A_null`` over the *full* remaining pool, with the
           ``Δ_agent <= 0`` restriction dropped. Refusing the best null-frontier
           candidate merely because it carries a proven SAR substitution is what
           starved the last rounds of payoff.

    Slots 4 and 5 are still chosen on the *same* batch, so every round yields a
    direct H1-vs-H0 comparison for
    :func:`core.tool_arsenal.audit_past_hypotheses` to grade — in harvest mode the
    control is simply allowed to be the better candidate.

    ``queries_spent`` / ``total_budget`` / ``batch_size`` drive ``H``. When a
    caller omits them, ``queries_spent`` is derived from the revealed ledger (the
    only defensible source) and the budget from settings: falling back to
    "0 spent" would leave a late round maximally exploratory.
    """
    if len(unqueried) < SLOT_COUNT:
        raise ValueError("At least five unqueried candidates are required")
    if queries_spent is None or total_budget is None:
        from config.settings import SETTINGS  # local import keeps core importable alone

        if total_budget is None:
            total_budget = SETTINGS.total_budget
        if queries_spent is None:
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

    # ``C`` is measured over the *qualified* pool, i.e. over the candidates the
    # batch could actually spend a slot on, not over optimistically-scored
    # candidates that can never be queried.
    context = voi_context(
        pool.delta_agent.to_numpy(),
        queries_spent=int(queries_spent),
        total_budget=int(total_budget),
        batch_size=int(batch_size),
    )
    pool = pool.assign(
        horizon=context["horizon"],
        sar_maturity=context["sar_maturity"],
        voi=context["voi"],
        optimistic=pool.a_h1 + SLOT3_OPTIMISM * context["voi"] * pool.sigma_joint,
    )

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

    # Slot 3: dynamic-VoI exploration; degenerates to pure A_H1 as VoI -> 0.
    rest = remaining()
    take(best(rest, "optimistic"), SLOT3_LABEL)

    # Slot 4: H1 arm. The planner's declared target can only *tie-break*; it can
    # never force a candidate the acquisition score does not rank.
    rest = remaining()
    supported = rest.loc[rest.delta_agent > 0]
    hypothesis = best(supported, "a_h1") if not supported.empty else best(rest, "a_h1")
    take(hypothesis, SLOT4_LABEL)

    # Slot 5: dynamic H0 control / payoff harvest. The Delta_agent <= 0
    # restriction is a falsification device, not a permanent rule: once VoI says
    # the SAR term is corroborated, boycotting proven mutations is a boycott.
    rest = remaining()
    if context["harvest"]:
        take(best(rest, "a_null"), SLOT5_FALSIFY_LABEL)
    else:
        control = rest.loc[rest.delta_agent <= 0]
        probe = best(control, "a_null") if not control.empty else best(rest, "a_null")
        take(probe, SLOT5_FALSIFY_LABEL)

    picked = [row["candidate_id"] for row in selected]
    if len(set(picked)) != len(picked):
        raise RuntimeError("Slot selection produced duplicate candidates")
    return selected
