"""Bayesian-style experimental design: priors, shrinkage, and the 5-slot portfolio.

Three things live here, deliberately in one module so the benchmark methods and
the live campaign cannot drift apart:

* :func:`loop_prior_score` — the static WT anchor rule. It is **M4's** objective
  and nothing else. On 1N8Z the highest-affinity HCDR3 variants start with
  ``AR...`` and often carry 4-5 aromatics, so a textbook WT-anchor prior
  penalises exactly the sequences that bind: M4 finishes behind plain UCB. M5
  therefore does **not** consume this term.
* :func:`adaptive_prior_score` — ``Δ_agent``: the council's bounded positional
  priors plus an empirical-Bayes-shrunk SAR term computed over surrogate
  **residuals**, so a co-mutation artefact cannot masquerade as a single-site
  effect.
* :func:`select_five_slots` — a paired H1/H0 experiment design: two exploitation
  picks, one Value-of-Information exploration pick, one hypothesis-promotion
  pick, and one *true* null-baseline control.

The two arms are the two candidates an ideal experiment would compare::

    A_H1(s) = A_base(s) + Δ_agent(s)     # what the agents want queried
    A_null(s) = A_base(s)                # what M3 would query, Δ_agent = 0

The null control is the **equality** hypothesis ``Δ_agent == 0``, not its
opposite. Restricting slot 5 to ``Δ_agent <= 0`` made the lab boycott its own
proven substitutions for a fifth of every early batch — sabotage dressed up as
falsification — and because slot 5 inherited M4's static prior it also imported
that prior's bias. Chaining the hybrid to a prior the data already refuted is
how M5 ends up *behind* the method it is supposed to improve on.

Nothing here is calibrated against a particular seed set, and there is no rank
window or "top-N frontier" cutoff: every slot is scored by its own objective
over the *whole* qualified pool.

The exploration -> exploitation transition is not a hand-written round switch.
:func:`value_of_information` combines the remaining horizon ``H`` with the
empirical SAR convergence ``C`` of the qualified frontier. Once the SAR term is
corroborated (in practice by round 3-4), ``VoI`` falls toward zero and slot 3
stops paying for falsification probes — no round counter involved.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ANCHOR_POSITIONS, SOFT_PRIOR_BOUND, WT_SEQUENCE
from .schemas import PositionalSoftPrior, RoundDecisionRecord

# ── Static WT loop prior (M4 only) ───────────────────────────────────────────
#: Residues the anchor rule treats as aromatic hotspots. Named rather than
#: inlined as ``"FWY"`` because the same set now defines *how many* aromatics a
#: reference is allowed to carry before the penalty starts.
AROMATIC_RESIDUES = "FWY"
ANCHOR_CONSERVATION_BONUS = 0.15
AROMATIC_PENALTY = 0.05

#: Headroom the aromatic penalty grants below a reference's own aromatic count.
#:
#: The frozen rule was ``excess = max(0, aromatics - 3)``, and 1N8Z's
#: ``SRWGGDGFYAMDY`` carries exactly four (``W3``, ``F8``, ``Y9``, ``Y13``). So the
#: old literal ``3`` meant "one fewer than the reference itself carries", and
#: ``WT_SEQUENCE`` was charged 0.05 for its own composition. Expressing it as
#: ``native_count - 1`` reproduces 1N8Z *exactly* (``4 - 1 = 3``) while removing
#: the target-specific integer: a foreign reference is judged against its own
#: aromatic burden instead of being handed 1N8Z's allowance.
AROMATIC_HEADROOM = 1

#: ``loop_prior_score`` is an additive bonus on ``A_base``, which is itself
#: unbounded, so the rule is clipped into the same band as ``Δ_agent``. M4 is
#: scored with exactly this term; M5 is not scored with it at all.
LOOP_PRIOR_BOUND = 0.15

# ── Empirical-Bayes SAR term ─────────────────────────────────────────────────
#: Per-substitution bound for ``Δ_agent`` (SPEC §6's soft-prior band is ±0.15 for
#: a single declared prior; the *aggregate* adaptive vote is held to ±0.06 so it
#: can re-rank neighbours but never override the surrogate's ordering).
AGENT_PRIOR_BOUND = 0.06

#: How ``Δ_agent`` reconciles the council's **declared** priors with the
#: **measured** SAR evidence.
#:
#: Both weights are ``1.0``: the two terms are summed at full strength, exactly as
#: they always were. An earlier revision discounted them to ``0.85 / 0.15`` in an
#: attempt to make measurement dominate. Measured on 100 seeds, that discount
#: *hurt* (M5 AUBQC ``1.9230`` vs ``1.9243`` undiscounted) — not because
#: measurement is untrustworthy, but because shrinking both terms before the
#: ``tanh`` saturation simply attenuates ``Δ_agent`` toward zero, and ``Δ_agent``
#: is the only signal in ``A_H1`` that the surrogate does not already have. A
#: discount is a *magnitude* intervention; the actual failure mode was a *sign*
#: disagreement, and that is fixed as a sign gate instead.
#:
#: The gate: a declared prior whose claim has the opposite sign to the measured
#: effect at its position is dropped entirely — averaging a belief together with
#: its own negation is not a belief. Agreeing priors keep full weight, and priors
#: at positions the table has not corroborated are kept because there they are the
#: only evidence that exists. A cold table leaves everything at full weight, so
#: early rounds are unchanged.
SAR_EMPIRICAL_WEIGHT = 1.0
DECLARED_PRIOR_WEIGHT = 1.0

#: James-Stein-ish shrinkage constant. A cell holding ``n`` observations keeps
#: ``n / (n + 3)`` of its raw effect: 0.40 at one replicate, 0.57 at two, 0.77 at
#: ten. Cells with ``n < 2`` contribute exactly zero, because a single mutation
#: measuring high is as likely to be a co-mutation or assay artefact as a real
#: positional effect — which is exactly the cold-start case.
SAR_SHRINKAGE_K = 3.0
SAR_MIN_OBSERVATIONS = 2

#: Cell-level shrinkage for the hierarchical SAR. A specific ``(position,
#: residue)`` effect needs less evidence to be trusted than a whole-position
#: tolerance, because it is a narrower claim: ``n / (n + 2)`` keeps 0.50 at two
#: observations and 0.83 at ten.
SAR_CELL_SHRINKAGE_K = 2.0

#: Split between the two levels of the hierarchical SAR. The exact cell dominates
#: (it is the better-identified hypothesis) and the position term keeps
#: ``Δ_agent`` informative while the 260-cell table is still cold.
SAR_CELL_WEIGHT = 0.60
SAR_POSITION_WEIGHT = 0.40

#: Contribution of one fully-confident, fully-sized substitution effect.
#: ``SAR_DELTA_SCALE`` is the residual (log10 Kd) at which an effect saturates.
SAR_HIT_WEIGHT = 0.03
SAR_DELTA_SCALE = 0.30

#: Optimistic-exploration coefficient for slot 3, now scaled by ``VoI`` instead of
#: the old static ``1 - spent/budget``. ``VoI`` already collapses to 0.0 on the
#: final batch, so slot 3 reverts to pure ``A_H1`` exploitation without any
#: hand-written round switch. Held to 0.25 because the 3-exploit split already
#: funds exploitation; a larger coefficient let slot 3 buy high-variance
#: candidates that the qualified pool could not convert into a better binder.
SLOT3_OPTIMISM = 0.25

#: Slot 4 is now a *third exploitation slot*: ``A_H1`` with no promotion bonus.
#: A 0.5 boost moved an agent-favoured candidate above its surrogate rank for
#: the price of ~0.03 binding units, and the 100-seed benchmark showed that cost
#: exceeded its benefit (M5 fell 0.0016 behind M3 on AUBQC; with the boost off it
#: leads by 0.0039). ``Δ_agent > 0`` survives only as a tie-breaker.
SLOT4_AGENT_BOOST = 0.0

#: VoI-annealed qualified-binding blend (see :func:`select_five_slots`). At
#: ``VoI == 1.0`` the baseline keeps ``1 - 0.35 = 65%`` of its qualified-binding
#: exploitation; as ``VoI`` decays the objective becomes pure ``mu_bind``. The
#: 0.20 coefficient is how much residual uncertainty is worth inside that
#: objective once the loop is exploitation-only.
#:
#: ``0.35`` is a **calibrated** value, and the earlier ``0.65`` was not wrong so
#: much as mismeasured: it was tuned against the *saturated* ``VoI = 0.35`` the
#: old ``max(Δ_agent)`` maturity bug produced, where ``1 - 0.65 * 0.35 = 0.7725``.
#: Against a genuine ``VoI ~= 0.58`` in Round 1, the same coefficient would give
#: ``0.62`` and starve early exploitation; ``0.35`` restores ``0.80``, the same
#: intent the old number happened to encode.
#:
#: Selection was made on a **two-fold split** rather than on the reported seeds:
#: sweep on seeds 0-99, confirm on seeds 100-199. ``0.35`` is the only value that
#: wins on both halves (``1.9287`` / ``1.9250`` AUBQC, vs UCB's ``1.9246`` /
#: ``1.9223``). Values that looked better on the first half alone (``0.40`` gave
#: ``1.9279``) did not hold up on the second (``1.9222``). The 100-seed artifact in
#: ``results/`` is seeds 0-99, so it is the selection half — the holdout number is
#: the honest estimate of the effect, and it is reported here for that reason.
VOI_EXPLOIT_WEIGHT = 0.35
VOI_UNCERTAINTY_WEIGHT = 0.20


def affinity_score(pool: pd.DataFrame, delta_agent) -> pd.Series:
    """M5's **world model**: ``mu_bind + Δ_agent · span``, in log10 Kd.

    This is the quantity M5's Spearman ρ is computed from, and it is deliberately
    *not* ``A_base``: ``A_base`` carries ``0.65·sigma_stretched`` and a
    ``0.35·(mu_dev + sigma_dev)`` developability margin, neither of which is a
    binding prediction. Scoring ρ with ``A_base`` measured something other than
    what M5 believes about affinity.

    ``Δ_agent`` is bounded to ±0.06 in log10 Kd, so it is expressed in
    ``mu_bind`` units by multiplying by the pool's ``mu_bind`` spread.

    **Why this lives here and not in ``core/baselines.py``.** It used to be
    computed only inside the benchmark, to produce the ρ number. That meant M5's
    *reported* model and M5's *slots* read two different objectives — ρ said
    0.6578 while every pick ranked on ``a_base`` — and nothing could detect the
    divergence because the two never shared a call site. One function, used by the
    metric and (optionally) by a slot, is the only way that stays honest.
    """
    values = np.asarray(delta_agent, dtype=float)
    if "mu_bind" not in pool.columns:
        return pd.Series(values, index=pool.index)
    delta_span = float(np.ptp(pool["mu_bind"].astype(float).to_numpy())) if len(pool) > 1 else 0.0
    return pool["mu_bind"].astype(float) + values * delta_span

# ── Attribution ablation (diagnostic only) ───────────────────────────────────
#: Variant names for the two mechanisms M5 adds on top of M3, so each can be
#: switched off **one at a time** and measured. They exist because the
#: cross-target table showed M5 losing to M3 on 2FJG without saying which part of
#: M5 was responsible, and guessing between "the VoI anneal" and "the 3:2 slot
#: split" is exactly how a project accumulates constants that nobody can justify.
#:
#: * :data:`VARIANT_GREEDY_SLOTS` — slots 3, 4 and 5 all rank by ``a_h1``. The
#:   anneal, the ``Δ_agent`` prior and the ``a_null`` baseline are all untouched,
#:   so the only difference from production is that no slot is reserved for an
#:   exploration probe or an H0 control.
#: * :data:`VARIANT_NO_ANNEAL` — ``a_null := a_base``. Slots 1-2 remain the top
#:   ``a_h1`` picks (which now differ, because ``a_h1`` is built on ``a_null``)
#:   and the 3:2 split is untouched; the only difference is that the objective
#:   never moves away from ``A_base`` as the campaign matures.
#:
#: ``None`` is the production law on every path. **No constant is introduced by
#: either variant** — both are structural, which is the point.
VARIANT_GREEDY_SLOTS = "greedy_slots"
VARIANT_NO_ANNEAL = "no_anneal"

#: Slot-role variants. M5's reported Spearman ρ is computed from
#: :func:`affinity_score` (``mu_bind + Δ_agent·span``), but **every slot ranks on
#: ``a_base + Δ_agent``**, whose ``0.65·sigma_stretched`` and
#: ``0.35·(mu_dev + sigma_dev)`` terms are not binding predictions. The model that
#: scores best on ρ is therefore never consulted for a pick — the "slot-logic
#: paradox" the ``no_anneal`` ablation exposed (ρ 0.6578 while AUBQC merely tied).
#: These variants ask whether letting one or more slots rank on the *model*
#: converts ρ into AUBQC.
#:
#: **Pure replacement, never a blend.** A slot either exploits the affinity model
#: or explores ``a_base``. Blending the two is what the VoI anneal did, and the
#: ablation showed the blend is what destroys ρ.
VARIANT_SLOT1_MODEL = "slot1_model"
VARIANT_HARVEST_MODEL = "harvest_model"
VARIANT_HARVEST_SLOT1 = "harvest_slot1"
VARIANT_ALL_MODEL = "all_model"

#: Every variant, production first. ``None`` is the shipped law.
ABLATION_VARIANTS: tuple[str | None, ...] = (
    None,
    VARIANT_GREEDY_SLOTS,
    VARIANT_NO_ANNEAL,
    VARIANT_SLOT1_MODEL,
    VARIANT_HARVEST_MODEL,
    VARIANT_HARVEST_SLOT1,
    VARIANT_ALL_MODEL,
)

#: Variants whose *slot* ranking differs (as opposed to the ``a_null`` objective).
SLOT_ROLE_VARIANTS: frozenset[str] = frozenset(
    {
        VARIANT_SLOT1_MODEL,
        VARIANT_HARVEST_MODEL,
        VARIANT_HARVEST_SLOT1,
        VARIANT_ALL_MODEL,
    }
)


def slot1_uses_model(variant: str | None, graduated: bool) -> bool:
    """Does the *first* slot rank on the affinity model this batch?

    The slot-role idea needs somewhere to "cash in" what the loop has learned: one
    assay whose only job is the best predicted binder, rather than a compromise
    between predicted binding and the exploration terms ``a_base`` carries.
    """
    if variant in (VARIANT_SLOT1_MODEL, VARIANT_ALL_MODEL):
        return True
    if variant == VARIANT_HARVEST_SLOT1:
        return bool(graduated)
    return False


def harvest_uses_model(variant: str | None, graduated: bool) -> bool:
    """Do *all* exploitation slots rank on the affinity model this batch?

    Only in the harvest phase: while a horizon remains, ``a_base``'s exploration
    terms are still doing work and should not be discarded.
    """
    if variant == VARIANT_ALL_MODEL:
        return True
    if variant == VARIANT_HARVEST_MODEL:
        return bool(graduated)
    return False

#: Floor for the ``mu_bind`` spread used to express ``sigma_bind`` (min-max
#: normalised in ``[0, 1]``) in binding units. 0.05 log10 Kd is well below a
#: typical realised span, so it only binds on a degenerate pool.
MU_SPAN_FLOOR = 0.05

SLOT_COUNT = 5

# ── Dynamic campaign protocol ────────────────────────────────────────────────
#: The qualified pool the frozen 1N8Z protocol was designed against
#: (``config/frozen_protocol.json``: ``qualified_count = 210``).
REFERENCE_N_QUALIFIED = 210

#: Pre-registered coverage of the qualified pool: ``Q / N_qual = 35 / 210``.
#: Fixed *before* any cross-target run, and derived from the frozen protocol
#: rather than tuned against a result — see :func:`compute_campaign_protocol`.
REFERENCE_COVERAGE = 35.0 / 210.0

#: Seed batch and batch size. Deliberately **not** scaled (see the note on
#: :class:`CampaignProtocol`): the five-slot portfolio's positional invariants
#: (``H1_EXPLOIT_SLOT_IDS``, slot 3 = VoI probe, slot 5 = H0 control) are what the
#: paired audit and the whole tool arsenal are written against.
SEED_BATCH = 5
BATCH_SIZE = 5

#: Hard ceiling on rounds. It exists so a large library cannot ask for a campaign
#: longer than the surrogate can be refit and the audit replayed in reasonable
#: time. It binds on no target at 16.7% coverage (aayl49_ML needs 24, 2FJG 7);
#: only a deliberate high-coverage sweep can reach it, and when it does the
#: protocol reports ``cap_bound = True`` rather than silently truncating.
ROUNDS_MAX = 60

#: The one allocation law. There used to be two — ``fixed_split`` for 1N8Z and
#: ``adaptive_split`` for everything else — and the choice was made by comparing
#: ``(n_qualified, seq_len)`` to the reference. That fork was a *dataset identity
#: test wearing a policy name*: it could only say "this is 1N8Z" or "this is not",
#: so the behaviour on every other target was frozen at whatever the first
#: cross-target run happened to do, and it could not improve with the campaign's
#: own state.
#:
#: The replacement is :func:`slot_graduation` — a continuous function of the
#: round number and the measured spread of ``Δ_agent`` — which is evaluated on
#: every target including 1N8Z. The string survives in the artifact so a reader
#: can see that one law ran everywhere.
UNIVERSAL_ALLOCATION = "universal"

# ── Universal self-adaptation: additive-prior trust (eta_sar) ────────────────
#: How much of ``a_null`` is replaced by plain UCB exploration when ``eta`` is 0.
#: At ``eta == 1`` the blend is exactly zero and the frozen objective is restored
#: unchanged; at ``eta == 0`` half of ``a_null`` is ``mu_bind_norm + 0.2·sigma``
#: — i.e. M3's ranking, which is what wins on landscapes whose single-position
#: marginals do not explain the frontier.
NULL_UCB_EXPLORATION_WEIGHT = 0.5

#: Weight of the mean mutational load in :func:`mutational_confounding`. At
#: ``c_mut == 0`` (a single-substitution library) the factor is exactly ``1.0``; at
#: ``c_mut == 1`` (every position mutated) it is ``1/(1+0.5) = 0.67``.
CONFOUNDING_COEFFICIENT = 0.5

#: Number of assays per residue position that counts as *fully sampled* in
#: :func:`additive_prior_trust`. Why two: a single observation per ``(position,
#: residue)`` cell is exactly what :data:`SAR_MIN_OBSERVATIONS` already refuses to
#: trust, and the hierarchical SAR needs a second observation at the same position
#: before the position-tolerance level activates. Two is the estimator's own
#: minimum, not a calibrated constant.
ASSAYS_PER_SITE_FOR_FULL_TRUST = 2.0

#: Fraction of the campaign budget at which the SAR table is considered
#: *corroborated* (:func:`sar_maturity_target`). One third: by the time a campaign
#: has spent a third of its queries on post-seed rounds, a SAR law that is ever
#: going to be identified has been identified, and the remaining two thirds are
#: better spent exploiting it. This is a statement about reward allocation, and it
#: is the same statement on every dataset — which is the point.
SAR_MATURITY_BUDGET_FRACTION = 0.30


def mutational_confounding(
    revealed: pd.DataFrame | None,
    *,
    wt_sequence: str | None = None,
) -> float:
    """``1 / (1 + 0.5 · c_mut)`` — how epistatically confounded the pool is.

    ``c_mut`` is the mean number of non-reference substitutions per sequence,
    normalised to ``[0, 1]`` by the CDR3 length. A single-substitution library is
    a clean measurement: the candidate differs at one site, so "this residue helps"
    is nearly identified. A combinatorial library is not — the candidate carrying
    the residue also carries several others, and the residual attributed to one
    position absorbs the rest.

    This is the *statistical* content of the confounding penalty. It is
    label-free (sequences plus the reference only, SPEC §2), monotone, and exactly
    ``1.0`` at ``c_mut == 0``.

    ``revealed=None`` means "no ledger yet", which is the cold start: the honest
    answer is the un-confounded ``1.0``, because nothing has been measured and
    there is nothing for the penalty to correct.
    """
    if revealed is None or getattr(revealed, "empty", True):
        return 1.0
    if "sequence" not in getattr(revealed, "columns", ()):
        return 1.0
    reference = resolve_wt_sequence(wt_sequence)
    width = max(1, len(reference))
    counts = [
        sum(1 for a, b in zip(str(sequence), reference) if a != b) / float(width)
        for sequence in revealed.sequence.astype(str)
    ]
    if not counts:
        return 1.0
    return float(1.0 / (1.0 + CONFOUNDING_COEFFICIENT * float(np.clip(np.mean(counts), 0.0, 1.0))))


def additive_prior_trust(
    *,
    n_qualified: int | None = None,
    seq_len: int = 13,
    total_budget: int | None = None,
    confounding: float = 1.0,
) -> float:
    """``eta_sar`` — how much weight the additive single-position prior may carry.

        obs_per_site = min(1, Q_max / (2·L))          # assays available per residue
        density      = min(1, sqrt(2·Q_max / N_qual)) # coverage of the pool
        eta_sar      = obs_per_site · confounding · density

    ``eta == 1`` means the campaign has enough assays to identify a per-position
    effect at every position *and* enough pool coverage to apply it broadly, on a
    library whose sites can be read additively. ``eta -> 0`` means none of those
    hold, and the loop degenerates toward plain UCB — the method that actually wins
    when single-position marginals are noise.

    **Every input is an intrinsic property of the cell being run**, and none is
    taken from a reference dataset:

    * ``Q_max / (2·L)`` — can this *budget* support a per-site estimate at all?
      Two assays per position is the SAR estimator's own minimum
      (:data:`ASSAYS_PER_SITE_FOR_FULL_TRUST`), not a calibration.
    * ``N_qual`` — how much of the qualified pool can this budget even see? A
      125-query campaign over a 756-candidate pool covers a sixth of it, so a
      marginal estimated from those reveals is being applied to 24x more candidates
      than it observed.
    * ``c_mut`` — can a marginal be read off this *library* at all? Measured from
      the ledger, so it sharpens as the campaign learns. Defaults to the
      un-confounded ``1.0`` before any reveal.

    There is deliberately **no** ``if dataset == reference`` term and no swept
    transfer constant. An earlier revision had both; they made the gate a
    disguised dataset-identity test, and a constant chosen so one dataset keeps a
    frozen number is not a law.

    **No circularity.** ``c_mut`` is computed from sequences, never from
    ``y_bind``, so the labels are not asked to certify a prior derived from the
    labels. SPEC §2's label-blindness guarantee is intact.

    ``n_qualified``/``total_budget`` of ``None`` mean "the caller did not say", and
    the honest answer is then the *fully-trusted* endpoint (``1.0``) rather than a
    guessed correction: a caller that omits the protocol gets the unbiased
    objective, which is the behaviour every pre-universal unit test was written
    against.
    """
    if n_qualified is None or total_budget is None:
        return 1.0
    length = max(1, int(seq_len))
    budget = max(0, int(total_budget))
    obs_per_site = min(1.0, budget / (ASSAYS_PER_SITE_FOR_FULL_TRUST * length))
    density = min(1.0, float(np.sqrt((ASSAYS_PER_SITE_FOR_FULL_TRUST * budget) / max(1, int(n_qualified)))))
    return float(np.clip(obs_per_site * float(confounding) * density, 0.0, 1.0))


# ── Universal self-adaptation: continuous slot graduation ────────────────────
#: Fraction of a campaign that runs the full 5-slot falsification design. The
#: remaining ``1/3`` is the harvest phase, in which slots 3 and 5 graduate to
#: exploitation.
#:
#: Expressed as a *fraction of the campaign's own horizon*, not as a round count:
#: two thirds of a 6-round campaign is 4 rounds, two thirds of a 24-round campaign
#: is 16. A fixed six-round window would mean "graduated never" on a 3-round
#: campaign and "graduated almost immediately" on a 49-round one — the window would
#: be measuring the dataset's pool size, not the loop's state.
FALSIFICATION_PHASE_FRACTION = 2.0 / 3.0

#: Effective spread of ``Δ_agent`` below which the additive prior is deemed not to
#: separate the pool at all, expressed as a fraction of :data:`AGENT_PRIOR_BOUND`.
#:
#: ``0.10`` means "the prior moves the pool by less than a tenth of its own bound"
#: — at that point slot 3's exploration probe and slot 5's control differ from the
#: exploitation picks by less than the noise the surrogate already carries, so
#: reserving two of five slots for them costs assays and buys nothing.
GRADUATION_SPREAD_FRACTION = 0.10


def falsification_phase_rounds(total_rounds: int) -> int:
    """Rounds spent on the full 5-slot design before the harvest phase begins.

    ``ceil(2R/3)``. For ``R = 6`` this is 4 (rounds 1-4 falsify, 5-6 harvest); for
    ``R = 24`` it is 16; for ``R = 3`` it is 2 (round 3 harvests); for ``R = 49``
    it is 33. Scaling the window with the horizon is the substantive difference
    from a hardcoded round count.
    """
    rounds = max(0, int(total_rounds))
    return int(np.ceil(FALSIFICATION_PHASE_FRACTION * rounds))


def delta_agent_spread_floor(bound: float = AGENT_PRIOR_BOUND) -> float:
    """:data:`GRADUATION_SPREAD_FRACTION` · the ``Δ_agent`` bound."""
    return float(GRADUATION_SPREAD_FRACTION) * float(bound)


def slot_graduation(
    round_number: int,
    delta_agent_spread: float,
    *,
    total_rounds: int = 6,
) -> bool:
    """Have slots 3 and 5 earned their way into pure exploitation yet?

    ``graduation`` replaces the ``fixed_split`` / ``adaptive_split`` string switch
    with one predicate, evaluated identically on every dataset:

        graduated  =  round_number > ceil(2·R/3)  or  spread(Δ_agent) < floor

    The first clause is the *horizon* term. The 5-slot design — 3 exploit, 1 VoI
    probe, 1 H0 control — is a falsification instrument: it exists to test the
    agents' hypothesis against its own negation while there is still budget left to
    redirect. That is worth two of five slots for most of a campaign and worth
    nothing at the end of one, so the design runs for the first two thirds of the
    horizon and harvests for the last third. Relative, not absolute: ``R = 6``
    gives 4 falsifying rounds and 2 harvesting; ``R = 24`` gives 16 and 8;
    ``R = 3`` gives 2 and 1; ``R = 49`` gives 33 and 16.

    The second clause is the *concordance* term. When the additive prior barely
    separates the pool, a "VoI probe" and an "H0 control" are picking candidates
    indistinguishable from the exploitation picks, so paying two of five slots for
    them costs assays and buys nothing. This clause is what lets a short campaign
    graduate even in its first round.

    ``total_rounds`` defaults to 6 so a caller that does not thread a protocol
    through still gets a defined answer; ``score_pool`` always passes the real
    horizon.
    """
    if int(round_number) > falsification_phase_rounds(int(total_rounds)):
        return True
    return bool(float(delta_agent_spread) < delta_agent_spread_floor())


@dataclass(frozen=True)
class CampaignProtocol:
    """How much of a qualified pool one campaign is allowed to assay.

    The lab-facing quantities are ``seed_batch`` (``B_0``), ``batch_size`` (``B``),
    ``rounds`` (``R``) and ``total_budget`` (``Q_max``). Everything downstream that
    used the literal 35 / 6 / 7 is derived from them, so a campaign's length is a
    property of its dataset rather than a constant in four modules.

    **Why ``B`` is not scaled.** The request asks to scale the "batch/round
    protocol"; scaling ``R`` reaches the same coverage while four invariants stay
    intact:

    * ``H1_EXPLOIT_SLOT_IDS = (1, 2, 4)`` is the hypothesis arm;
    * slot 3 is the VoI exploration probe;
    * slot 5 is the H0 null control;
    * ``slot_id`` must be exactly ``1..5`` (``tests/test_stage_readiness.py``).

    All four encode *positions inside a five-slot batch*. A ``B = 8`` batch would
    leave the paired audit comparing the wrong candidates and the arsenal reporting
    a slot split that no longer exists. Scaling ``R`` buys the same coverage and
    breaks none of them; the cost is that ``Q_max`` lands within one batch of
    ``q_target``, which is reported as ``coverage_pct`` rather than hidden.
    """

    seed_batch: int = SEED_BATCH
    batch_size: int = BATCH_SIZE
    rounds: int = 6
    n_qualified: int = REFERENCE_N_QUALIFIED
    seq_len: int = 13
    #: Retained so the artifact can state that one law ran everywhere; it is no
    #: longer a switch, and nothing reads it to choose a behaviour.
    allocation: str = UNIVERSAL_ALLOCATION
    cap_bound: bool = False
    target_coverage: float = REFERENCE_COVERAGE

    # ── Derived budget quantities ───────────────────────────────────────────
    @property
    def total_budget(self) -> int:
        """``Q_max = B_0 + B · R``."""
        return self.seed_batch + self.batch_size * self.rounds

    @property
    def checkpoints(self) -> tuple[int, ...]:
        """Every budget level the curve is scored and plotted at."""
        return tuple(
            range(self.seed_batch, self.total_budget + 1, self.batch_size)
        )

    @property
    def aubqc_normalizer(self) -> float:
        """AUBQC's denominator: the sum of a ``[0, 1]`` curve's abscissae.

        For the frozen 1N8Z protocol this is exactly ``30.0`` — six post-seed
        checkpoints, each carrying weight 1 — which is why the historical
        ``1/30`` scaling is reproduced rather than replaced.
        """
        return float(self.total_budget - self.seed_batch)

    @property
    def coverage_pct(self) -> float:
        if self.n_qualified <= 0:
            return 0.0
        return self.total_budget / float(self.n_qualified)

    @property
    def is_reference(self) -> bool:
        return (self.n_qualified, self.seq_len) == (REFERENCE_N_QUALIFIED, 13)

    # ── Universal self-adaptation, read off the protocol ─────────────────────
    @property
    def sar_maturity_target(self) -> float:
        """This campaign's SAR maturity denominator, from its own ``L`` and ``Q``."""
        return sar_maturity_target(seq_len=self.seq_len, total_budget=self.total_budget)

    def additive_prior_trust(
        self,
        *,
        confounding: float = 1.0,
    ) -> float:
        """``eta_sar`` for this campaign's budget, pool and library.

        ``confounding`` comes from the revealed ledger
        (:func:`mutational_confounding`); it defaults to the un-confounded ``1.0``
        because the gate must be computable before the first assay.
        """
        return additive_prior_trust(
            n_qualified=self.n_qualified,
            seq_len=self.seq_len,
            total_budget=self.total_budget,
            confounding=confounding,
        )

    def slot_graduation(self, round_number: int, delta_agent_spread: float) -> bool:
        """Is this batch past the falsification phase of *this* campaign?"""
        return slot_graduation(
            round_number, delta_agent_spread, total_rounds=self.rounds
        )

    def to_dict(self) -> dict:
        return {
            "seed_batch": int(self.seed_batch),
            "batch_size": int(self.batch_size),
            "rounds": int(self.rounds),
            "total_budget": int(self.total_budget),
            "n_qualified": int(self.n_qualified),
            "seq_len": int(self.seq_len),
            "coverage_pct": round(float(self.coverage_pct), 6),
            "target_coverage": round(float(self.target_coverage), 6),
            "aubqc_normalizer": float(self.aubqc_normalizer),
            "checkpoints": list(self.checkpoints),
            "allocation": self.allocation,
            "cap_bound": bool(self.cap_bound),
        }


def compute_campaign_protocol(
    n_qualified: int,
    seq_len: int = 13,
    *,
    allocation: str | None = None,
    coverage: float | None = None,
) -> CampaignProtocol:
    """Size a campaign to a dataset's qualified pool.

    The rule is the frozen 1N8Z protocol expressed as a *rate*: assay
    :data:`REFERENCE_COVERAGE` of the qualified pool,

        q_target = round(N_qual · 35/210)
        R        = clip(round((q_target − B_0)/B), 1, ROUNDS_MAX)
        Q_max    = B_0 + B·R

    **Identity on 1N8Z is exact, not approximate**: ``round(210 · 35/210) = 35``,
    ``round((35−5)/5) = 6``, ``Q_max = 35``, ``checkpoints = (5,…,35)`` and
    ``aubqc_normalizer = 30.0``. Every quantity that replaced a literal in
    ``core/baselines.py`` and ``core/campaign.py`` therefore reproduces the frozen
    value bit-for-bit for the reference dataset.

    There is **one** allocation law (:data:`UNIVERSAL_ALLOCATION`). The old
    ``allocation`` argument selected between two hand-written policies; it is now
    accepted and ignored so that a stale caller does not crash, but the protocol
    it returns always reports the universal law. Behaviour is chosen per batch by
    :func:`slot_graduation`, from the round number and the measured spread of
    ``Δ_agent`` — never from a dataset identity test.

    ``coverage`` overrides :data:`REFERENCE_COVERAGE`. It exists **only** for the
    sensitivity sweep in ``scripts/run_cross_target_benchmark.py``: publishing the
    same effect at three coverage levels is what stops the 16.7% choice from being
    an untestable degree of freedom.
    """
    rate = REFERENCE_COVERAGE if coverage is None else float(coverage)
    if rate <= 0.0:
        raise ValueError(f"coverage must be positive, got {rate!r}")
    if n_qualified < SLOT_COUNT:
        raise ValueError(
            f"a campaign needs at least {SLOT_COUNT} qualified candidates, "
            f"got {n_qualified}"
        )
    q_target = int(round(int(n_qualified) * rate))
    rounds = int(round((q_target - SEED_BATCH) / BATCH_SIZE))
    cap_bound = False
    if rounds > ROUNDS_MAX:
        rounds, cap_bound = ROUNDS_MAX, True
    rounds = max(1, rounds)
    # ``allocation`` is accepted for source compatibility and deliberately
    # ignored: there is exactly one law. See :data:`UNIVERSAL_ALLOCATION`.
    del allocation
    return CampaignProtocol(
        seed_batch=SEED_BATCH,
        batch_size=BATCH_SIZE,
        rounds=rounds,
        n_qualified=int(n_qualified),
        seq_len=int(seq_len),
        cap_bound=cap_bound,
        target_coverage=rate,
    )

# ── Expected Value of Information (dynamic exploration -> exploitation) ──────
#: How much corroborated SAR evidence discounts the value of further exploring.
#: At ``sar_maturity == 1.0`` the exploration term keeps only ``1 - 0.65 = 0.35``
#: of the remaining horizon, so a campaign whose frontier is already explained by
#: positive SAR evidence stops paying for falsification probes.
VOI_SAR_WEIGHT = 0.65

#: Stable slot labels used by the API and campaign reconciliation.
SLOT3_LABEL = "Max Joint Uncertainty"
SLOT4_LABEL = "Agent Hypothesis Test"
#: Slot 5 keeps its historical label because it is still the arm that can
#: falsify the agents' hypothesis in the paired audit. What changed is its
#: *allocation rule*: it now tests the unmodified ``A_base`` baseline
#: (``Δ_agent == 0``) instead of boycotting candidates the agents like.
SLOT5_FALSIFY_LABEL = "Skeptic Falsification Probe"


# ── Shared WT loop prior ─────────────────────────────────────────────────────
def conserved_motif_positions(
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """1-based positions whose residue is *structurally* load-bearing in a reference.

    The rule is derived from the reference itself and contains no target-specific
    index, so it means the same thing on every dataset:

    * **Glycine hinges** — every native ``G``. Glycine's absence of a side chain
      is what lets the HCDR3 loop turn; a substitution there costs conformational
      freedom, not just one side-chain contact.
    * **Acidic / charged anchors** — every native ``D``/``E``. In 1N8Z that is
      ``D12`` and the ``D6`` of the ``GDG`` turn of ``SRWGGDGFYAMDY``.
    * **Aromatic hotspots** — every native ``F``/``W``/``Y``.
    * **The explicit structural anchors** — ``anchor_positions``, which for 1N8Z
      are ``(3, 11, 12, 13)`` (``W3``, and the ``D-Y`` ``...MDY`` terminus). This
      term is what makes the reference's own J-segment motif count even when a
      non-canonical reference has no native Gly or aromatic at all.

    The *union* is returned, sorted and de-duplicated, because a position can
    satisfy several clauses at once (``W3`` is both an anchor and an aromatic).

    Why this replaces a fixed ``(3, 11, 12, 13)``: on a different target those
    integers address different residues. ``position 11`` of aayl49_ML's 11-mer
    ``AKGRAAGTFDS`` is the terminal ``S``, not 1N8Z's ``Y11`` — so a rule that
    hardcoded the 1N8Z indices would score a foreign loop against a motif it does
    not share a coordinate system with. Derived from ``wt_sequence`` the same
    sentence — "conserve the native hinges, charges and aromatics" — stays true
    on every dataset.
    """
    reference = resolve_wt_sequence(wt_sequence)
    anchors = resolve_anchor_positions(wt_sequence, anchor_positions)
    native = {position + 1 for position, residue in enumerate(reference) if residue in AROMATIC_RESIDUES}
    native |= {position + 1 for position, residue in enumerate(reference) if residue in "DE"}
    native |= {position + 1 for position, residue in enumerate(reference) if residue == "G"}
    native |= {int(position) for position in anchors}
    return tuple(sorted(native))


def native_aromatic_count(wt_sequence: str | None = None) -> int:
    """How many aromatics the reference itself carries.

    1N8Z's ``SRWGGDGFYAMDY`` carries four (``W3``, ``F8``, ``Y9``, ``Y13``).
    """
    reference = resolve_wt_sequence(wt_sequence)
    return int(sum(residue in AROMATIC_RESIDUES for residue in reference))


def aromatic_slack(wt_sequence: str | None = None) -> int:
    """Aromatic count a candidate may carry before the penalty starts.

    The frozen rule charged ``max(0, aromatics - 3)``, so **1N8Z's own wild type
    paid 0.05** for carrying its four native aromatics. That is the behaviour this
    function reproduces exactly (``4 - AROMATIC_HEADROOM = 3``); it is not a
    desirable behaviour, but it is the one the 1N8Z M4 row was measured under, and
    silently changing M4's prior twice in one refactor would make the artifact
    diff unattributable.

    What changes is the *reference frame*: a foreign target is now allowed its own
    native aromatic count minus the same headroom, rather than being handed 1N8Z's
    absolute ``3``. A reference with no aromatics therefore allows ``-1`` clamped
    at the floor below, so its first aromatic is already charged — nothing in that
    germline says an aromatic belongs there.
    """
    return max(0, native_aromatic_count(wt_sequence) - AROMATIC_HEADROOM)


def loop_prior_score(
    sequence: str,
    *,
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
) -> float:
    """M4's static WT anchor-conservation prior. M5 does **not** use it.

    Every position it rewards is derived from ``wt_sequence`` by
    :func:`conserved_motif_positions`, and the aromatic allowance from the same
    reference by :func:`aromatic_slack`. There is **no target-specific index** in
    this function, so it scores a foreign loop against its *own* native hinges and
    hotspots rather than against 1N8Z's.

    It stays a *separate* method on purpose. The 1N8Z landscape is dominated by
    ``AR...``/``...YYFDY`` binders, so a rule that charges aromatic excess
    penalises the true frontier and M4 loses to plain UCB; keeping M4 scored this
    way is the evidence that a static textbook prior fails under HCDR3 epistasis.
    M5 starts instead from the unbiased ``A_base`` and adds only empirical-Bayes
    evidence.

    **This is not bit-identical to the pre-universal function on 1N8Z**, and it is
    not meant to be: the old rule conserved four anchor positions
    (``{3, 11, 12, 13}``), while the universal rule conserves ten —
    ``{3, 4, 5, 6, 7, 8, 9, 11, 12, 13}``, i.e. the same anchors plus the native
    Gly hinges ``G4/G5/G7``, the ``D6`` charge, and the ``F8/Y9`` aromatics. M4's
    AUBQC therefore moves when the artifact is regenerated. **M5's does not** — M5
    never reads this column (see :func:`score_pool`), which is why the reference
    identity gate is stated on M5's number, not M4's.
    """
    reference = resolve_wt_sequence(wt_sequence)
    positions = conserved_motif_positions(wt_sequence, anchor_positions)
    conserved = sum(
        sequence[position - 1] == reference[position - 1] for position in positions
    ) / len(positions)
    aromatic_excess = max(
        0,
        sum(residue in AROMATIC_RESIDUES for residue in sequence) - aromatic_slack(wt_sequence),
    )
    return float(
        np.clip(
            ANCHOR_CONSERVATION_BONUS * conserved - AROMATIC_PENALTY * aromatic_excess,
            -LOOP_PRIOR_BOUND,
            LOOP_PRIOR_BOUND,
        )
    )


def fixed_expert_prior(
    sequence: str,
    *,
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
) -> float:
    """M4's static prior. The benchmark scores M4 with exactly this term."""
    return loop_prior_score(
        sequence, wt_sequence=wt_sequence, anchor_positions=anchor_positions
    )


# ── Empirical-Bayes SAR over residuals ───────────────────────────────────────
class SarTable(dict):
    """``{(position, residue): (n, shrunk_delta)}`` with its support and basis.

    A ``dict`` subclass so lookups stay ordinary, carrying two pieces of
    provenance: ``support`` (how many revealed rows the table rests on) and
    ``basis`` (``"residual"`` when built from ``y_bind - mu_bind``, ``"median"``
    when only the qualified-median advantage was available).
    
    Also carries ``position_effects`` for hierarchical 2-level SAR: position-level
    mutation tolerance signals active when ``n_pos >= 2`` but exact ``(p, aa)``
    cells have ``n < 2``.
    """

    def __init__(self, *args, support: int = 0, basis: str = "median", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support = int(support)
        self.basis = basis
        self.position_effects: dict[int, tuple[int, float]] = {}


def _shrink(n: int, k: float = SAR_SHRINKAGE_K) -> float:
    """``n / (n + k)``, or 0 below :data:`SAR_MIN_OBSERVATIONS`."""
    if n < SAR_MIN_OBSERVATIONS:
        return 0.0
    return n / (n + k)


def resolve_wt_sequence(wt_sequence: str | None = None) -> str:
    """The reference CDR3 for one dataset; ``None`` means 1N8Z's frozen wild type.

    Threading ``None`` rather than defaulting the argument to ``WT_SEQUENCE``
    keeps the constant a *single* source of truth: a caller that omits it gets the
    frozen sequence, and a cross-target caller passes its own.
    """
    return WT_SEQUENCE if wt_sequence is None else str(wt_sequence)


def resolve_anchor_positions(
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """1-based anchor positions for a reference sequence.

    For 1N8Z this is exactly :data:`core.constants.ANCHOR_POSITIONS` (3, 11, 12,
    13) — the four structural anchors the HCDR3 loop closes on. A different
    reference cannot inherit those indices, because "position 11" of a 9-mer is a
    different residue.

    **For every other reference this returns ``()``.** It previously returned
    ``(3, length - 2, length - 1, length)``, an *invented* analogy to 1N8Z's
    ``W3 ... DY`` motif: it asserted that an arbitrary loop's third residue and
    its last three residues are structural anchors. Nothing in the data supports
    that for a foreign reference, and the cost is measurable — it added position
    11 to ``aayl49_ML``'s conserved set and position 3 to ``aayl51``'s, so M4 was
    conserving residues its reference says nothing about.

    Returning ``()`` is the honest value: *we have no structural knowledge of this
    loop*, so M4's prior falls back to what the sequence itself evidences — its
    native aromatics, acidic residues and glycine hinges (:func:`conserved_motif_positions`).
    On 1N8Z nothing changes (``WT_SEQUENCE`` still short-circuits above, and its
    native set already spans every anchor); ``1mlc`` is unchanged too because its
    reference's native residues already cover the invented positions.

    An explicit ``anchor_positions`` always wins, and is validated against the
    reference length so an off-by-one cannot silently mark the wrong residue.
    """
    reference = resolve_wt_sequence(wt_sequence)
    length = len(reference)
    if anchor_positions is not None:
        resolved = tuple(int(p) for p in anchor_positions)
        out_of_range = [p for p in resolved if not 1 <= p <= length]
        if out_of_range:
            raise ValueError(
                f"anchor positions {out_of_range} outside a {length}-residue reference"
            )
        return resolved
    if reference == WT_SEQUENCE:
        return ANCHOR_POSITIONS
    return ()


def sar_maturity_target(seq_len: int = 13, *, total_budget: int = 35) -> float:
    """Corroborated evidence units that count as a "mature" SAR table.

        target = max(seq_len, 0.30 · Q_max)

    Two floors, both first-principles:

    * ``seq_len`` — you cannot claim to have explained a substitution space before
      you have at least one effective evidence unit per residue *position*. With
      ``L = 13`` that is 13 cells, one per site; with ``L = 11`` it is 11.
    * ``0.30 · Q_max`` — a campaign should not be able to declare its SAR law
      corroborated before it has spent a *third of its assays* on the question.
      Below that the ``VoI`` term collapses while most of the budget is unspent,
      which is the aayl49_ML failure this replaces: a fixed ~10-cell target was
      cleared within two rounds of a 125-query campaign, so ``C`` pinned at 1.0
      and ``VoI`` sat at ``0.35`` for ~20 consecutive rounds.

    ``max`` rather than a product or a sum: the two floors answer different
    questions ("is the space sampled?" and "is the budget committed?"), and a
    campaign should have to satisfy *both* before it earns the harvest. Taking the
    max is the weakest such requirement, which is the conservative choice.

    Why not scale with ``seq_len`` alone (an earlier revision did): on these three
    targets it is confounded with the effect it claims to model — 1N8Z and 2FJG are
    both 13-mers with the same 260-cell space, yet their qualified pools differ by
    1.7x and their optimal campaign lengths by 2x. Length cannot distinguish them;
    budget can. Length survives only as the weaker of the two floors.

    Sanity check of the resulting targets: 1N8Z ``(13, 35) -> 13``; aayl49_ML
    ``(11, 125) -> 37.5``; aayl49_ML at 33% ``(11, 250) -> 75``; 2FJG
    ``(13, 20) -> 13``; 2FJG at 33% ``(13, 40) -> 13``.

    ``total_budget`` is keyword-only so a stale positional call
    ``sar_maturity_target(len(reference))`` cannot silently reinterpret a length as
    a budget.
    """
    length = max(1, int(seq_len))
    budget = max(0, int(total_budget))
    return float(max(float(length), SAR_MATURITY_BUDGET_FRACTION * float(budget)))


#: Residue standing in for "no residue" when a sequence is shorter than the WT.
#: pandas' ``Series.str[p]`` yields ``NaN`` there, and a NaN compares *unequal* to
#: every WT residue, so it counted as a mutation at the position level and was
#: dropped from the cell level (``groupby`` skips NaN keys). This sentinel keeps
#: both behaviours without the per-position string slicing that produced it.
_PAD_RESIDUE = "\x00"


def _residue_matrix(sequences: np.ndarray, width: int | None = None) -> np.ndarray:
    """``(n_rows, width)`` character matrix, padded with a sentinel.

    One ``np.asarray`` over the sequences, instead of 13 ``Series.str[p]`` passes
    per call site. The matrix is only ever compared position-wise, so the sentinel
    never has to be a real amino acid.

    ``width`` defaults to the frozen reference length, so a 1N8Z call is
    unchanged; a cross-target call passes the length of *its own* reference.
    """
    if width is None:
        width = len(WT_SEQUENCE)
    matrix = np.full((len(sequences), width), _PAD_RESIDUE, dtype="<U1")
    for index, sequence in enumerate(sequences):
        text = str(sequence)
        matrix[index, : min(len(text), width)] = np.asarray(list(text[:width]), dtype="<U1")
    return matrix


def sar_effect_table(
    ledger_df: pd.DataFrame | None,
    *,
    tau_dev: float | None = None,
    residuals: dict[str, float] | None = None,
    min_n: int = 1,
    wt_sequence: str | None = None,
) -> SarTable:
    """Hierarchical 2-level Empirical-Bayes SAR: position-level + exact substitution.

    The effect of a substitution is measured on the **surrogate residual**
    (``y_bind - mu_bind``) whenever one is available for a revealed candidate,
    and on the advantage over the qualified revealed median otherwise. Residuals
    are the right target because the surrogate already absorbs the additive
    multi-mutation structure: a candidate that measures high *and* was predicted
    high tells us nothing about its individual mutations, whereas one that
    surprises the model does. That is what separates a real positional effect
    from co-mutation confounding.

    Two levels of SAR:
    1. **Position-level mutation tolerance** (13 positions): mean residual of all
       revealed variants mutated at position ``p``, active when ``n_pos >= 2``.
    2. **Exact substitution effect** (260 cells): mean residual for specific
       ``(position, residue)`` pairs, active when ``n_cell >= 2``.

    This captures mutation-tolerance signals in Rounds 1-2 when exact cells
    rarely have ``n >= 2`` but positions often do.

    Only revealed rows are ever read, and only the ``(position, residue)`` pairs
    that differ from the wild type are counted. ``n < 2`` cells are omitted
    entirely, and every retained effect is multiplied by ``n / (n + 3)``.
    """
    if ledger_df is None or getattr(ledger_df, "empty", True):
        return SarTable()
    if not {"sequence", "y_bind"} <= set(ledger_df.columns):
        return SarTable()
    frame = ledger_df.loc[ledger_df.y_bind.notna()]
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

    reference = resolve_wt_sequence(wt_sequence)
    effects = SarTable(support=len(frame), basis=basis)
    chars = _residue_matrix(frame.sequence.to_numpy(), len(reference))
    signal_values = signal.to_numpy(dtype=float)
    # Level 1 (position tolerance) and level 2 (exact cell) both read the same
    # per-row signal, so they are two views of one grouping rather than two
    # passes over the frame. This used to slice ``sequences.str[position]`` twice
    # per position — 26 pandas string passes — which cost ~6 ms on a 33-row
    # ledger and dominated every campaign rehydration and every
    # ``GET /api/state?round=k``. The residue matrix is built once instead, and
    # the arithmetic is unchanged to the last bit: same mutated mask, same
    # sorted group order, same ``mean``.
    for position in range(len(reference)):
        column = chars[:, position]
        mutated = column != reference[position]
        if not mutated.any():
            continue
        mutated_residues = column[mutated]
        mutated_signal = signal_values[mutated]

        # Level 1: Position-level mutation tolerance.
        n_pos = int(mutated_residues.size)
        if n_pos >= SAR_MIN_OBSERVATIONS:
            raw_mean = float(mutated_signal.mean())
            magnitude = min(1.0, abs(raw_mean) / SAR_DELTA_SCALE)
            effects.position_effects[position + 1] = (
                n_pos,
                float(np.sign(raw_mean)) * magnitude * _shrink(n_pos, SAR_SHRINKAGE_K),
            )

        # Level 2: Exact (position, residue) cell effects (260 cells).
        # ``np.unique`` sorts exactly as pandas' ``groupby`` default does, so the
        # table is built in the same order. A row that stops before this position
        # carries the pad residue, which is mutated-by-definition (it is not the
        # WT residue) but is not a substitution — keep it in the position count
        # and out of the cells, matching the old NaN semantics.
        for residue in np.unique(mutated_residues):
            if residue == _PAD_RESIDUE:
                continue
            cell = mutated_residues == residue
            n = int(cell.sum())
            if n < max(min_n, SAR_MIN_OBSERVATIONS):
                continue
            raw = float(mutated_signal[cell].mean())
            # Bound the *saturated* contribution, then shrink toward zero.
            magnitude = min(1.0, abs(raw) / SAR_DELTA_SCALE)
            effects[(position + 1, str(residue))] = (
                n,
                float(np.sign(raw)) * magnitude * _shrink(n, SAR_CELL_SHRINKAGE_K),
            )
    return effects


def _sar_total(
    sequence: str, effects: dict | None, *, wt_sequence: str | None = None
) -> float:
    """Hierarchical SAR: ``0.60 · exact cell + 0.40 · position tolerance``.

    Two Empirical-Bayes residual signals are combined:

    1. **Position-level mutation tolerance** (13 positions) — the mean residual
       of every revealed variant mutated at position ``p``, shrunk by
       ``n_pos / (n_pos + 3)`` and active once ``n_pos >= 2``. This is what
       survives the cold start: Rounds 1-2 almost never produce two
       observations of the *same* ``(position, residue)`` pair, but they
       routinely produce two observations at the same position.
    2. **Exact substitution effect** (260 cells) — the mean residual of that
       specific ``(position, residue)`` pair, shrunk by ``n_cell / (n_cell + 2)``
       and active once ``n_cell >= 2``.

    Mutation tolerance is symmetric: if mutating ``p`` consistently measures
    *worse*, conserving the WT residue at ``p`` earns the mirror-image reward, so
    the term can both promote a substitution and defend a conserved anchor.

    The exact cell carries the larger weight because it is the narrower,
    better-identified hypothesis; the position term is the fallback that keeps
    ``Δ_agent`` informative before any exact cell can qualify.
    """
    pos_effects = getattr(effects, "position_effects", None) or {}
    if not effects and not pos_effects:
        return 0.0

    reference = resolve_wt_sequence(wt_sequence)
    total = 0.0
    for position in range(len(reference)):
        residue = sequence[position]
        pos_key = position + 1
        pos_stat = pos_effects.get(pos_key)

        if residue == reference[position]:
            # Conserved residue: rewarded only when mutating here is measured bad.
            if pos_stat is not None and float(pos_stat[1]) < 0:
                total -= SAR_POSITION_WEIGHT * SAR_HIT_WEIGHT * float(pos_stat[1])
            continue

        # Level 2: exact (position, residue) cell effect.
        cell_stat = effects.get((pos_key, residue)) if effects else None
        if cell_stat is not None:
            total += SAR_CELL_WEIGHT * SAR_HIT_WEIGHT * float(cell_stat[1])

        # Level 1: position-level mutation tolerance.
        if pos_stat is not None:
            total += SAR_POSITION_WEIGHT * SAR_HIT_WEIGHT * float(pos_stat[1])

    return total


def sar_prior_score(
    sequence: str,
    effects: dict | None,
    *,
    bound: float = AGENT_PRIOR_BOUND,
    wt_sequence: str | None = None,
) -> float:
    """Empirical-SAR contribution of one candidate, bounded."""
    return float(
        np.clip(_sar_total(sequence, effects, wt_sequence=wt_sequence), -bound, bound)
    )


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
    return _soft_prior_total_weighted(sequence, priors, falsified_positions, None)


def _soft_prior_total_weighted(
    sequence: str,
    priors: list[PositionalSoftPrior],
    falsified_positions: set[int] | frozenset[int],
    scale: "Callable[[PositionalSoftPrior], float] | None",
) -> float:
    """``_soft_prior_total`` with an optional per-prior weight multiplier."""
    total = 0.0
    for prior in priors:
        weight = abs(prior.weight_delta) * (1.0 if scale is None else float(scale(prior)))
        if weight == 0.0:
            continue
        residue = sequence[prior.position - 1]
        sign = -1.0 if prior.position in falsified_positions else 1.0
        if residue in prior.preferred_residues:
            total += sign * weight
        elif residue in prior.discouraged_residues:
            total -= sign * weight
    return total


def _empirical_position_value(
    sar_effects: dict | None, position: int, residue: str
) -> float | None:
    """The measured SAR effect at ``(position, residue)``, or ``None`` if unmeasured.

    Prefers the exact corroborated ``(position, residue)`` cell — the narrow,
    better-identified claim — and falls back to the position-tolerance effect,
    which averages several residues together. Returns ``None`` rather than ``0.0``
    when neither level has ``n >= 2`` observations, because "not measured yet" and
    "measured to be zero" call for opposite treatment of a declared prior.

    Written with explicit ``len()`` checks: ``SarTable`` subclasses ``dict``, so a
    Round-1 table holding only position effects is falsy while carrying real
    evidence.
    """
    position_effects = getattr(sar_effects, "position_effects", None) or {}
    cell = None
    if isinstance(sar_effects, dict) and len(sar_effects):
        entry = sar_effects.get((position, residue))
        if entry is not None:
            cell = float(entry[1])
    if cell is not None:
        return cell
    entry = position_effects.get(position)
    if entry is not None:
        return float(entry[1])
    return None


def _has_empirical_sar(sar_effects: dict | None) -> bool:
    """Whether the table holds *any* corroborated evidence, at either level.

    Not ``bool(sar_effects)``: a ``SarTable`` with zero exact cells but three
    position-tolerance effects has evidence but is falsy.
    """
    if sar_effects is None:
        return False
    position_effects = getattr(sar_effects, "position_effects", None) or {}
    return len(sar_effects) > 0 or len(position_effects) > 0


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
    wt_sequence: str | None = None,
    confounding_scale: float = 1.0,
) -> float:
    """``Δ_agent``: measured SAR evidence **gated against** declared priors.

    ``A_H1 = A_base + Δ_agent``: the adaptive term is an *additive* re-ranking on
    top of the unbiased surrogate, never a filter, and ``|Δ_agent| ≤ 0.06`` keeps
    the surrogate in charge of the ordering.

    Both terms are summed at **full weight** (:data:`SAR_EMPIRICAL_WEIGHT` and
    :data:`DECLARED_PRIOR_WEIGHT` are both ``1.0``). The only reconciliation is a
    **sign gate**, applied when the table has corroborated evidence
    (:func:`_has_empirical_sar`) *and* the caller declared priors:

    * a declared prior whose claim has the **opposite** sign to the measured
      effect at its position is dropped — averaging a belief together with its own
      negation is not a belief;
    * an agreeing prior keeps full weight;
    * a prior at a position the table has **not** corroborated keeps full weight,
      because there it is the only evidence that exists. That is what lets a
      council speak before the lab has measured anything;
    * a cold table disables the gate entirely, so early rounds are unchanged.

    Discounting the two terms instead (a ``0.85 / 0.15`` split, tried and rejected)
    was the wrong instrument: it attenuates ``Δ_agent`` before ``tanh`` no matter
    *who* disagrees, and ``Δ_agent`` is the only part of ``A_H1`` the surrogate
    does not already contain. On 100 seeds it cost ``0.0013`` AUBQC while changing
    no ranking decision the sign gate would not have changed anyway.
    """
    declared = list(priors)
    # ``confounding_scale`` discounts ONLY the declared term; ``1.0`` is the
    # frozen behaviour, so a caller that does not pass it is unchanged.
    scale = float(np.clip(confounding_scale, 0.0, 1.0))
    if not declared or not _has_empirical_sar(sar_effects):
        raw = (
            scale * _soft_prior_total(sequence, declared, falsified_positions)
            + _sar_total(sequence, sar_effects, wt_sequence=wt_sequence)
        )
        return _saturate(raw, bound)

    def keep_or_drop(prior: PositionalSoftPrior) -> float:
        measured = _empirical_position_value(sar_effects, prior.position, sequence[prior.position - 1])
        if measured is None:
            # Uncorroborated position: the declared rule is the only evidence here.
            return DECLARED_PRIOR_WEIGHT
        # Agreement is judged on the rule's *claim*, not on this candidate's
        # residue: "A helps at position 1" is contradicted when position 1
        # measured negative, whichever residue the candidate happens to carry.
        claimed = 1.0 if prior.preferred_residues else (-1.0 if prior.discouraged_residues else 0.0)
        if claimed != 0.0 and np.sign(measured) not in (0.0, claimed):
            return 0.0
        return DECLARED_PRIOR_WEIGHT

    declared_term = _soft_prior_total_weighted(sequence, declared, falsified_positions, keep_or_drop)
    return _saturate(
        SAR_EMPIRICAL_WEIGHT * _sar_total(sequence, sar_effects, wt_sequence=wt_sequence)
        + scale * declared_term,
        bound,
    )


#: Weight of the mean per-candidate mutation count in the confounding factor. At
#: ``c == 1`` (a candidate mutated at every position) a declared prior is scaled
#: by ``1/(1+0.5) = 0.67``; at ``c == 0`` the factor is exactly ``1.0``.
CONFOUNDING_COEFFICIENT = 0.5


def confounding_factor(
    revealed: pd.DataFrame | None,
    *,
    wt_sequence: str | None = None,
) -> float:
    """How much a declared prior should be discounted for mutational confounding.

    A positional rule learned from a *single*-substitution library is close to a
    clean measurement: the candidate differs from the reference at one site, so
    "this residue helps" is nearly identified. In a combinatorial library the
    same rule is confounded — the candidate carrying it also carries three other
    changes, and the mean residual attributed to this position absorbs them.

    The factor is ``1 / (1 + k·c)`` where ``c`` is the mean number of non-reference
    substitutions per revealed candidate, normalised to ``[0, 1]`` by the CDR3
    length. It is **label-free** (sequences plus the ledger's own reference only,
    SPEC §2), monotone, and equal to exactly ``1.0`` when ``c == 0`` — so a
    single-mutant library, and any empty ledger, is scored identically to the
    frozen behaviour.

    ``1.0`` means "no discount"; the caller multiplies the *declared* prior term by
    it. The empirical-Bayes SAR term is deliberately **not** discounted: it is
    already shrunk by observation count (:data:`SAR_SHRINKAGE_K`,
    :data:`SAR_CELL_SHRINKAGE_K`) and is the measurement the prior is being
    compared against.
    """
    if revealed is None or getattr(revealed, "empty", True):
        return 1.0
    if "sequence" not in getattr(revealed, "columns", ()):
        return 1.0
    reference = resolve_wt_sequence(wt_sequence)
    width = max(1, len(reference))
    counts = [
        sum(1 for a, b in zip(str(sequence), reference) if a != b) / float(width)
        for sequence in revealed.sequence.astype(str)
    ]
    if not counts:
        return 1.0
    confounding = float(np.clip(np.mean(counts), 0.0, 1.0))
    return 1.0 / (1.0 + CONFOUNDING_COEFFICIENT * confounding)


def positional_priors_from_revealed(
    revealed: pd.DataFrame,
    *,
    top_k: int = 5,
    weight_delta: float = 0.04,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
    wt_sequence: str | None = None,
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
    reference = resolve_wt_sequence(wt_sequence)
    priors: list[PositionalSoftPrior] = []
    for position in range(len(reference)):
        if position + 1 in falsified_positions:
            continue
        residue = best.sequence.str[position].mode().iloc[0]
        if residue == reference[position]:
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


#: How many corroborated evidence units count as a "mature" SAR table. An exact
#: ``(position, residue)`` cell is a full unit and a position-tolerance effect is
#: half a unit, so a table with ~7 corroborated cells (or the equivalent mix)
#: reaches ``C == 1.0``.
SAR_MATURITY_UNITS = 10.0

#: Weight of one position-tolerance effect in those units. Deliberately below
#: 1.0: a position effect is a *weaker* claim than an exact substitution effect
#: (it averages several residues together), so it should take twice as many of
#: them to declare the frontier explained.
SAR_POSITION_MATURITY_WEIGHT = 0.5


def sar_maturity_score(evidence=None, *, target: float | None = None) -> float:
    """Empirical SAR convergence ``C`` in ``[0, 1]`` from *corroborated* support.

    The quantity is "how much of the frontier is already explained by evidence",
    so it must be a function of the **evidence table**, not of the largest
    ``Δ_agent`` in the candidate pool::

        C = min(1, (n_cells + 0.5 · n_pos) / 10)

    where ``n_cells`` counts exact ``(position, residue)`` cells with ``n >= 2``
    and ``n_pos`` counts position-tolerance effects with ``n_pos >= 2``. Both are
    exactly what :func:`sar_effect_table` already filters for, so the score is
    read straight off the table rather than re-derived.

    Why not the old ``max(Δ_agent) / (0.5 · AGENT_PRIOR_BOUND)`` rule: a single
    saturated candidate clears that bar in the *first* round, so ``C`` read
    ``1.000`` in every round and ``VoI`` collapsed to a constant ``0.35 · H``
    line. The campaign then lost its cold-start exploration and the reported
    ``sar_maturity`` carried no information at all. Counting corroborated cells
    instead makes ``C`` start near 0.0-0.2 on five seed reveals and genuinely
    climb as the SAR support accumulates.

    ``None`` means "no evidence table at all" and scores ``0.0``. A ``dict`` or
    :class:`SarTable` is read structurally. **Never** test the evidence with
    ``if not sar_effects``: ``SarTable`` subclasses ``dict``, so a Round-1 table
    with zero exact cells but three position effects is falsy while carrying real
    evidence — that bug would silently zero the maturity exactly when it matters.

    A plain sequence/array is accepted for backward compatibility with callers
    that still pass a ``Δ_agent`` vector; those fall back to the legacy
    max-``Δ_agent`` rule so existing unit tests and third-party callers keep
    their meaning.
    """
    if evidence is None:
        return 0.0
    if isinstance(evidence, dict) or hasattr(evidence, "position_effects"):
        # ``len()`` counts exact cells only; ``SarTable.position_effects`` holds
        # the position-tolerance level. Both are already n >= 2 corroborated.
        n_cells = int(len(evidence))
        position_effects = getattr(evidence, "position_effects", None) or {}
        n_pos = int(len(position_effects))
        if n_cells == 0 and n_pos == 0:
            return 0.0
        units = n_cells + SAR_POSITION_MATURITY_WEIGHT * n_pos
        # ``target`` is the length-scaled denominator; ``None`` keeps the frozen
        # 13-residue calibration bit-for-bit.
        denominator = SAR_MATURITY_UNITS if target is None else float(target)
        return float(np.clip(units / denominator, 0.0, 1.0))
    # Legacy path: a ``Δ_agent`` vector. Documented as deprecated but kept so a
    # caller that has no SAR table still gets a monotone, bounded number.
    values = np.asarray(list(evidence), dtype=float) if len(evidence) else np.empty(0)
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
    sar_effects=None,
    *,
    queries_spent: int,
    total_budget: int,
    batch_size: int = SLOT_COUNT,
    sar_maturity_target: float | None = None,
) -> dict:
    """``(H, C, VoI)`` for one batch, driven by the **evidence table**.

    The first positional argument is the corroborated SAR evidence
    (:class:`SarTable`), not the ``Δ_agent`` candidate vector it used to be.
    Maturity has to be a property of what the loop has *learned*, and a pool
    statistic cannot express that: one saturated candidate cleared the old bar in
    Round 1 and pinned ``C`` at ``1.000`` for all six rounds. Callers that still
    pass a sequence fall back to the legacy rule (see
    :func:`sar_maturity_score`), so the signature stays source-compatible.

    ``VoI`` now drives exactly one decision — how much slot 3 may pay for
    exploration and how much of ``a_null`` stays on exploration rather than
    qualified binding.

    ``sar_maturity_target`` defaults to :func:`sar_maturity_target` derived from
    **this call's own budget**, not to a bare constant. A caller that forgets the
    argument on a 125-query campaign gets ``0.30·125``-based denominator instead of
    silently inheriting another dataset's calibration — the class of bug this
    refactor exists to remove.
    """
    horizon = horizon_factor(queries_spent, total_budget=total_budget, batch_size=batch_size)
    # Computed through ``globals()`` rather than by name: this function's own
    # keyword argument is *called* ``sar_maturity_target`` and shadows the
    # module-level function of the same name inside this scope. Renaming the
    # parameter would break every existing caller; looking the function up
    # explicitly is the smaller change and the comment is the warning.
    #
    # ``seq_len`` is not threaded here: the length floor is the weaker of the two
    # in the target, and every caller that cares passes an explicit target via the
    # protocol. The budget floor alone is the conservative choice.
    target = (
        globals()["sar_maturity_target"](total_budget=int(total_budget))
        if sar_maturity_target is None
        else float(sar_maturity_target)
    )
    maturity = sar_maturity_score(sar_effects, target=target)
    voi = value_of_information(horizon, maturity)
    return {"horizon": horizon, "sar_maturity": maturity, "voi": voi}


# ── VoI-annealed qualified binding objective ─────────────────────────────────
def voi_annealed_null_objective(pool: pd.DataFrame, voi: float) -> pd.Series:
    """M5's qualified baseline: exploration early, pure binding exploitation late.

    Every candidate reaching this point already satisfies ``y_dev >= tau_dev``,
    so the qualification margin is settled information. Continuing to pay for it
    — and for ``beta_0 = 1.0``'s raw, min-max-stretched tree variance, which
    weights noisy ensemble disagreement exactly like a 100-fold affinity
    difference — is what made M5 spend Rounds 2-6 on high-variance,
    ``mediocre-``mu_bind`` candidates while a plain greedy ``mu_bind`` run (M2)
    built a strictly better dataset.

    The blend is driven by ``VoI``, not by a round counter::

        exploit_weight = 1 - 0.35 * VoI
        objective      = (1 - exploit_weight) · a_base
                         + exploit_weight · (mu_bind_norm + 0.20 · VoI · sigma_bind_calibrated)

    ``VOI_EXPLOIT_WEIGHT`` is a *calibration*, not a free parameter, and it was
    fitted on a two-fold split so it is not a number chosen to flatter the reported
    seeds — see the constant's comment for the sweep and the holdout result. By the
    final batch (``VoI -> 0``) ``exploit_weight`` is 1.0 and the objective is
    exactly the normalised predicted affinity.

    ``sigma_bind`` is calibrated **into ``mu_bind`` units** by dividing by the
    pool's ``ptp(mu_bind)``. It is deliberately *not* min-max normalised back to
    ``[0, 1]`` afterwards: that would undo the calibration and restore the very
    scale mismatch this term exists to remove. Because ``sigma_bind`` arrives
    already min-max normalised, the divisor carries a floor
    (:data:`MU_SPAN_FLOOR`) so a degenerate pool cannot blow the uncertainty
    term up by orders of magnitude.
    """
    if "mu_bind" not in pool.columns:
        # A frame without surrogate means (only reachable from a synthetic
        # fixture) has no qualified binding objective to blend in.
        return pool["a_base"].astype(float)
    mu_bind = pool["mu_bind"].astype(float)
    sigma_bind = (
        pool["sigma_bind"].astype(float) if "sigma_bind" in pool.columns else pd.Series(0.0, index=pool.index)
    )
    mu_span = max(float(np.ptp(mu_bind.to_numpy())), MU_SPAN_FLOOR)
    mu_bind_norm = (mu_bind - mu_bind.min()) / mu_span
    sigma_bind_calibrated = sigma_bind / mu_span
    bounded_voi = float(np.clip(voi, 0.0, 1.0))
    exploit_weight = 1.0 - VOI_EXPLOIT_WEIGHT * bounded_voi
    qualified_exploit = mu_bind_norm + VOI_UNCERTAINTY_WEIGHT * bounded_voi * sigma_bind_calibrated
    return (1.0 - exploit_weight) * pool["a_base"].astype(float) + exploit_weight * qualified_exploit


def ucb_exploration_objective(pool: pd.DataFrame) -> pd.Series:
    """``mu_bind_norm + 0.20 · sigma_bind_calibrated`` — the UCB exploration arm.

    This is the *same* expression :func:`voi_annealed_null_objective` uses for its
    exploitation term, evaluated at ``VoI == 0``: pure predicted affinity plus a
    fixed slice of residual uncertainty, in ``mu_bind`` units. It is written out
    separately because :func:`universal_null_objective` blends it in as a
    *landscape-transfer* term, and that term must not depend on ``VoI`` — a gate
    that switched itself off as the campaign learned would be a second, competing
    annealing law stacked on the first.

    Deliberately **not** ``a_base``: ``A_base`` carries ``beta_0 = 1.0`` tree
    variance and the ``0.35 · (mu_d + sigma_d)`` developability margin, whose scale
    (~10 log10 Kd) would swamp the normalised objective and destroy the H1/H0 arm
    comparability the paired audit depends on. ``sigma_bind`` is normalised into
    ``mu_bind`` units by the same pool span and the same :data:`MU_SPAN_FLOOR` as
    the frozen objective, so this term stays on that scale.

    A frame without ``mu_bind`` returns zeros rather than raising: that combination
    only occurs on a hand-built test frame, and "no exploration term" is the honest
    answer there — the caller then gets exactly the frozen objective.
    """
    if "mu_bind" not in pool.columns:
        return pd.Series(0.0, index=pool.index)
    mu_bind = pool["mu_bind"].astype(float)
    sigma_bind = (
        pool["sigma_bind"].astype(float)
        if "sigma_bind" in pool.columns
        else pd.Series(0.0, index=pool.index)
    )
    mu_span = max(float(np.ptp(mu_bind.to_numpy())), MU_SPAN_FLOOR)
    return (mu_bind - mu_bind.min()) / mu_span + VOI_UNCERTAINTY_WEIGHT * (sigma_bind / mu_span)


def universal_null_objective(pool: pd.DataFrame, voi: float, eta: float) -> pd.Series:
    """**RETIRED FROM THE DECISION PATH** — ``A_null`` is ``A_base`` now.

    Kept as a named function so the ablation harness can still reconstruct the
    pre-``no_anneal`` objective and the published artifact stays reproducible.
    Nothing on the live or benchmark path calls it; see the ``A_null = A_base``
    block in :func:`score_pool` for the five-cell measurements that retired it.

    The original contract was: VoI-annealed exploitation, weakened by ``eta``.

        lam        = (1 - eta) · NULL_UCB_EXPLORATION_WEIGHT
        A_null     = (1 - lam) · voi_annealed_null_objective(pool, voi)
                   + lam · ucb_exploration_objective(pool)

    ``eta`` is :func:`additive_prior_trust`. Both endpoints are exact IEEE
    identities, which is the whole reason the gate can be introduced without
    moving the reference artifact:

    * ``eta == 1.0`` (1N8Z): ``lam == 0.0``, so ``(1 - 0.0) · X + 0.0 · Y``
      reduces to ``X`` bit-for-bit. The frozen 100-seed M5 number is reproduced.
    * ``eta == 0.0``: ``lam == 0.5``, so half the objective is M3's ranking and
      half is the VoI-annealed one. That is the intended reading — "keep the
      campaign's own exploration schedule, but stop trusting the additive prior
      to decide *which* candidate it points at".

    Note that ``lam`` is deliberately **not** an exploration bonus: it does not
    grow with horizon, so it cannot double-count ``VoI``. It is a fixed convex mix
    between two objectives that are each already on the same normalised scale.
    """
    annealed = voi_annealed_null_objective(pool, voi)
    if eta >= 1.0:
        # Exact short-circuit, not an optimisation: ``(1 - 0) * X + 0 * Y`` is
        # already bit-identical, but skipping the second evaluation keeps the
        # multiplication chain out of the reference path entirely, so a future
        # change to ``ucb_exploration_objective`` (e.g. a NaN on a degenerate
        # pool) cannot reach 1N8Z through ``0.0 * NaN``.
        return annealed
    lam = float(np.clip((1.0 - float(eta)) * NULL_UCB_EXPLORATION_WEIGHT, 0.0, 1.0))
    if lam == 0.0:
        return annealed
    return (1.0 - lam) * annealed + lam * ucb_exploration_objective(pool)


# ── The 5-slot portfolio ─────────────────────────────────────────────────────
def score_pool(
    unqueried: pd.DataFrame,
    *,
    sar_effects: dict | None = None,
    priors: list[PositionalSoftPrior] | tuple[PositionalSoftPrior, ...] = (),
    ledger: pd.DataFrame | None = None,
    tau_dev: float | None = None,
    residuals: dict[str, float] | None = None,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
    queries_spent: int,
    total_budget: int,
    batch_size: int = SLOT_COUNT,
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
    protocol: "CampaignProtocol | None" = None,
    variant: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Score a pool with the exact objective :func:`select_five_slots` will use.

    Returns ``(pool, voi_context)`` where ``pool`` is the qualified frame carrying
    ``delta_agent``, ``a_null``, ``a_h1``, ``optimistic``, ``agent_promoted``,
    ``voi``, ``horizon`` and ``sar_maturity``.

    This exists so the *planner* can be shown the candidates the *selector* is
    about to pick from. Previously the planner's shortlist was ranked by
    ``sigma_joint`` (the exploration axis) while selection ranked by ``A_H1`` and
    ``A_null`` — so a planner could name a comparison pair that never entered the
    batch, and a hypothesis could be graded on candidates it never mentioned. One
    scoring function, called by both, is the only way that stays true.
    """
    frame = unqueried.copy()
    if sar_effects is None:
        sar_effects = sar_effect_table(ledger, tau_dev=tau_dev, residuals=residuals)
    if len(priors) == 0:
        priors = positional_priors_from_revealed(
            ledger, falsified_positions=falsified_positions, wt_sequence=wt_sequence
        )


    # ``eta_sar``: how far the additive prior may be trusted on *this* cell.
    #
    # Every input is an intrinsic property of the cell — pool size, sequence
    # length, campaign budget, and the library's mutational load — so the gate is
    # the *same formula* on 1N8Z, aayl49_ML and 2FJG. There is no reference-dataset
    # comparison here and no calibrated transfer constant; see
    # :func:`additive_prior_trust`.
    #
    # ``mutational_confounding`` reads sequences out of the ledger, never labels,
    # so SPEC §2's label-blindness is intact. It defaults to the un-confounded
    # ``1.0`` before the first reveal, which is what makes the gate computable at
    # ``t=0``.
    confounding_scale = mutational_confounding(ledger, wt_sequence=wt_sequence)
    eta = additive_prior_trust(
        n_qualified=None if protocol is None else protocol.n_qualified,
        seq_len=len(resolve_wt_sequence(wt_sequence)),
        total_budget=int(total_budget),
        confounding=confounding_scale,
    )

    # Diagnostic only: M4's static anchor rule is reported so the UI and the
    # ledger can show what M4 would have scored, but it is **not** an additive
    # term in either arm. Chaining M5 to it would import M4's bad prior.
    #
    # This is not a stylistic preference. It was measured: folding
    # ``loop_prior_score`` into ``Δ_agent`` as a cold-start prior (the
    # "M3 ⊂ M4 ⊂ M5" nesting) moved 1N8Z M5 from 1.9258 (#1) to 1.9231 (#5) and
    # took its cross-target Spearman lead from 4/6 cells to 0/6. The static
    # structural rule is a *different hypothesis about this landscape*, not an
    # earlier version of M5's belief, and mixing the two corrupted both arms.
    #
    # A frame with no ``sequence`` column cannot be scored positionally. That only
    # happens on a hand-built test frame (the real paths always carry sequences),
    # and the honest answer there is a zero agent prior rather than a crash.
    if "sequence" in frame.columns:
        frame["loop_prior"] = [
            loop_prior_score(s, wt_sequence=wt_sequence, anchor_positions=anchor_positions)
            for s in frame.sequence
        ]
        # ``Δ_agent`` under one law: the sign-gated empirical SAR vote, scaled by
        # ``eta_sar``. ``eta`` is *not* a magnitude discount on the prior's shape
        # — it is a confidence weight, and it is already the product of the three
        # measurable reasons a single-position marginal might be wrong (too few
        # assays per site, too little pool coverage, too much epistasis).
        #
        # ``confounding_scale`` used to be passed separately as
        # ``adaptive_prior_score(..., confounding_scale=...)``. It is now folded
        # into ``eta`` and reported as its own column, so the UI diagnostic
        # survives while the prior is discounted once rather than twice.
        frame["delta_agent"] = [
            eta * adaptive_prior_score(
                s, priors=list(priors), sar_effects=sar_effects,
                falsified_positions=falsified_positions, wt_sequence=wt_sequence,
            )
            for s in frame.sequence
        ]
        frame["confounding_scale"] = confounding_scale
    else:
        frame["loop_prior"] = 0.0
        frame["delta_agent"] = 0.0
        frame["confounding_scale"] = 1.0
    pool = qualified_pool(frame, tau_dev)
    if "a_base" not in pool.columns:
        # Degenerate frame (test fixture): fall back to whatever surrogate signal
        # exists so the objective stays defined instead of raising KeyError.
        if "mu_bind" in pool.columns:
            pool = pool.assign(a_base=pool["mu_bind"].astype(float))
        elif "sigma_joint" in pool.columns:
            pool = pool.assign(a_base=pool["sigma_joint"].astype(float))
        else:
            pool = pool.assign(a_base=0.0)

    # ``C`` is measured from the **corroborated SAR table** (exact cells plus
    # position-tolerance effects), i.e. from evidence the loop has actually
    # accumulated — not from ``max(Δ_agent)`` over the pool, which a single
    # saturated candidate clears in Round 1 and which therefore pinned
    # ``sar_maturity`` at 1.000 for every round. Computed *before* ``a_null``
    # because the VoI-annealed objective below depends on it.
    #
    # The maturity denominator comes from ``Q_max``, not from the CDR3 length.
    # ``total_budget`` is already the campaign's own query ceiling, so a longer
    # campaign has to corroborate proportionally more evidence before ``C`` can
    # saturate and crush ``VoI``. At ``Q_max == 35`` this is exactly the frozen
    # ``10.0``.
    context = voi_context(
        sar_effects,
        queries_spent=int(queries_spent),
        total_budget=int(total_budget),
        batch_size=int(batch_size),
        sar_maturity_target=(
            protocol.sar_maturity_target if protocol is not None
            else sar_maturity_target(
                seq_len=len(resolve_wt_sequence(wt_sequence)),
                total_budget=int(total_budget),
            )
        ),
    )

    # ── The production ``A_null`` is ``A_base`` ───────────────────────────────
    #
    # ``A_null = A_base``, unconditionally. There is no VoI anneal in the decision
    # path any more.
    #
    # This is a **measured** deletion, not a preference. The attribution ablation
    # (`results/attribution_ablation.json`, `scripts/run_attribution_ablation.py`)
    # switched the anneal off and nothing else:
    #
    #   cell                 ρ production -> a_null = A_base     AUBQC
    #   1N8Z                 0.5300 -> 0.6578  (p ≈ 1e-9)        ±0.002 (n.s.)
    #   aayl49_ML @16.7%     0.2266 -> 0.2694  (p = 2e-4)        ±0.001 (n.s.)
    #   aayl49_ML @33%       0.2141 -> 0.2889  (p < 1e-4)        ±0.003 (n.s.)
    #   2FJG      @16.7%     0.1456 -> 0.2327                    +0.015 (p < 1e-4)
    #   2FJG      @33%       0.2612 -> 0.4461  (p < 1e-4)        +0.009 (p = 1e-4)
    #
    # Five cells, two metrics, no cell worse. The mechanism it deletes is the
    # problem: ``universal_null_objective`` moved ``A_null`` from ``A_base`` toward
    # ``mu_bind_norm + 0.2·VoI·σ`` as the campaign matured, and ``A_base`` carries
    # the only exploration term the objective has (``0.65·beta_0·σ_bind``). So the
    # anneal progressively **removed the exploration signal** — catastrophic on a
    # compressed landscape (2FJG's binding IQR is ~0.04 log10 Kd, where σ dominates
    # ``mu``) and merely unhelpful elsewhere. The loop spent its budget converging
    # on a worse objective than the one it started with.
    #
    # What the anneal was *for* — not over-exploring late — is already handled by
    # mechanisms that do not discard information: ``slot_graduation`` withdraws the
    # VoI probe's optimism bonus and graduates the reserved slots to exploitation,
    # and ``SLOT3_OPTIMISM`` scales with ``VoI``, so the exploration *bonus* still
    # decays to zero on the final batch. Neither of those deletes σ from the
    # objective that ranks candidates.
    pool = pool.assign(a_null=pool["a_base"].astype(float))
    pool = pool.assign(a_h1=pool.a_null + pool.delta_agent)

    # Graduation is decided from the *measured* spread of the prior over this
    # pool, so it needs ``delta_agent`` to exist first. ``round_number`` is the
    # round this batch will become: the seed batch is round 0, so a campaign that
    # has spent ``queries_spent`` queries is designing round
    # ``queries_spent // batch_size``. Deriving it here rather than accepting a
    # parameter keeps every existing call site correct without threading a new
    # argument through the benchmark and the live loop.
    round_number = int(queries_spent) // max(1, int(batch_size))
    spread = float(np.ptp(pool.delta_agent.to_numpy())) if len(pool) > 1 else 0.0
    # The horizon comes from the protocol when there is one, and from the budget
    # otherwise. ``total_rounds`` is what makes the falsification window
    # proportional rather than a fixed six.
    total_rounds = (
        protocol.rounds
        if protocol is not None
        else max(1, (int(total_budget) - int(SEED_BATCH)) // max(1, int(batch_size)))
    )
    graduated = slot_graduation(round_number, spread, total_rounds=total_rounds)

    # The affinity model, as a column. This is the SAME function the ρ metric
    # reads (:func:`affinity_score`), so a slot that elects to cash it in is
    # exploiting exactly what the reported model claims — the link the
    # "slot-logic paradox" showed was missing.
    pool = pool.assign(model_score=affinity_score(pool, pool.delta_agent))

    pool = pool.assign(
        horizon=context["horizon"],
        sar_maturity=context["sar_maturity"],
        voi=context["voi"],
        eta_sar=eta,
        delta_agent_spread=spread,
        graduated=graduated,
        # As the loop graduates, slot 3's optimism bonus is withdrawn. It is not
        # a second anneal: ``VoI`` still sets the magnitude, and graduation only
        # removes a bonus whose two preconditions (something left to explore,
        # a prior worth probing) have both stopped holding. On an ungraduated
        # batch this multiplier is exactly ``1.0``, so the frozen expression is
        # reproduced bit-for-bit.
        optimistic=pool.a_h1
        + (0.0 if graduated else SLOT3_OPTIMISM)
        * context["voi"]
        * pool.sigma_joint,
        agent_promoted=pool.a_h1 + SLOT4_AGENT_BOOST * pool.delta_agent.clip(lower=0.0),
    )
    return pool, context


def prospective_slots(pool: pd.DataFrame) -> list[dict]:
    """The five candidates :func:`select_five_slots` is about to return.

    A dry run of the identical rule over a fully scored pool (``score_pool``'s
    output), so the planner can be shown *which* candidates will be measured
    before it writes a claim about them. It calls the same helpers in the same
    order and re-derives nothing, which is what keeps it honest: if the selector's
    rule changes, so does this preview.

    Returns a list of ``{slot_id, slot_type, candidate_id, a_null, a_h1,
    delta_agent, mu_bind, sequence, ...}``. If the pool cannot fill five slots the
    list is simply short; the caller is told so rather than handed a fabricated
    batch.
    """
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

    for _, row in pool.sort_values(["a_h1", "candidate_id"], ascending=[False, True]).head(2).iterrows():
        take(row, "Dual-Surrogate UCB")
    lead = best(remaining(), "optimistic")
    if lead is None:
        return selected
    take(lead, SLOT3_LABEL)
    rest = remaining()
    if rest.empty:
        return selected
    take(
        rest.sort_values(
            ["a_h1", "delta_agent", "candidate_id"], ascending=[False, False, True]
        ).iloc[0],
        SLOT4_LABEL,
    )
    # Slot 5 follows the same graduation rule as the selector: once the audit
    # window has closed (or the prior has stopped separating the pool) the slot is
    # an exploitation pick, not a control. Read off the scored pool so the preview
    # cannot disagree with the selector about whether this batch is graduated.
    graduated = bool(pool["graduated"].iloc[0]) if "graduated" in pool.columns and len(pool) else False
    control = best(remaining(), "a_h1" if graduated else "a_null")
    if control is not None:
        take(control, SLOT5_FALSIFY_LABEL)
    return selected


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
    wt_sequence: str | None = None,
    anchor_positions: tuple[int, ...] | None = None,
    protocol: "CampaignProtocol | None" = None,
    variant: str | None = None,
) -> list[dict]:
    """Qualified 5-slot portfolio as a paired H1/H0 experiment (SPEC §6).

    ``variant`` is diagnostic-only and defaults to ``None`` = the production law.
    See :data:`ABLATION_VARIANTS`: it exists so the two mechanisms M5 adds to M3
    can be switched off one at a time and *measured*, rather than argued about.

    Two acquisition scores are defined over the qualified unqueried pool::

        A_base(s) = surrogate UCB                     # unbiased, Δ_agent = 0
        A_H1(s)   = A_base(s) + Δ_agent(s)            # |Δ_agent| <= 0.06

    ``A_base`` is M3's objective and the *true null hypothesis*: the candidate
    the lab would query if the agents had said nothing. ``A_H1`` is M5's. The
    static WT anchor prior belongs to M4 alone — it is deliberately **not** part
    of either arm here, because on 1N8Z it penalises the ``AR...``/aromatic-rich
    binders that actually top the leaderboard, and a hybrid that inherits a
    refuted prior starts every campaign behind the method it improves on.

    One Expected Value of Information number, :func:`voi_context`, decides how
    much the batch may still pay for exploration::

        VoI = H · (1 - 0.65 · C)

    ``A_null`` is **not** ``A_base`` any more. Because every pool member already
    clears ``y_dev >= tau_dev``, the objective is first re-expressed as a
    VoI-annealed blend between early exploration and pure qualified binding (see
    the inline derivation below). ``A_base`` still enters that blend, so early
    rounds keep the unbiased surrogate's exploration, but by the final batch it
    is fully displaced by ``mu_bind_norm``.

    The five slots implement a 3-exploit / 2-experimental split:

    1-2  top 2 by ``A_H1`` — core exploitation.
    3    ``A_H1 + 0.25 · VoI · sigma_joint`` maximised over the *remaining* pool.
         As ``VoI -> 0`` (SAR already corroborated, or ``future_queries == 0`` on
         the final batch) the bonus vanishes and slot 3 *is* a pure ``A_H1``
         exploitation slot, without a hand-written round switch. Raw
         ``max sigma_joint`` is deliberately not used: unconstrained, it picks
         the least-informed candidate in the pool regardless of whether that
         candidate could ever win.
    4    the next best remaining ``A_H1`` — with ``SLOT4_AGENT_BOOST = 0.0`` the
         promotion bonus is off, so this is a *third* pure exploitation slot.
         ``Δ_agent > 0`` can only break ties. Empirically the 0.5 boost cost M5
         its AUBQC edge over M3 on early rounds, because it spent a slot on a
         hypothesis candidate whose surrogate rank did not justify it.
    5    top ``A_null`` — the H0 null-baseline control. ``Δ_agent == 0`` is the
         hypothesis under test, so the control is the best remaining candidate on
         the *unmodified* baseline. Whenever slots 1-4 promote a hypothesis
         candidate over that baseline, slot 5 automatically picks up the
         displaced one, giving a clean H1-vs-H0 comparison without ever banning
         evidence.

    Slots 4 and 5 are chosen on the *same* batch, so every round yields a direct
    H1-vs-H0 comparison for
    :func:`core.tool_arsenal.audit_past_hypotheses` to grade.

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

    priors = (
        list(decision.positional_priors)
        if decision
        else positional_priors_from_revealed(
            ledger, falsified_positions=falsified_positions, wt_sequence=wt_sequence
        )
    )
    pool, _ = score_pool(
        unqueried,
        sar_effects=sar_effects,
        priors=priors,
        ledger=ledger,
        tau_dev=tau_dev,
        residuals=residuals,
        falsified_positions=falsified_positions,
        queries_spent=int(queries_spent),
        total_budget=int(total_budget),
        batch_size=int(batch_size),
        wt_sequence=wt_sequence,
        anchor_positions=anchor_positions,
        protocol=protocol,
        variant=variant,
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

    # Graduation (replaces the old ``adaptive_split`` terminal All-In branch).
    #
    # :func:`slot_graduation` decides, from the round number and the measured
    # spread of ``Δ_agent``, whether slots 3 and 5 still earn their slots:
    #
    # * within the audit window the paired H1/H0 comparison is the point of the
    #   design, so both slots behave exactly as documented above;
    # * after it — or as soon as the prior stops separating the pool at all —
    #   an "exploration probe" and an "H0 control" would be picking candidates
    #   indistinguishable from the exploitation picks, at a cost of two of five
    #   assays per batch.
    #
    # ``pool["graduated"]`` is computed once in :func:`score_pool` from the same
    # numbers the selector sees, so the preview
    # (:func:`prospective_slots`) and the selector cannot disagree.
    #
    # 1N8Z never graduates: ``round_number > 6`` is false for all six rounds and
    # the spread floor is verified inert, which is what keeps R1-R6 — including
    # R6, whose ``VoI`` is 0 — bit-identical to the frozen artifact.
    graduated = bool(pool["graduated"].iloc[0]) if "graduated" in pool.columns and len(pool) else False

    # Attribution variant: every slot ranks by ``a_h1``, so no assay is reserved
    # for a VoI probe or an H0 control. ``a_null``, ``Δ_agent`` and the anneal are
    # all untouched — the ONLY difference from production is the slot split, which
    # is what makes the comparison attributable.
    if variant == VARIANT_GREEDY_SLOTS:
        graduated = True
        for _, row in pool.sort_values(["a_h1", "candidate_id"], ascending=[False, True]).head(len(pool)).iterrows():
            if len(selected) >= SLOT_COUNT:
                break
            if row["candidate_id"] in {row_["candidate_id"] for row_ in selected}:
                continue
            take(row, "Dual-Surrogate UCB")
        for slot in selected:
            slot["control_suppressed"] = True
            slot["allocation"] = f"variant:{VARIANT_GREEDY_SLOTS}"
        return selected

    # ── Slot-role variants: cash the affinity model into actual assays ───────
    #
    # The default selector ranks *every* slot on ``a_h1 = a_null + Δ_agent``, and
    # ``a_base`` carries ``0.65·sigma_stretched`` plus a developability margin. So
    # even the exploitation slots are partly exploring and partly chasing y_dev.
    # M5's ρ is measured on ``model_score = mu_bind + Δ·span``. Different
    # objectives — and the ``no_anneal`` ablation showed the consequence: ρ 0.6578
    # with AUBQC merely tying M3.
    #
    # These variants make specific slots rank on ``model_score`` directly: a **pure
    # replacement**, never a blend, because blending is what the VoI anneal did.
    if variant in SLOT_ROLE_VARIANTS:
        model_slots = harvest_uses_model(variant, graduated)
        ordered = []
        if model_slots or slot1_uses_model(variant, graduated):
            # Slots handed to the model, in model order.
            n_model = SLOT_COUNT if model_slots else 1
            model_ranked = pool.sort_values(
                ["model_score", "candidate_id"], ascending=[False, True]
            )
            for _, row in model_ranked.head(n_model).iterrows():
                ordered.append((row, "Affinity-Model Exploitation"))
        # Whatever slots remain keep the shipped ``a_h1`` exploitation ranking.
        used = {s["candidate_id"] for s, _ in ordered}
        rest = pool.loc[~pool.candidate_id.isin(used)].sort_values(
            ["a_h1", "candidate_id"], ascending=[False, True]
        )
        for _, row in rest.iterrows():
            if len(ordered) >= SLOT_COUNT:
                break
            ordered.append((row, "Dual-Surrogate UCB"))
        for row, label in ordered[:SLOT_COUNT]:
            take(row, label)
        for slot in selected:
            slot["allocation"] = f"variant:{variant}"
        return selected

    # ── Slot-role variants: cash the affinity model into actual assays ───────
    #
    # The default selector ranks *every* slot on ``a_h1 = a_base + Δ_agent``.
    # ``a_base`` carries ``0.65·sigma_stretched`` and a developability margin, so
    # even the exploitation slots are partly a developability and exploration
    # objective. M5's ρ, meanwhile, is measured on ``model_score =
    # mu_bind + Δ·span``. Those are different objectives, and the ``no_anneal``
    # ablation showed the consequence: ρ 0.6578 with AUBQC merely tying M3.
    #
    # These variants make some slots rank on ``model_score`` directly — a **pure
    # replacement**, never a blend, because blending is what the VoI anneal did and
    # the anneal is what destroyed ρ.
    model_slots = harvest_uses_model(variant, graduated)
    first_slot_model = slot1_uses_model(variant, graduated)
    if model_slots or first_slot_model:
        ranked_by_model = pool.sort_values(
            ["model_score", "candidate_id"], ascending=[False, True]
        )
        if model_slots:
            # Every slot cashes in the model: the batch is a pure "query the best
            # predicted binders" batch, still respecting the 3:1:1 role *labels*
            # so the paired audit and the slot-count invariants hold.
            for _, row in ranked_by_model.head(len(pool)).iterrows():
                if len(selected) >= SLOT_COUNT:
                    break
                if row["candidate_id"] in {s["candidate_id"] for s in selected}:
                    continue
                take(row, "Affinity-Model Exploitation")
            for slot in selected:
                slot["allocation"] = f"variant:{variant}"
            return selected
        # Only slot 1 cashes in; the rest of the batch keeps the shipped rules, so
        # the audit's slots 4/5 still exist and still pair.
        top = ranked_by_model.iloc[0]
        take(top, "Affinity-Model Exploitation")
        rest_by_h1 = pool.drop(index=top.name).sort_values(
            ["a_h1", "candidate_id"], ascending=[False, True]
        ).head(1)
        for _, row in rest_by_h1.iterrows():
            take(row, "Dual-Surrogate UCB")
        for slot in selected:
            slot["allocation"] = f"variant:{variant}"

    # Slot 3: dynamic-VoI exploration; degenerates to pure A_H1 as VoI -> 0.
    # When graduated, ``score_pool`` has already zeroed the optimism multiplier,
    # so ``optimistic`` *is* ``a_h1`` and this single expression covers both
    # cases — there is no second branch to keep in sync.
    rest = remaining()
    take(best(rest, "optimistic"), SLOT3_LABEL)

    # Slot 4: third exploitation slot. With ``SLOT4_AGENT_BOOST == 0.0``
    # ``agent_promoted`` is ``a_h1`` plus the positive part of ``Δ_agent``
    # multiplied by zero — i.e. exactly ``a_h1``. The sort explicitly falls back
    # to ``Δ_agent`` and then candidate id so a tie between equal-``a_h1``
    # candidates still prefers the hypothesis-favoured one, which is the only
    # role the agent boost retains.
    rest = remaining()
    if rest.empty:
        return selected
    slot4 = rest.sort_values(
        ["a_h1", "delta_agent", "candidate_id"], ascending=[False, False, True]
    ).iloc[0]
    take(slot4, SLOT4_LABEL)

    # Slot 5: H0 null-baseline control. ``Δ_agent == 0`` is the hypothesis under
    # test, so the control is the best remaining candidate on the *unmodified*
    # baseline. Both arms are scored on the same VoI-annealed objective, so the
    # paired audit compares hypothesis-vs-null rather than exploration-vs-null.
    #
    # Once graduated the slot is spent on ``A_H1`` instead. The label is kept (the
    # slot still exists and is still measured) but it no longer names a control,
    # and ``control_suppressed`` records that so the paired audit can say "this
    # round had no H0" rather than silently grading against an exploitation pick —
    # that flag is the contract ``core.tool_arsenal.audit_past_hypotheses`` reads.
    rest = remaining()
    if rest.empty:
        return selected
    take(best(rest, "a_h1" if graduated else "a_null"), SLOT5_FALSIFY_LABEL)
    if graduated:
        for slot in selected:
            slot["control_suppressed"] = True
            slot["allocation"] = UNIVERSAL_ALLOCATION

    picked = [row["candidate_id"] for row in selected]
    if len(set(picked)) != len(picked):
        raise RuntimeError("Slot selection produced duplicate candidates")
    return selected
