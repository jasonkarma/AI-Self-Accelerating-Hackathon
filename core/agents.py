"""Async scientist agents; all numerical claims originate from deterministic tools.

Council composition is data, not control flow.  ``AGENT_DEFINITIONS`` binds each
role to (a) the prompt it receives, (b) the deterministic tools it may cite, and
(c) the ``AgentModel`` transport assignment from :mod:`core.llm`.  Thinking
budgets therefore live in exactly one place and can be retuned per agent without
touching orchestration code.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pandas as pd

from .llm import (
    AGENT_MODELS,
    AgentModel,
    ModelRouter,
    resolve_reasoning_effort,
    resolve_thinking_budget,
)
from .prompts import (
    AFFINITY_SCIENTIST_PROMPT,
    DEVELOPABILITY_SCIENTIST_PROMPT,
    EXPERIMENTAL_PLANNER_SYNTHESIZER_PROMPT,
    SKEPTIC_CRITIC_PROMPT,
)
from .acquisition import (
    AGENT_PRIOR_BOUND,
    adaptive_prior_score,
    positional_priors_from_revealed,
    sar_effect_table,
)
from .schemas import AgentMemo, PositionalSoftPrior, RoundDecisionRecord


@dataclass(frozen=True)
class AgentDefinition:
    """One seat on the council."""

    slug: str
    prompt: str
    tools_invoked: tuple[str, ...]
    model: AgentModel
    #: Schema enforced by strict structured outputs (planner only).
    response_schema: type | None = None

    @property
    def role(self) -> str:
        return self.model.role

    @property
    def thinking_budget(self) -> int | None:
        return self.model.thinking_budget


#: Cognitive-diversity roster. Model ids come from ``core.llm.AGENT_MODELS``;
#: thinking budgets are intentionally asymmetric — the Skeptic deliberates,
#: the Developability Scientist answers directly.
AGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "affinity": AgentDefinition(
        slug="affinity",
        prompt=AFFINITY_SCIENTIST_PROMPT,
        tools_invoked=("scan_positional_sar", "detect_epistasis_clashes"),
        model=AGENT_MODELS["affinity"],
    ),
    "developability": AgentDefinition(
        slug="developability",
        prompt=DEVELOPABILITY_SCIENTIST_PROMPT,
        tools_invoked=("profile_developability_liabilities",),
        model=AGENT_MODELS["developability"],
    ),
    "skeptic": AgentDefinition(
        slug="skeptic",
        prompt=SKEPTIC_CRITIC_PROMPT,
        tools_invoked=("audit_past_hypotheses", "inspect_joint_uncertainty_hotspots"),
        model=AGENT_MODELS["skeptic"],
    ),
    "planner": AgentDefinition(
        slug="planner",
        prompt=EXPERIMENTAL_PLANNER_SYNTHESIZER_PROMPT,
        tools_invoked=(),
        model=AGENT_MODELS["planner"],
        response_schema=RoundDecisionRecord,
    ),
}

#: The three debating scientists, in stable council order.
SCIENTIST_SLUGS: tuple[str, ...] = ("affinity", "developability", "skeptic")


def _schema_for(definition: AgentDefinition) -> dict | None:
    """Serialize the pydantic contract for strict structured outputs."""
    if definition.response_schema is None:
        return None
    return definition.response_schema.model_json_schema()


#: How many SAR rows the planner is shown. Enough to reason about which
#: substitutions are actually working, few enough to keep the prompt bounded.
SAR_HINT_ROWS = 12


def surrogate_residuals(
    revealed: pd.DataFrame | None,
    mu: dict[str, float] | None,
) -> dict[str, float]:
    """``y_bind - mu_bind`` for each revealed candidate.

    Residuals, not raw bindings, are the right target for a positional effect:
    the surrogate already absorbs the additive multi-mutation structure, so a
    candidate that measured high *and* was predicted high says nothing about its
    individual substitutions, while a candidate that surprised the model does.
    Using raw ``y_bind`` is what let co-mutation confounding masquerade as a
    single-site effect.
    """
    if revealed is None or getattr(revealed, "empty", True) or not mu:
        return {}
    residuals: dict[str, float] = {}
    for row in revealed.itertuples():
        predicted = mu.get(str(row.candidate_id))
        if predicted is None or pd.isna(row.y_bind):
            continue
        residuals[str(row.candidate_id)] = float(row.y_bind) - float(predicted)
    return residuals


def adaptive_prior_evidence(
    ledger: pd.DataFrame | None,
    tau_dev: float | None,
    *,
    priors: list[PositionalSoftPrior] | None = None,
    sequences: list[str] | None = None,
    residuals: dict[str, float] | None = None,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
) -> dict:
    """Deterministic evidence behind ``Δ_agent``, for the planner and the fallback.

    The agents' prior is the council's declared positional priors **plus** an
    empirical-Bayes-shrunk SAR term. Both halves are shown to the planner so a
    memo can cite *why* a substitution scores, and the same numbers feed
    :func:`core.acquisition.adaptive_prior_score` at selection time - one code
    path, so a live run and the benchmark cannot diverge.

    Nothing here reads ``y_bind`` for an unrevealed candidate: ``ledger`` is the
    campaign's revealed frame by construction.
    """
    effects = sar_effect_table(ledger, tau_dev=tau_dev, residuals=residuals)
    derived = list(
        priors
        if priors is not None
        else positional_priors_from_revealed(ledger, falsified_positions=falsified_positions)
    )
    ranked = sorted(effects.items(), key=lambda item: abs(item[1][1]), reverse=True)[:SAR_HINT_ROWS]
    return {
        "sar_basis": effects.basis,
        "sar_support": effects.support,
        "sar_effects": [
            {"position": position, "residue": residue, "n": n, "shrunk_effect": round(effect, 3)}
            for (position, residue), (n, effect) in ranked
        ],
        "falsified_positions": sorted(int(p) for p in falsified_positions),
        "positional_priors": [prior.model_dump() for prior in derived],
        "scored_examples": (
            {
                sequence: round(
                    adaptive_prior_score(
                        sequence,
                        priors=derived,
                        sar_effects=effects,
                        falsified_positions=falsified_positions,
                    ),
                    4,
                )
                for sequence in sequences
            }
            if sequences
            else {}
        ),
        "bounds": {
            "delta_agent": AGENT_PRIOR_BOUND,
            "note": "Δ_agent is tanh-saturated; it can re-rank neighbours, never override the surrogate",
        },
    }


def deterministic_priors(
    ledger: pd.DataFrame | None,
    tau_dev: float | None,
    *,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
) -> list[PositionalSoftPrior]:
    """Priors used by the offline fallback decision and by ``AdaptiveToolPolicy``."""
    return positional_priors_from_revealed(ledger, falsified_positions=falsified_positions)


async def run_agent(
    router: ModelRouter,
    slug: str,
    *,
    context: dict,
    definition: AgentDefinition | None = None,
) -> str:
    """Execute a single council seat and return its memo text."""
    spec = definition or AGENT_DEFINITIONS[slug]
    schema = _schema_for(spec)
    return await router.run(spec.role, spec.prompt, context, schema=schema)


async def run_scientists(
    router: ModelRouter,
    tools: dict,
    *,
    on_memo: Callable[[AgentMemo], Awaitable[None]] | None = None,
    on_error: Callable[[str, BaseException], Awaitable[None]] | None = None,
) -> list[AgentMemo]:
    """Run the three scientists concurrently (Skeptic joins the same gather).

    ``on_memo`` fires as each seat finishes, so a CLI/SSE consumer can stream the
    debate instead of waiting for the slowest model. A seat that raises still
    produces a memo describing the failure, so one bad credential or model id
    does not abort the round.
    """

    async def one(definition: AgentDefinition) -> AgentMemo:
        resolved = router.spec_for(definition.role)
        context = {
            "role": definition.role,
            "tools": {name: tools.get(name) for name in definition.tools_invoked},
            "fallback": f"{definition.role} reviewed deterministic evidence; confidence remains bounded.",
        }
        try:
            text = await run_agent(router, definition.slug, context=context, definition=definition)
            trace = router.thinking_trace(definition.role) or text
        except Exception as exc:  # noqa: BLE001 - one bad seat must not kill the round
            text = f"{definition.role} unavailable ({type(exc).__name__}): {exc}"
            trace = text
            if on_error is not None:
                await on_error(definition.role, exc)
        memo = AgentMemo(
            role=definition.role,
            tools_invoked=list(definition.tools_invoked),
            thinking_trace=trace,
            memo=text,
            model=resolved.model if resolved else definition.model.model,
            provider=resolved.provider if resolved else definition.model.provider,
            thinking_budget=resolved.thinking_budget if resolved else definition.thinking_budget,
        )
        if on_memo is not None:
            await on_memo(memo)
        return memo

    memos = await asyncio.gather(*(one(AGENT_DEFINITIONS[s]) for s in SCIENTIST_SLUGS))
    return list(memos)


async def run_planner(
    router: ModelRouter,
    *,
    deterministic_evidence: dict,
    memos: list[AgentMemo],
) -> str:
    """Ask the planner for a ``RoundDecisionRecord`` as JSON.

    The contract lives in the prompt (see ``EXPERIMENTAL_PLANNER_SYNTHESIZER_PROMPT``)
    because gateways frequently reject ``json_schema`` and leak the schema text
    into the completion. The pydantic model is still enforced downstream.
    """
    definition = AGENT_DEFINITIONS["planner"]
    # Scientists write freely (markdown tables and all). Cap each memo so the
    # planner's context — and therefore its completion budget — stays bounded.
    memo_cap = 1200
    context = {
        "round_number": deterministic_evidence.get("round_number"),
        "phase": deterministic_evidence.get("phase"),
        "max_joint_uncertainty_candidate": deterministic_evidence.get(
            "max_joint_uncertainty_candidate"
        ),
        "next_uncertainty_candidates": deterministic_evidence.get("next_uncertainty_candidates", []),
        "latest_revealed_candidate_ids": deterministic_evidence.get(
            "latest_revealed_candidate_ids", []
        ),
        # The exact SAR table and derived priors the selector will use for
        # Δ_agent. The planner chooses *which* rule to state; the numbers behind
        # it are deterministic and identical to the acquisition-time values.
        "adaptive_prior_evidence": deterministic_evidence.get("adaptive_prior_evidence", {}),
        "deterministic_evidence": deterministic_evidence,
        "council_memos": [
            {
                "role": memo.role,
                "memo": memo.memo[:memo_cap],
                "truncated": len(memo.memo) > memo_cap,
            }
            for memo in memos
        ],
        "fallback": "{}",
    }
    return await run_agent(router, "planner", context=context, definition=definition)


def thinking_configuration(router: ModelRouter | None = None) -> list[dict]:
    """Audit view: which agents deliberate, and how hard (resolved values)."""
    report = []
    for definition in AGENT_DEFINITIONS.values():
        if router is not None:
            spec = router.effective_spec_for(definition.role) or router.spec_for(definition.role)
        else:
            spec = definition.model
        spec = spec or definition.model
        budget = resolve_thinking_budget(spec)
        effort = resolve_reasoning_effort(spec)
        report.append(
            {
                "role": definition.role,
                "model": spec.model,
                "provider": spec.provider,
                "requested_model": definition.model.model,
                "thinking_budget": budget,
                "extended_thinking": bool(budget),
                "supports_thinking": spec.provider == "anthropic",
                "reasoning_effort": effort,
                "effort_label": (
                    f"{budget} tokens" if budget else (effort or "none")
                ),
            }
        )
    return report
