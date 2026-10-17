"""Async scientist agents; all numerical claims originate from deterministic tools.

Council composition is data, not control flow.  ``AGENT_DEFINITIONS`` binds each
role to (a) the prompt it receives, (b) the deterministic tools it may cite, and
(c) the ``AgentModel`` transport assignment from :mod:`core.llm`.  Thinking
budgets therefore live in exactly one place and can be retuned per agent without
touching orchestration code.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .llm import AGENT_MODELS, AgentModel, ModelRouter, resolve_thinking_budget
from .prompts import (
    AFFINITY_SCIENTIST_PROMPT,
    DEVELOPABILITY_SCIENTIST_PROMPT,
    EXPERIMENTAL_PLANNER_SYNTHESIZER_PROMPT,
    SKEPTIC_CRITIC_PROMPT,
)
from .schemas import AgentMemo, RoundDecisionRecord


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


async def run_scientists(router: ModelRouter, tools: dict) -> list[AgentMemo]:
    """Run the three scientists concurrently (Skeptic joins the same gather)."""

    async def one(definition: AgentDefinition) -> AgentMemo:
        context = {
            "role": definition.role,
            "tools": {name: tools.get(name) for name in definition.tools_invoked},
            "fallback": f"{definition.role} reviewed deterministic evidence; confidence remains bounded.",
        }
        text = await run_agent(router, definition.slug, context=context, definition=definition)
        resolved = router.spec_for(definition.role)
        return AgentMemo(
            role=definition.role,
            tools_invoked=list(definition.tools_invoked),
            thinking_trace=router.thinking_trace(definition.role) or text,
            memo=text,
            model=resolved.model if resolved else definition.model.model,
            provider=resolved.provider if resolved else definition.model.provider,
            thinking_budget=resolved.thinking_budget if resolved else definition.thinking_budget,
        )

    memos = await asyncio.gather(*(one(AGENT_DEFINITIONS[s]) for s in SCIENTIST_SLUGS))
    return list(memos)


async def run_planner(
    router: ModelRouter,
    *,
    deterministic_evidence: dict,
    memos: list[AgentMemo],
) -> str:
    """Ask the OpenAI planner for a ``RoundDecisionRecord`` under a strict schema."""
    definition = AGENT_DEFINITIONS["planner"]
    context = {
        "deterministic_evidence": deterministic_evidence,
        "council_memos": [memo.model_dump() for memo in memos],
        "fallback": "{}",
    }
    return await run_agent(router, "planner", context=context, definition=definition)


def thinking_configuration(router: ModelRouter | None = None) -> list[dict]:
    """Audit view: which agents think, and with what (resolved) budget."""
    report = []
    for definition in AGENT_DEFINITIONS.values():
        if router is not None:
            resolved = router.spec_for(definition.role)
            budget = resolved.thinking_budget if resolved else definition.thinking_budget
        else:
            budget = resolve_thinking_budget(definition.model)
        report.append(
            {
                "role": definition.role,
                "model": definition.model.model,
                "provider": definition.model.provider,
                "thinking_budget": budget,
                "extended_thinking": bool(budget),
                "supports_thinking": definition.model.provider == "anthropic",
            }
        )
    return report
