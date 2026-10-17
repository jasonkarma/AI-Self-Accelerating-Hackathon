"""Async scientist agents; all numerical claims originate from deterministic tools."""
from __future__ import annotations
import asyncio
from .llm import ModelRouter
from .prompts import AFFINITY_SCIENTIST_PROMPT, DEVELOPABILITY_SCIENTIST_PROMPT, SKEPTIC_CRITIC_PROMPT
from .schemas import AgentMemo

async def run_scientists(router: ModelRouter, tools: dict) -> list[AgentMemo]:
    async def one(role, prompt, names):
        text = await router.run(role, prompt, {"tools": {n: tools.get(n) for n in names}, "fallback": f"{role} reviewed deterministic evidence; confidence remains bounded."})
        return AgentMemo(role=role, tools_invoked=names, thinking_trace=text, memo=text)
    affinity, dev = await asyncio.gather(one("Affinity Scientist", AFFINITY_SCIENTIST_PROMPT, ["scan_positional_sar", "detect_epistasis_clashes"]), one("Developability Scientist", DEVELOPABILITY_SCIENTIST_PROMPT, ["profile_developability_liabilities"]))
    skeptic = await one("Scientific Skeptic", SKEPTIC_CRITIC_PROMPT, ["audit_past_hypotheses", "inspect_joint_uncertainty_hotspots"])
    return [affinity, dev, skeptic]
