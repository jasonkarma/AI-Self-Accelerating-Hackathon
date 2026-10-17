"""Tool-first async virtual-lab round orchestration with optional HITL gate."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
import pandas as pd

from .agents import run_scientists
from .llm import ModelRouter
from .schemas import PositionalSoftPrior, RoundDecisionRecord
from .tool_arsenal import scan_positional_sar, detect_epistasis_clashes, inspect_joint_uncertainty_hotspots, audit_past_hypotheses


@dataclass
class RoundEventBus:
    history: list[dict[str, Any]] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)

    async def emit(self, event: str, data: Any) -> None:
        message = {"event": event, "data": data}
        self.history.append(message)
        for queue in self._subscribers:
            await queue.put(message)

    async def subscribe(self, replay: bool = True) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            if replay:
                for item in self.history:
                    yield item
            while True:
                item = await queue.get()
                yield item
                if item["event"] == "stream_end":
                    return
        finally:
            self._subscribers.remove(queue)


class VirtualLabRunner:
    def __init__(self, router: ModelRouter | None = None, *, hitl: bool = False):
        self.router, self.hitl, self.events = router or ModelRouter(), hitl, RoundEventBus()
        self._approval = asyncio.Event()
        self.previous_decisions: list[dict] = []

    def approve(self) -> None: self._approval.set()

    async def plan_round(self, round_number: int, ledger: pd.DataFrame, unqueried: pd.DataFrame) -> tuple[RoundDecisionRecord, list]:
        await self.events.emit("round_started", {"round": round_number, "phase": "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff"})
        tools = {
            "scan_positional_sar": scan_positional_sar(ledger),
            "detect_epistasis_clashes": detect_epistasis_clashes(ledger),
            "inspect_joint_uncertainty_hotspots": inspect_joint_uncertainty_hotspots(unqueried),
            "audit_past_hypotheses": audit_past_hypotheses(ledger, self.previous_decisions),
            "profile_developability_liabilities": {"tool": "profile_developability_liabilities", "rows": []},
        }
        for name, output in tools.items(): await self.events.emit("tool_output", {"tool": name, "output": output})
        memos = await run_scientists(self.router, tools)
        for memo in memos: await self.events.emit("agent_memo", memo.model_dump())
        ranked = unqueried.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.tolist()
        decision = RoundDecisionRecord(round_number=round_number, phase="Cold-Start Tuition (Exploration)" if round_number <= 2 else "Self-Accelerating Payoff (Exploitation)", tools_invoked=list(tools), hypothesis_id=f"HYP-R{round_number}-01", hypothesis_statement="Conserve W3 while exploring lower-liability substitutions.", previous_hypothesis_verdict="N/A (Round 1)" if round_number == 1 else "INCONCLUSIVE", supporting_candidate_ids=ledger.candidate_id.astype(str).head(3).tolist() if not ledger.empty else [], agent_disagreement_summary="Skeptic reserves a direct falsification slot.", confidence_rating="Low" if len(ledger) < 15 else "Medium", positional_priors=[PositionalSoftPrior(position=3, preferred_residues=["W"], discouraged_residues=[], weight_delta=.15)], slot3_uncertainty_target_id=ranked[0], slot4_hypothesis_target_id=ranked[1], slot5_falsification_target_id=ranked[2])
        await self.events.emit("planner_synthesis", decision.model_dump())
        if self.hitl:
            await self.events.emit("batch_pending_approval", {"decision": decision.model_dump()})
            await self._approval.wait(); self._approval.clear()
        self.previous_decisions.append(decision.model_dump())
        return decision, memos
