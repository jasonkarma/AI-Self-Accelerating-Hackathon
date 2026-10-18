"""Tool-first async virtual-lab round orchestration with optional HITL gate."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
import pandas as pd

from .agents import AGENT_DEFINITIONS, run_planner, run_scientists
from .llm import DeterministicReasoner, ModelRouter
from .schemas import AgentMemo, PositionalSoftPrior, RoundDecisionRecord
from .tool_arsenal import (
    audit_past_hypotheses,
    detect_epistasis_clashes,
    inspect_joint_uncertainty_hotspots,
    profile_developability_liabilities,
    scan_positional_sar,
)


@dataclass
class RoundEventBus:
    """Fan-out for round progress.

    Supports both the async-consumer style used by the SSE endpoint and a plain
    synchronous listener, which is what the CLI uses to print as work happens.
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)

    def subscribe_sync(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Register a synchronous callback, invoked on every ``emit``."""
        self._listeners.append(listener)

    def reset(self) -> None:
        """Start a new round without discarding existing subscribers/listeners."""
        self.history.clear()

    async def emit(self, event: str, data: Any) -> None:
        message = {"event": event, "data": data}
        self.history.append(message)
        for listener in tuple(self._listeners):
            listener(message)
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


def _deterministic_decision(
    round_number: int, ledger: pd.DataFrame, ranked: list[str], tools: list[str]
) -> RoundDecisionRecord:
    """Frozen contract-satisfying decision, used when the planner is offline."""
    return RoundDecisionRecord(
        round_number=round_number,
        phase="Cold-Start Tuition (Exploration)" if round_number <= 2 else "Self-Accelerating Payoff (Exploitation)",
        tools_invoked=list(tools),
        hypothesis_id=f"HYP-R{round_number}-01",
        hypothesis_statement="Conserve W3 while exploring lower-liability substitutions.",
        previous_hypothesis_verdict="N/A (Round 1)" if round_number == 1 else "INCONCLUSIVE",
        supporting_candidate_ids=ledger.candidate_id.astype(str).head(3).tolist() if not ledger.empty else [],
        agent_disagreement_summary="Skeptic reserves a direct falsification slot.",
        confidence_rating="Low" if len(ledger) < 15 else "Medium",
        positional_priors=[PositionalSoftPrior(position=3, preferred_residues=["W"], discouraged_residues=[], weight_delta=.15)],
        slot3_uncertainty_target_id=ranked[0],
        slot4_hypothesis_target_id=ranked[1],
        slot5_falsification_target_id=ranked[2],
    )


def _strip_fences(raw: str) -> str:
    """Planner models often wrap JSON in ```json fences despite instructions."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Models wrap output in fences or add a sentence of preamble regardless of
    instructions, so scan for the first balanced ``{...}`` instead of trusting
    that the whole reply is JSON.
    """
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("planner returned no JSON object")
    depth, in_string, escaped = 0, False, False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("planner JSON object was not terminated")


#: Slot fields in contract order; used for duplicate repair and reconciliation.
SLOT_FIELDS = (
    "slot3_uncertainty_target_id",
    "slot4_hypothesis_target_id",
    "slot5_falsification_target_id",
)


def _parse_decision(raw: str, ranked: list[str]) -> RoundDecisionRecord:
    """Parse planner output, repairing duplicate slots before validation.

    Pointing two slots at the same candidate is a mundane model mistake, not an
    invalid decision, and the contract's ``unique_targets`` validator would
    otherwise throw the whole generation away.
    """
    payload = _extract_json(raw)
    seen: set[str] = set()
    spare = iter(cid for cid in ranked)
    for field in SLOT_FIELDS:
        value = payload.get(field)
        if value in seen or value not in ranked:
            value = next((cid for cid in spare if cid not in seen), None)
        if value is None:
            raise ValueError(f"no unqueried candidate available for {field}")
        payload[field] = value
        seen.add(value)
    return RoundDecisionRecord.model_validate(payload)


def _reconcile_decision(
    decision: RoundDecisionRecord,
    *,
    round_number: int,
    phase: str,
    ranked: list[str],
    tools: list[str],
) -> RoundDecisionRecord:
    """Force the model's output onto the frozen contract's hard invariants.

    Slot 3 must be the max joint-uncertainty candidate and slots 3-5 must be
    distinct; everything else the planner decided is preserved.
    """
    if len(ranked) < 3:
        raise ValueError("Planner requires at least three unqueried candidates")

    def next_unused(used: set[str], preferred: str) -> str:
        # Only candidates the acquisition step can actually score are eligible.
        if preferred in ranked and preferred not in used:
            return preferred
        return next(cid for cid in ranked if cid not in used)

    slot3 = ranked[0]
    used = {slot3}
    slot4 = next_unused(used, decision.slot4_hypothesis_target_id)
    used.add(slot4)
    slot5 = next_unused(used, decision.slot5_falsification_target_id)

    # Re-validate instead of ``model_copy(update=...)``: the latter skips
    # validation, so a bad reconciliation would reach the acquisition step.
    return RoundDecisionRecord.model_validate(
        {
            **decision.model_dump(),
            "round_number": round_number,
            "phase": phase,
            "tools_invoked": list(tools),
            "slot3_uncertainty_target_id": slot3,
            "slot4_hypothesis_target_id": slot4,
            "slot5_falsification_target_id": slot5,
        }
    )


class VirtualLabRunner:
    def __init__(self, router: ModelRouter | None = None, *, hitl: bool = False):
        self.router, self.hitl, self.events = router or ModelRouter(), hitl, RoundEventBus()
        self._approval = asyncio.Event()
        self.previous_decisions: list[dict] = []

    def approve(self) -> None: self._approval.set()

    def _developability_tool(self, unqueried: pd.DataFrame) -> dict:
        """Profile the highest-uncertainty candidates instead of returning an empty stub."""
        ids = unqueried.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.head(15).astype(str).tolist()
        try:
            return profile_developability_liabilities(ids)
        except (KeyError, TypeError) as exc:
            return {"tool": "profile_developability_liabilities", "rows": [], "error": str(exc)}

    async def plan_round(self, round_number: int, ledger: pd.DataFrame, unqueried: pd.DataFrame) -> tuple[RoundDecisionRecord, list[AgentMemo]]:
        phase_short = "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff"
        phase_exact = "Cold-Start Tuition (Exploration)" if round_number <= 2 else "Self-Accelerating Payoff (Exploitation)"
        await self.events.emit("round_started", {"round": round_number, "phase": phase_short})

        tools = {
            "scan_positional_sar": scan_positional_sar(ledger),
            "detect_epistasis_clashes": detect_epistasis_clashes(ledger),
            "inspect_joint_uncertainty_hotspots": inspect_joint_uncertainty_hotspots(unqueried),
            "audit_past_hypotheses": audit_past_hypotheses(ledger, self.previous_decisions),
            "profile_developability_liabilities": self._developability_tool(unqueried),
        }
        for name, output in tools.items():
            await self.events.emit("tool_output", {"tool": name, "output": output})

        ranked = unqueried.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.tolist()

        async def stream_memo(memo: AgentMemo) -> None:
            await self.events.emit("agent_memo", memo.model_dump())

        async def report_error(role: str, exc: BaseException) -> None:
            await self.events.emit(
                "agent_error", {"role": role, "error": f"{type(exc).__name__}: {exc}"}
            )

        memos = await run_scientists(
            self.router, tools, on_memo=stream_memo, on_error=report_error
        )

        decision = _deterministic_decision(round_number, ledger, ranked, list(tools))

        if self.router.is_live(AGENT_DEFINITIONS["planner"].role):
            await self.events.emit("planner_started", {"role": AGENT_DEFINITIONS["planner"].role})
            try:
                raw = await run_planner(
                    self.router,
                    deterministic_evidence={
                        "round_number": round_number,
                        "phase": phase_exact,
                        "tools": tools,
                        "max_joint_uncertainty_candidate": ranked[0],
                        "next_uncertainty_candidates": ranked[1:6],
                        "latest_revealed_candidate_ids": (
                            ledger.candidate_id.astype(str).tolist() if not ledger.empty else []
                        ),
                    },
                    memos=memos,
                )
                parsed = _reconcile_decision(
                    _parse_decision(raw, ranked),
                    round_number=round_number,
                    phase=phase_exact,
                    ranked=ranked,
                    tools=list(tools),
                )
                decision = parsed
                await self.events.emit("planner_reconciled", {"note": "slot 3 pinned to max joint uncertainty"})
            except Exception as exc:  # noqa: BLE001 - never lose a round to a bad generation
                await self.events.emit(
                    "planner_error",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "fallback": "deterministic decision",
                        "raw_output": (getattr(self.router.client_for(AGENT_DEFINITIONS["planner"].role), "last_raw", None) or "")[:600],
                    },
                )

        await self.events.emit("planner_synthesis", decision.model_dump())
        if self.hitl:
            await self.events.emit("batch_pending_approval", {"decision": decision.model_dump()})
            await self._approval.wait()
            self._approval.clear()
        self.previous_decisions.append(decision.model_dump())
        return decision, memos
