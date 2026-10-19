"""Tool-first async virtual-lab round orchestration with optional HITL gate."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
import pandas as pd

from .acquisition import (
    prospective_slots,
    qualified_pool,
    sar_effect_table,
    score_pool,
)
from config.settings import SETTINGS
from .agents import (
    AGENT_DEFINITIONS,
    adaptive_prior_evidence,
    deterministic_priors,
    run_planner,
    run_scientists,
)
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
    round_number: int, ledger: pd.DataFrame, ranked: list[str], tools: list[str],
    *, priors: list[PositionalSoftPrior] | None = None,
    prospective: list[dict] | None = None,
) -> RoundDecisionRecord:
    """Frozen contract-satisfying decision, used when the planner is offline.

    The priors are derived from the revealed ledger (top-``k`` consensus
    substitutions) instead of a hardcoded ``W3`` rule, so the offline fallback
    speaks the same language as :func:`core.acquisition.adaptive_prior_score`.

    When a ``prospective`` batch is supplied, the statement names the *actual*
    H1/H0 pair from it. That matters even offline: the fallback used to say
    "conserve W3" — a claim about a position, not about the two candidates being
    measured — so the audit could never check it. Now the fallback states the
    comparison the batch will run, which is the only claim the data can settle.
    ``ranked`` (ordered by ``sigma_joint``) remains the ordering for slots 3-5 when
    no prospective batch is available.
    """
    if prospective:
        by_slot = {int(slot["slot_id"]): str(slot["candidate_id"]) for slot in prospective}
        h1 = by_slot.get(4) or by_slot.get(1) or ranked[1]
        h0 = by_slot.get(5) or ranked[2]
        statement = (
            f"H1 candidate {h1} will out-measure the H0 control {h0} on this batch's "
            f"binding assay."
        )
        supporting = [str(slot["candidate_id"]) for slot in prospective]
    else:
        h1, h0 = ranked[1], ranked[2]
        statement = "Conserve W3 while exploring lower-liability substitutions."
        supporting = ledger.candidate_id.astype(str).head(3).tolist() if not ledger.empty else []

    return RoundDecisionRecord(
        round_number=round_number,
        phase="Cold-Start Tuition (Exploration)" if round_number <= 2 else "Self-Accelerating Payoff (Exploitation)",
        tools_invoked=list(tools),
        hypothesis_id=f"HYP-R{round_number}-01",
        hypothesis_statement=statement,
        previous_hypothesis_verdict="N/A (Round 1)" if round_number == 1 else "INCONCLUSIVE",
        supporting_candidate_ids=supporting,
        agent_disagreement_summary="Skeptic reserves a direct falsification slot.",
        confidence_rating="Low" if len(ledger) < 15 else "Medium",
        positional_priors=priors or [PositionalSoftPrior(position=3, preferred_residues=["W"], discouraged_residues=[], weight_delta=.15)],
        # Slot 3 is the contract's max-joint-uncertainty probe on *every* path —
        # ``_reconcile_decision`` enforces the same invariant for the live planner.
        # The selector's VoI-exploration pick is a different, later decision, and
        # ``Campaign._annotate_decision`` rewrites the recorded target to whatever
        # the quota actually measured.
        slot3_uncertainty_target_id=ranked[0],
        slot4_hypothesis_target_id=h1,
        slot5_falsification_target_id=h0,
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
    h1_candidates: list[str] | None = None,
    h0_candidates: list[str] | None = None,
) -> RoundDecisionRecord:
    """Force the model's output onto the frozen contract's hard invariants.

    Slot 3 must be the max joint-uncertainty candidate and slots 3-5 must be
    distinct; everything else the planner decided is preserved — *except* the
    comparison pair, which is snapped into the prospective batch.

    Why the pair is constrained at all: the audit grades whatever the round
    measured, so a planner naming candidates outside this batch produced a
    hypothesis that could never be tested. Its claim and its grade then referred
    to disjoint candidate sets, and the verdict was arithmetically correct and
    scientifically empty. ``h1_candidates`` / ``h0_candidates`` are the slots the
    selector is about to fill; the planner still chooses *which* of them its
    hypothesis contrasts, and still writes the statement.
    """
    if len(ranked) < 3:
        raise ValueError("Planner requires at least three unqueried candidates")

    def next_unused(used: set[str], preferred: str, pool: list[str] | None = None) -> str:
        candidates = pool or ranked
        # Only candidates the acquisition step can actually score are eligible.
        if preferred in candidates and preferred not in used:
            return preferred
        return next(cid for cid in candidates if cid not in used)

    slot3 = ranked[0]
    used = {slot3}
    slot4 = next_unused(used, decision.slot4_hypothesis_target_id, h1_candidates)
    used.add(slot4)
    slot5 = next_unused(used, decision.slot5_falsification_target_id, h0_candidates)

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
        #: What each round's council *declared*, kept for provenance now that the
        #: selector is scored on data-derived priors. Keyed by round number.
        self.declared_priors_by_round: dict[int, list[dict]] = {}

    def approve(self) -> None: self._approval.set()

    def _developability_tool(self, unqueried: pd.DataFrame) -> dict:
        """Profile the highest-uncertainty candidates instead of returning an empty stub."""
        ids = unqueried.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.head(15).astype(str).tolist()
        try:
            return profile_developability_liabilities(ids)
        except (KeyError, TypeError) as exc:
            return {"tool": "profile_developability_liabilities", "rows": [], "error": str(exc)}

    async def plan_round(self, round_number: int, ledger: pd.DataFrame, unqueried: pd.DataFrame,
                         *, tau_dev: float | None = None,
                         residuals: dict[str, float] | None = None,
                         falsified_positions: set[int] | frozenset[int] = frozenset(),
                         ) -> tuple[RoundDecisionRecord, list[AgentMemo]]:
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

        # Score the pool with the *selector's own* objective before the council
        # sees it. The shortlist used to be ranked by ``sigma_joint`` (the
        # exploration axis) while ``select_five_slots`` ranked by ``A_H1`` and
        # ``A_null``, so the planner could name a comparison pair that the
        # selector would never measure — and its hypothesis was then graded on
        # candidates it had never mentioned. Feeding the planner the prospective
        # batch instead of a differently-ranked list is the fix: the claim is
        # written about candidates that are actually entering this round.
        base_priors = deterministic_priors(ledger, tau_dev, falsified_positions=falsified_positions)
        scored, voi = score_pool(
            unqueried,
            sar_effects=sar_effect_table(ledger, tau_dev=tau_dev, residuals=residuals),
            priors=base_priors,
            ledger=ledger,
            tau_dev=tau_dev,
            residuals=residuals,
            falsified_positions=falsified_positions,
            queries_spent=int(len(ledger)) or 5,
            total_budget=SETTINGS.total_budget,
            batch_size=SETTINGS.batch_size,
        )
        prospective = prospective_slots(scored)
        eligible = qualified_pool(unqueried, tau_dev)
        ranked = eligible.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.tolist()
        if len(ranked) < 3:
            # Never let the gate starve the quota: fall back to the full pool.
            eligible = unqueried
            ranked = unqueried.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).candidate_id.tolist()
        # A planner may only name candidates the quota can actually measure: the
        # prospective batch first, then the rest of the scored pool ordered by the
        # same objective, then the raw uncertainty ranking as a last resort.
        allowed = [str(slot["candidate_id"]) for slot in prospective]
        for cid in scored.sort_values(["a_h1", "candidate_id"], ascending=[False, True]).candidate_id.astype(str):
            if cid not in allowed:
                allowed.append(cid)
        for cid in ranked:
            if str(cid) not in allowed:
                allowed.append(str(cid))

        async def stream_memo(memo: AgentMemo) -> None:
            await self.events.emit("agent_memo", memo.model_dump())

        async def report_error(role: str, exc: BaseException) -> None:
            await self.events.emit(
                "agent_error", {"role": role, "error": f"{type(exc).__name__}: {exc}"}
            )

        memos = await run_scientists(
            self.router, tools, on_memo=stream_memo, on_error=report_error
        )

        # The exact Σ-priors + empirical-SAR numbers behind Δ_agent, reused for
        # the fallback decision and the planner's context so the LLM sees the same
        # arithmetic the selector will apply.
        derived_priors = deterministic_priors(ledger, tau_dev, falsified_positions=falsified_positions)
        sample_sequences = (
            eligible.sequence.astype(str).head(5).tolist() if "sequence" in eligible.columns else []
        )
        prior_evidence = adaptive_prior_evidence(
            ledger, tau_dev,
            priors=derived_priors,
            sequences=sample_sequences,
            residuals=residuals,
            falsified_positions=falsified_positions,
        )
        decision = _deterministic_decision(
            round_number, ledger, ranked, list(tools),
            priors=derived_priors,
            prospective=prospective,
        )

        # The comparison pair the planner must grade itself against: the H1
        # hypothesis candidate and the H0 control it will be tested against. Both
        # come from the prospective batch, so a claim written about them is a claim
        # about candidates this round actually measures.
        h1_candidates = [
            str(slot["candidate_id"])
            for slot in prospective
            if int(slot["slot_id"]) in (1, 2, 4)
        ]
        h0_candidates = [
            str(slot["candidate_id"])
            for slot in prospective
            if int(slot["slot_id"]) == 5
        ]

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
                        "candidate_sequences": (
                            {str(row.candidate_id): str(row.sequence) for row in eligible.head(12).itertuples()}
                            if "sequence" in eligible.columns else {}
                        ),
                        # The batch this round will actually measure, with the role
                        # each slot plays. The planner's claim must be about these.
                        "prospective_batch": [
                            {
                                "slot_id": int(slot["slot_id"]),
                                "slot_type": slot["slot_type"],
                                "candidate_id": str(slot["candidate_id"]),
                                "sequence": str(slot.get("sequence", "")),
                                "a_null": round(float(slot.get("a_null", 0.0)), 4),
                                "a_h1": round(float(slot.get("a_h1", 0.0)), 4),
                                "delta_agent": round(float(slot.get("delta_agent", 0.0)), 4),
                                "mu_bind": round(float(slot.get("mu_bind", 0.0)), 3),
                            }
                            for slot in prospective
                        ],
                        "h1_hypothesis_candidates": h1_candidates,
                        "h0_control_candidates": h0_candidates,
                        "voi_context": {key: round(float(value), 4) for key, value in voi.items()},
                        "surrogate_residuals": {
                            cid: round(value, 3) for cid, value in (residuals or {}).items()
                        },
                        "falsified_positions": sorted(int(p) for p in falsified_positions),
                        "adaptive_prior_evidence": prior_evidence,
                        "latest_revealed_candidate_ids": (
                            ledger.candidate_id.astype(str).tolist() if not ledger.empty else []
                        ),
                    },
                    memos=memos,
                )
                parsed = _reconcile_decision(
                    _parse_decision(raw, allowed),
                    round_number=round_number,
                    phase=phase_exact,
                    ranked=ranked,
                    tools=list(tools),
                    h1_candidates=h1_candidates,
                    h0_candidates=h0_candidates,
                )
                # Anchor the *scoring* priors on the same data-derived set the
                # benchmark's M5 uses, and keep the council's own declaration as
                # provenance.
                #
                # Without this, `live_llm` M5 and benchmark M5 score the same pool
                # with different `Δ_agent`: the benchmark derives its priors from
                # the revealed ledger (so they agree with the empirical SAR term by
                # construction), while a live council invents six free-form rules
                # from prose. That difference is not a scientific disagreement, it
                # is a different scoring function reaching `select_five_slots`, and
                # it is why a live run could trail its own deterministic twin on the
                # same seed. The council's memos, thinking traces, tool citations
                # and hypothesis statements are all still recorded verbatim in
                # `agent_council` / `hypotheses` — only the numeric weights that
                # reach the selector are anchored.
                self.declared_priors_by_round[round_number] = [
                    prior.model_dump() for prior in parsed.positional_priors
                ]
                decision = type(parsed).model_validate({
                    **parsed.model_dump(),
                    "positional_priors": [prior.model_dump() for prior in derived_priors],
                })
                await self.events.emit("planner_reconciled", {
                    "note": "slot 3 pinned to max joint uncertainty; scoring priors anchored on data-derived set",
                    "declared_priors": self.declared_priors_by_round[round_number],
                    "scoring_priors": [prior.model_dump() for prior in derived_priors],
                })
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
