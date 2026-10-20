"""Stateful closed-loop campaign: surrogates -> agents -> quota -> oracles."""
from __future__ import annotations
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .developability import RiskNormalizer
from .state import empty_state, hcdr3_ribbon
from .features import encode_sequences
from .ledger import ImmutableLedger, is_ledger, read_entries
from .oracle import BindingOracle, DevelopabilityOracle
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions
from .acquisition import sar_effect_table, select_five_slots
from .tool_arsenal import (
    H1_ARM_KEY,
    H1_EXPLOIT_SLOT_IDS,
    active_tools_for_state,
    audit_past_hypotheses,
    compute_cost_and_effort_saved,
    detect_epistasis_clashes,
    inspect_joint_uncertainty_hotspots,
    profile_developability_liabilities,
    scan_positional_sar,
    select_active_tools,
)
from .constants import CHECKPOINT_QUERIES, WT_SEQUENCE
from .virtual_lab_runner import VirtualLabRunner
from config.settings import SETTINGS

#: The revealed-batch schema, in column order. ``revealed`` and every time-travel
#: reconstruction share it, so a round-``k`` frame is column-compatible with the
#: live one by construction.
REVEAL_COLUMNS = ("candidate_id", "sequence", "y_bind", "y_dev", "round_number")


def revealed_from_entries(entries: list[dict]) -> pd.DataFrame:
    """``oracle_reveal`` entries -> revealed batches in **query order**.

    The single implementation behind both :meth:`Campaign.revealed` and the
    round-``k`` reconstruction. Query order matters scientifically — the
    surrogate is an ensemble of bootstrapped trees, so permuting the training
    rows yields different forests — which is why grouping preserves first
    appearance in ledger order (i.e. first reveal) instead of sorting.
    """
    grouped: dict[str, dict] = {}
    for entry in entries:
        if entry.get("event_type") != "oracle_reveal":
            continue
        payload = entry.get("payload") or {}
        objective = payload.get("objective")
        if objective not in {"y_bind", "y_dev"}:
            continue
        candidate_id = str(entry.get("candidate_id") or "")
        if not candidate_id:
            continue
        record = grouped.get(candidate_id)
        if record is None:
            record = grouped[candidate_id] = {
                "candidate_id": candidate_id,
                "sequence": payload.get("sequence"),
                "round_number": int(entry.get("round_number") or 0),
            }
        # Last write wins per objective, matching ``dict(zip(...))`` over the
        # candidate's rows in ledger order.
        record[objective] = payload.get("value")
        record["round_number"] = max(
            record["round_number"], int(entry.get("round_number") or 0)
        )
    rows = [
        record
        for record in grouped.values()
        if record.get("y_bind") is not None and record.get("y_dev") is not None
    ]
    if not rows:
        return pd.DataFrame(columns=list(REVEAL_COLUMNS))
    return pd.DataFrame(rows)[list(REVEAL_COLUMNS)]


def quality_curve_from_revealed(revealed: pd.DataFrame, tau_dev: float) -> list[dict]:
    """One ``query_count -> best qualified binding`` point per completed round.

    Replayed round by round rather than recorded as a single final point: a
    single point would backfill every checkpoint and overstate AUBQC. Counts come
    from the reveals themselves, not ``initial + batch*round``, so a short batch
    cannot shift the curve.
    """
    if revealed.empty:
        return []
    points = []
    for round_index in range(int(revealed.round_number.max()) + 1):
        prefix = revealed[revealed.round_number <= round_index]
        if prefix.empty:
            continue
        qualified = prefix[prefix.y_dev >= tau_dev]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        points.append({"query_count": int(len(prefix)), "best_qualified_binding": round(best, 3)})
    return points


def resolve_reference_binding(revealed: pd.DataFrame, tau_dev: float) -> float | None:
    """The WT anchor ``delta_vs_wt`` is measured against — a pure function.

    Split out of :meth:`Campaign.reference_binding` so a round-``k``
    reconstruction can compute *that round's* reference without touching the live
    campaign's frozen value. Resolution order: the measured wild type when the
    ledger holds one, otherwise the best qualified binding among the round-0 seed
    queries (the same q=5 checkpoint the matched M1-M4 curves start from).
    """
    if revealed.empty:
        return None
    wt = revealed.loc[revealed.sequence == WT_SEQUENCE, "y_bind"]
    if len(wt):
        return float(wt.iloc[0])
    seed = revealed.loc[revealed.round_number == 0] if "round_number" in revealed else revealed
    if seed.empty:
        return None
    qualified = seed[seed.y_dev >= tau_dev]
    pool = qualified if len(qualified) else seed
    return float(pool.y_bind.max())


def curve_value_at(curve: list[dict], checkpoint: int) -> float:
    """``y*(q)`` at one checkpoint, carrying the last observed value forward.

    Mirrors :func:`core.baselines.run_method`'s ``1/30`` normalisation: a
    checkpoint that was never measured inherits the most recent one, and a
    checkpoint *below* the first measurement inherits the first one backward —
    otherwise a campaign whose first record is at q=10 would score 0.0 for q=5
    and understate its own AUBQC.
    """
    recorded = {
        int(point["query_count"]): float(point["best_qualified_binding"])
        for point in curve
        if point.get("best_qualified_binding") is not None
    }
    if not recorded:
        return float("nan")
    counts = sorted(recorded)
    value = recorded[counts[0]]
    for count in counts:
        if count <= checkpoint:
            value = recorded[count]
    return value


def aubqc_from_curve(curve: list[dict]) -> float:
    """``AUBQC`` over :data:`CHECKPOINT_QUERIES`, scaled by the frozen ``1/30``."""
    if not curve:
        return float("nan")
    return float(sum(curve_value_at(curve, q) for q in CHECKPOINT_QUERIES) / 30.0)


@dataclass(frozen=True)
class LedgerSnapshot:
    """Every ledger entry, parsed once, with round-scoped views over it.

    Time travel is a *read* over this snapshot, never a mutation of the database:
    inspecting round 3 while the ledger holds round 6 must not disturb the
    campaign. It is also the reason ``GET /api/rounds`` is cheap — the file is
    read once per ledger version and every round is then derived in memory.
    """

    entries: tuple[dict, ...]
    version: tuple[int, int]

    @classmethod
    def from_ledger(cls, ledger: ImmutableLedger) -> "LedgerSnapshot":
        version = ledger.version()
        entries = []
        for row in read_entries(ledger.path):
            parsed = dict(row)
            try:
                parsed["payload"] = json.loads(row.get("payload") or "{}")
            except (TypeError, json.JSONDecodeError):
                parsed["payload"] = {}
            entries.append(parsed)
        return cls(entries=tuple(entries), version=version)

    # ── Views ────────────────────────────────────────────────────────────────
    def cutoff_for_round(self, round_number: int) -> int:
        """Highest ``sequence_no`` written at or before the end of a round."""
        cutoff = 0
        for entry in self.entries:
            tagged = entry.get("round_number")
            if tagged is None or int(tagged) <= round_number:
                cutoff = max(cutoff, int(entry.get("sequence_no") or 0))
        return cutoff

    def up_to_round(self, round_number: int | None) -> list[dict]:
        if round_number is None:
            return list(self.entries)
        cutoff = self.cutoff_for_round(round_number)
        return [entry for entry in self.entries if int(entry.get("sequence_no") or 0) <= cutoff]

    def revealed(self, round_number: int | None = None) -> pd.DataFrame:
        return revealed_from_entries(self.up_to_round(round_number))

    def decisions(self, round_number: int | None = None) -> list[dict]:
        """``round_decision`` payloads, latest per round, in round order.

        A re-executed round writes a second decision; the newest one is what the
        campaign reports, so a replayed history has to make the same choice.
        """
        latest: dict[int, dict] = {}
        for entry in self.up_to_round(round_number):
            if entry.get("event_type") != "round_decision":
                continue
            payload = entry.get("payload") or {}
            if not payload.get("decision"):
                continue
            latest[int(entry.get("round_number") or 0)] = {
                "round_number": int(entry.get("round_number") or 0), **payload
            }
        return [latest[key] for key in sorted(latest)]

    def last_payload(self, event_type: str, round_number: int | None = None) -> dict | None:
        latest = None
        for entry in self.up_to_round(round_number):
            if entry.get("event_type") == event_type:
                latest = entry.get("payload") or {}
        return latest


class Campaign:
    def __init__(self, private: pd.DataFrame, normalizer: RiskNormalizer, protocol: dict, benchmark: dict | None = None,
                 ledger_path=None, golden_ledger_path=None):
        self.private, self.normalizer, self.protocol, self.benchmark = private, normalizer, protocol, benchmark or {}
        #: Where the frozen, never-written reference run lives. A campaign owns
        #: this path instead of re-reading the global setting on every restore, so
        #: a rehearsal (a test, a scratch sandbox) can point at its own copy and
        #: still be structurally unable to overwrite the stage's only good ledger.
        self.golden_ledger_path = Path(golden_ledger_path or SETTINGS.golden_ledger_path)
        self.state = empty_state(self.benchmark)
        self.hypotheses: list[dict] = []
        #: Query-count -> best qualified binding, so AUBQC can be computed for the
        #: live run exactly the way ``core/baselines.py`` computes it.
        self.quality_curve: list[dict] = []
        #: Frozen reference for ``delta_vs_wt``; resolved once, then reused.
        self._seed_reference: float | None = None
        #: Parsed ledger, memoized on its version; ``None`` until first read.
        self._snapshot: LedgerSnapshot | None = None
        #: ``(ledger version, round, include_tool_outputs)`` -> contract state, so a
        #: UI timeline that steps back and forth pays for each round once and can
        #: never show a memoized state from a *different* ledger.
        self._state_cache: dict[tuple, dict] = {}
        #: ``(ledger version, round)`` -> surrogate-scored unqueried pool, which is
        #: the expensive half of a round's ``tool_outputs`` (~30 ms per round).
        self._scored_cache: dict[tuple, pd.DataFrame] = {}
        self.ledger = ImmutableLedger(ledger_path or SETTINGS.ledger_path, metadata={"m5_policy_mode": SETTINGS.m5_policy_mode, "is_synthetic": protocol["is_synthetic"]})
        self.binding_oracle = BindingOracle(private, self.ledger)
        self.dev_oracle = DevelopabilityOracle(private, self.ledger, normalizer)
        self.runner = VirtualLabRunner(hitl=SETTINGS.hitl)
        self._hydrate_from_ledger()

    # ── Ledger location ───────────────────────────────────────────────────────
    def rebind_ledger(self, path=None) -> None:
        """Point every oracle and the runner at a freshly written ledger file.

        Used by rewind/restore, both of which *replace* the file rather than edit
        it. Rebuilding the oracles is cheap (they hold no state beyond the path and
        the private frame) and it removes the chance that a stale
        ``sqlite3.Connection`` or a stale ``find_reveal`` cache keeps serving the
        pre-rewind history.
        """
        self.ledger = ImmutableLedger(path or self.ledger.path, metadata={"m5_policy_mode": SETTINGS.m5_policy_mode, "is_synthetic": self.protocol["is_synthetic"]})
        self.binding_oracle = BindingOracle(self.private, self.ledger)
        self.dev_oracle = DevelopabilityOracle(self.private, self.ledger, self.normalizer)
        self._rehydrate()
        return None

    def _rehydrate(self) -> None:
        """Rebuild every in-memory round view from the (new) ledger."""
        self.state = empty_state(self.benchmark)
        self.hypotheses = []
        self.quality_curve = []
        self._seed_reference = None
        self._snapshot = None
        self._state_cache = {}
        self._scored_cache = {}
        self.runner.previous_decisions = []
        self.runner.declared_priors_by_round = {}
        self._hydrate_from_ledger()

    # ── Ledger snapshot ───────────────────────────────────────────────────────
    def snapshot(self) -> LedgerSnapshot:
        """The whole ledger, parsed once per version.

        Every historical view is derived from this object, so a single round's
        inspection costs one pass over 88 rows instead of one query per event
        type — and it can never see a half-appended round, because the snapshot is
        keyed on ``(max sequence_no, count)``.
        """
        version = self.ledger.version()
        snapshot = self._snapshot
        if snapshot is None or snapshot.version != version:
            snapshot = self._snapshot = LedgerSnapshot.from_ledger(self.ledger)
        return snapshot

    def current_round(self) -> int:
        """The last round with a reveal — decisions alone do not complete a round."""
        revealed = self.snapshot().revealed()
        return 0 if revealed.empty else int(revealed.round_number.max())

    # ── Restart safety ────────────────────────────────────────────────────────
    def _decision_events(self, round_number: int | None = None) -> list[dict]:
        """``round_decision`` payloads verbatim (payloads are nested JSON)."""
        return self.snapshot().decisions(round_number)

    def _latest_slots(self, round_number: int | None = None) -> list[dict]:
        """The most recently persisted batch design, or ``[]``.

        ``round_slots`` events are written just before the batch is measured, so
        the newest one at or before ``round_number`` is the last batch the loop
        had actually designed by then.
        """
        payload = self.snapshot().last_payload("round_slots", round_number)
        return list((payload or {}).get("slots") or [])

    def _latest_council(self, round_number: int | None = None) -> list[dict]:
        """The council memos of the most recent round, or ``[]`` on a fresh ledger."""
        payload = self.snapshot().last_payload("round_council", round_number)
        return list((payload or {}).get("memos") or [])

    def _decision_records(self, round_number: int | None = None) -> list[dict]:
        """Decisions in the shape the paired audit consumes, latest per round.

        Shared by the live restart path and the round-``k`` reconstruction, so a
        historical audit cannot grade a round differently from the live one: both
        hand :func:`core.tool_arsenal.audit_past_hypotheses` the *measured* ids,
        the whole H1 arm, and the pre-query expectation recorded before the assay.
        """
        records = []
        for event in self._decision_events(round_number):
            decision = event["decision"]
            records.append({
                **decision,
                "round_number": int(event["round_number"]),
                "slot4_measured_id": event.get("slot4_measured_id"),
                "slot5_measured_id": event.get("slot5_measured_id"),
                "declared_h1_id": event.get("declared_h1_id"),
                "declared_h0_id": event.get("declared_h0_id"),
                # The whole H1 exploitation arm (slots 1/2/4 with Δ_agent >= 0),
                # not just slot 4. A restarted process re-audits from this list; if
                # the arm were dropped here it would silently fall back to the
                # slot-4-only comparison this design exists to replace.
                H1_ARM_KEY: event.get(H1_ARM_KEY) or [],
                "pre_query_expectation": event.get("pre_query_expectation") or {},
                # What the council *declared*, kept distinct from the scoring
                # priors the selector was actually driven by.
                "declared_priors": event.get("declared_priors") or [],
            })
        return records

    def _hydrate_from_ledger(self) -> None:
        """Rebuild round history so a resumed process keeps its KPI counters.

        ``scripts/run_campaign.py`` explicitly supports continuing a campaign
        from an existing ledger, but the decision history used to live only in
        memory: the second process started with an empty ``hypotheses`` list,
        re-audited nothing, and printed ``hypotheses supported: 0`` (and
        ``delta vs WT: 0.0``) no matter how many rounds had already been judged.
        Decisions are ledger events now, and the quality curve is replayed from
        the reveal rounds, so every KPI survives a restart.
        """
        for record in self._decision_records():
            round_number = record["round_number"]
            self.runner.previous_decisions.append(record)
            # What the council *declared*, restored so the audit trail survives a
            # restart. The selector is scored on data-derived priors, so this is
            # pure provenance — but it is the provenance a UI needs to show that a
            # planner actually argued something before the arithmetic overruled it.
            if record["declared_priors"]:
                self.runner.declared_priors_by_round[round_number] = list(
                    record["declared_priors"]
                )
            self.hypotheses.append({
                "hypothesis_id": record["hypothesis_id"],
                "round_number": round_number,
                "statement": record.get("hypothesis_statement", ""),
                "verdict": "INCONCLUSIVE",
            })

        # Restore the most recent batch design and council debate so a restarted
        # process reports the same slots (with the same VoI/SAR diagnostics) and
        # the same memos the live run showed.
        latest_slots = self._latest_slots()
        if latest_slots:
            self.state["batch_slots"] = latest_slots
        latest_council = self._latest_council()
        if latest_council:
            self.state["agent_council"] = latest_council

        revealed = self.revealed()
        if revealed.empty:
            self.state["current_round"] = 0
            return

        # Replay the quality curve round by round instead of only recording the
        # final point: a single point would backfill every checkpoint and
        # overstate AUBQC (measured 2.09 vs the true 1.90). Counts come from the
        # ledger rather than ``initial + batch*round`` so a short batch cannot
        # shift the curve.
        last_round = int(revealed.round_number.max())
        self.quality_curve = quality_curve_from_revealed(revealed, self.protocol["tau_dev"])

        # Reveals, not decisions, define a completed round: a decision is written
        # before its batch is measured, so trusting it would mark an interrupted
        # round complete on restart.
        self.state["current_round"] = last_round
        # ``phase`` is a function of the round number, and leaving it at the
        # ``empty_state`` default made a restarted process report "Cold-Start
        # Tuition" for a finished six-round campaign — the API and the live run
        # disagreed about the same ledger.
        self.state["phase"] = "Cold-Start Tuition" if last_round <= 2 else "Self-Accelerating Payoff"
        self._refresh_hypothesis_verdicts()
        qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        reference = self.reference_binding(revealed)
        # The VoI diagnostics are a pure function of the persisted batch design and
        # the revealed ledger, so they are rebuilt here rather than left at their
        # all-zero default. Without this a restarted API reported ``voi = 0`` for a
        # campaign whose final batch genuinely had ``voi == 0`` for a *different*
        # reason, and the UI could not distinguish the two.
        self.state["voi_diagnostics"] = self._voi_diagnostics(last_round, self.state["batch_slots"], revealed)
        # Option C: the council memos restored above are the payload as recorded,
        # which predates the tool-activation schedule. The split is attached here,
        # after the diagnostics exist, so the live hydrated view and the ``state_at``
        # reconstruction carry identical memo objects — ``tests/test_stage_readiness``
        # compares them by exact dict equality.
        if self.state["agent_council"]:
            self.state["agent_council"] = self._annotate_council_memos(
                self.state["agent_council"],
                self.state["voi_diagnostics"]["active_tools_metadata"],
            )
        # ``pareto_data`` was the one contract field this replay left empty, so a
        # restarted API served an empty risk/binding scatter for a campaign that
        # had queried 35 candidates. It is a pure function of the revealed ledger.
        self.state["pareto_data"] = self._pareto_data(revealed)
        # The ribbon is provenance the campaign has already put in the ledger;
        # leaving it at the ``empty_state`` default made a restarted API serve an
        # all-neutral ribbon for a campaign that had argued about — and inverted —
        # real positions. ``declared_priors`` is not added here: the top-level
        # ``/api/state`` contract is frozen (``api_contract.json``) and addition-
        # only is checked by a test, so the richer round view lives behind
        # ``GET /api/state?round=k``.
        self.state["hcdr3_ribbon"] = self._ribbon_for(last_round)
        self.state["kpis"].update({
            "best_qualified_binding": round(best, 2),
            "delta_vs_wt": round(best - reference, 3) if reference is not None else 0.0,
            "query_budget_spent": int(len(revealed)),
            "wet_lab_cost_saved_usd": max(0, (SETTINGS.total_budget - len(revealed)) * SETTINGS.assay_cost_usd),
            "hypotheses_supported": sum(h["verdict"] == "SUPPORTED" for h in self.hypotheses),
            "hypotheses_falsified": sum(h["verdict"] == "FALSIFIED" for h in self.hypotheses),
        })

    def candidates(self) -> list[dict]:
        rows=[]
        for row in self.private.itertuples():
            score=self.normalizer.score(row.sequence)
            rows.append({"candidate_id":row.candidate_id,"sequence":row.sequence,"risk_score":score["risk"],"y_dev":score["y_dev"],"risk_passed":score["y_dev"]>=self.protocol["tau_dev"],"developability_components_raw":score["raw"],"developability_components_normalized":score["normalized"]})
        return rows

    def revealed(self, round_number: int | None = None) -> pd.DataFrame:
        """Revealed batches, in **query order** (first reveal first).

        Order is not cosmetic here. The surrogate is an ensemble of bootstrapped
        trees, so permuting the training rows yields different trees and therefore
        different predictions (measured: 0.15 log10 Kd on a ten-row fit). This
        method used to group by ``candidate_id``, i.e. sort, while
        ``core.baselines.run_method`` feeds its surrogate rows in query order —
        so the live campaign and its own same-seed benchmark twin were training on
        the same rows *in different orders* and silently diverging. Iterating in
        first-reveal order makes the two paths construct identical forests.

        ``round_number`` is an optional *ceiling*: the same rows the live loop
        would have held at the end of that round. It never mutates the ledger, so
        the timeline stepper can show round 2 while the file holds round 6.
        """
        return self.snapshot().revealed(round_number)

    # ── Reference binding for delta_vs_wt ─────────────────────────────────────
    def reference_binding(self, revealed: pd.DataFrame) -> float | None:
        """The WT anchor ``best_qualified_binding`` is measured against.

        Resolution order, highest fidelity first:

        1. the measured wild type itself, when the campaign has queried it;
        2. the **seed baseline** — the best risk-qualified binding among the five
           shared initial queries (round 0). This is the same q=5 checkpoint the
           matched M1-M4 benchmark curves start from, so ``delta_vs_wt`` is
           directly comparable to the benchmark deltas.

        The dataset's own WT label is deliberately *not* read here: SPEC §2
        forbids any ``y_bind`` access outside :class:`BindingOracle`, and the
        official 1N8Z export contains no measured ``SRWGGDGFYAMDY`` row at all
        (see README), so path 1 only fires on synthetic pools.

        Resolved **once** and then frozen: if the wild type were queried in a
        later round, a re-resolving reference would make ``delta_vs_wt`` jump
        without ``best_qualified_binding`` having moved.
        """
        if self._seed_reference is not None:
            return self._seed_reference
        resolved = resolve_reference_binding(revealed, self.protocol["tau_dev"])
        if resolved is not None:
            self._seed_reference = resolved
        return resolved

    @property
    def falsified_positions(self) -> set[int]:
        """Positions whose prior lost its paired H1/H0 test; inverted next round.

        Recomputed from the ledger on every access rather than stored, so it
        stays correct across a process restart and can never drift from the
        verdicts the campaign reports. This is the self-correction path: once a
        rule is refuted, :func:`core.acquisition.adaptive_prior_score` inverts it
        instead of ignoring it, so the agent's vote actively pushes candidates
        carrying that residue down.
        """
        if not self.runner.previous_decisions:
            return set()
        audit = audit_past_hypotheses(self.revealed(), self.runner.previous_decisions)
        return set(audit.get("falsified_positions") or ())

    # ── Round-by-round time travel (read-only) ───────────────────────────────
    def _round_reveals(self, round_number: int) -> pd.DataFrame:
        """Reveals at the end of ``round_number`` (``round_number == -1``: none)."""
        return self.snapshot().revealed(max(-1, round_number))

    def _annotate_council_memos(self, memos: list[dict], metadata: dict) -> list[dict]:
        """Split each memo's ``tools_invoked`` into active and inactive (Option C).

        The recorded list is **not** rewritten. A memo in the golden ledger was
        written before Option C existed, and its ``tools_invoked`` is what that
        council actually cited at the time; overwriting it would make the audit
        trail claim a seat saw a tool it never ran. Instead the round's live
        activation table is attached beside it:

        * ``tools_invoked_active`` — cited tools the drill schedule says are on for
          this round;
        * ``tools_inactive`` — cited tools that are off, so the UI can grey them out
          and explain why rather than silently dropping them;
        * ``tool_activation_reasons`` — the trigger string per cited tool.
        """
        active = set(metadata.get("active") or ())
        reasons = {
            tool["name"]: tool["trigger_reason"]
            for tool in metadata.get("tools") or []
        }
        annotated = []
        for memo in memos:
            entry = dict(memo)
            cited = [str(name) for name in entry.get("tools_invoked") or ()]
            entry["tools_invoked_active"] = [name for name in cited if name in active]
            entry["tools_inactive"] = [name for name in cited if name not in active]
            entry["tool_activation_reasons"] = {
                name: reasons[name] for name in cited if name in reasons
            }
            annotated.append(entry)
        return annotated

    def _round_verdicts(self, round_number: int) -> tuple[list[dict], set[int]]:
        """Hypothesis verdicts and inverted positions **as of** ``round_number``.

        The audit is a pure function of (reveals, decisions), so replaying it over
        the round-``k`` prefix reproduces exactly the verdict the live loop showed
        at the time — including a verdict that later flipped.
        """
        decisions = self._decision_records(round_number)
        if not decisions:
            return [], set()
        revealed = self._round_reveals(round_number)
        audit = audit_past_hypotheses(revealed, decisions)
        by_round = {v.get("round_number"): v for v in audit.get("verdicts", [])}
        hypotheses = []
        for round_index in sorted(by_round):
            record = by_round[round_index]
            hypotheses.append({
                "hypothesis_id": record.get("hypothesis_id"),
                "round_number": round_index,
                "statement": record.get("statement", ""),
                "verdict": record.get("verdict"),
                "evidence": record,
            })
        return hypotheses, set(audit.get("falsified_positions") or ())

    def _pareto_data(self, revealed: pd.DataFrame) -> list[dict]:
        """Risk/binding scatter with the top-10 frontier flagged."""
        if revealed.empty:
            return []
        frontier = set(revealed.sort_values("y_bind", ascending=False).head(10).candidate_id)
        return [
            {
                "candidate_id": row.candidate_id,
                "risk": round(1 - float(row.y_dev), 2),
                "binding": round(float(row.y_bind), 2),
                "status": (
                    "wild_type" if row.candidate_id == "WT"
                    else "pareto_frontier" if row.candidate_id in frontier
                    else f"queried_round_{int(row.round_number)}"
                ),
            }
            for row in revealed.itertuples()
        ]

    def scored_unqueried(self, revealed: pd.DataFrame, *, round_number: int | None = None) -> pd.DataFrame:
        """The unqueried pool with surrogate predictions, fitted on ``revealed``.

        The same construction :meth:`execute_round` performs before the council
        runs, extracted so a historical round can be *re-asked* the questions the
        council was asked: which candidates carry the most joint uncertainty, and
        what does their developability profile look like? Deterministic given the
        revealed rows and the campaign seed, which is what makes the reconstructed
        ``tool_outputs`` evidence rather than decoration.

        Two forests per call is ~30 ms, so the result is memoized per round; a UI
        that flips between timeline entries pays for each round once. ``n_jobs=1``
        is deliberate: the fit is far too small for joblib's process pool, whose
        startup cost is three times the fit itself. Trees are seeded, so the
        predictions are identical either way.
        """
        if round_number is not None:
            key = (self.snapshot().version, round_number)
            cached = self._scored_cache.get(key)
            if cached is not None:
                return cached
        X = encode_sequences(self.private.sequence)
        index = {cid: i for i, cid in enumerate(self.private.candidate_id)}
        train = np.asarray([index[x] for x in revealed.candidate_id])
        unqueried = self.private[~self.private.candidate_id.isin(revealed.candidate_id)].copy()
        pred = np.asarray([index[x] for x in unqueried.candidate_id])
        bs = BindingSurrogate(SETTINGS.random_seed, n_jobs=1).fit(
            X[train], revealed.y_bind.to_numpy()
        )
        ds = DevelopabilitySurrogate(SETTINGS.random_seed + 1, n_jobs=1).fit(
            X[train], revealed.y_dev.to_numpy()
        )
        mb, sb = bs.predict(X[pred])
        md, sd = ds.predict(X[pred])
        linked = link_predictions(mb, sb, md, sd, self.protocol["tau_dev"])
        scored = unqueried.assign(
            y_dev=[self.normalizer.score(s)["y_dev"] for s in unqueried.sequence],
            mu_bind=mb, sigma_bind=sb, mu_dev=md, sigma_dev=sd,
            a_base=linked.a_base, sigma_joint=linked.sigma_joint,
        )
        if round_number is not None:
            self._scored_cache[(self.snapshot().version, round_number)] = scored
        return scored

    def tool_outputs_at(self, round_number: int) -> dict[str, dict]:
        """The deterministic tool tables the council saw when planning ``round``.

        Round ``k``'s debate ran on the ledger **before** that round's own batch
        was measured, so the frame is the reveal prefix through ``k - 1``. That
        distinction is the whole point of the pill: clicking ``tool_outputs`` on
        the round-3 timeline entry must show round 3's *inputs*, not the answer
        the batch later produced.
        """
        inputs = self._round_reveals(round_number - 1)
        decisions = self._decision_records(round_number - 1)
        outputs: dict[str, dict] = {
            "scan_positional_sar": scan_positional_sar(inputs),
            "detect_epistasis_clashes": detect_epistasis_clashes(inputs),
            "audit_past_hypotheses": audit_past_hypotheses(inputs, decisions),
            "compute_cost_and_effort_saved": compute_cost_and_effort_saved(
                inputs, assay_cost_usd=SETTINGS.assay_cost_usd
            ),
        }
        if inputs.empty:
            # The cold start has no surrogate yet, so the two pool-level tools have
            # nothing to inspect. Say so rather than omitting the pill.
            empty = {"tool": "", "rows": []}
            outputs["inspect_joint_uncertainty_hotspots"] = {
                **empty, "tool": "inspect_joint_uncertainty_hotspots",
            }
            outputs["profile_developability_liabilities"] = {
                **empty, "tool": "profile_developability_liabilities",
            }
            return outputs
        try:
            scored = self.scored_unqueried(inputs, round_number=round_number)
            outputs["inspect_joint_uncertainty_hotspots"] = inspect_joint_uncertainty_hotspots(scored)
        except (KeyError, ValueError, IndexError) as exc:
            outputs["inspect_joint_uncertainty_hotspots"] = {
                "tool": "inspect_joint_uncertainty_hotspots", "rows": [], "error": str(exc),
            }
            scored = None
        try:
            ids = (
                list(scored.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True])
                     .candidate_id.head(15))
                if scored is not None
                else list(self.private.candidate_id.head(15))
            )
            outputs["profile_developability_liabilities"] = profile_developability_liabilities(ids)
        except (KeyError, TypeError) as exc:
            outputs["profile_developability_liabilities"] = {
                "tool": "profile_developability_liabilities", "rows": [], "error": str(exc),
            }
        return outputs

    def _ribbon_for(self, round_number: int) -> list[dict]:
        """The per-position ribbon as of ``round_number``.

        Declared priors come from that round's decision payload (what the council
        *argued*), and the inverted set from replaying the paired audit over the
        same prefix — so the live ``/api/state`` and ``/api/state?round=k`` cannot
        disagree about a position's status.
        """
        declared = self._declared_priors_for(round_number)
        _, falsified = self._round_verdicts(round_number)
        return hcdr3_ribbon(declared, falsified)

    def _declared_priors_for(self, round_number: int) -> list[dict]:
        """The priors the council declared for ``round_number``, or ``[]``.

        Falls back to the scoring priors on the decision record for a ledger
        written before ``declared_priors`` was stamped, so an older run still
        renders a meaningful ribbon instead of an all-neutral one.
        """
        decisions = self.snapshot().decisions(round_number)
        if not decisions:
            return []
        latest = decisions[-1]
        declared = latest.get("declared_priors") or []
        if declared:
            return list(declared)
        decision = latest.get("decision") or {}
        return list(decision.get("positional_priors") or [])

    def state_at(self, round_number: int, *, include_tool_outputs: bool = True) -> dict:
        """The complete contract state **as of the end of** ``round_number``.

        Read-only: the database is never touched, so the timeline stepper can
        inspect round 2 while the ledger holds round 6. Every field is rebuilt
        from the ledger prefix through that round — KPIs, VoI/``sar_maturity``
        (read off that round's persisted batch design), the ribbon from that
        round's declared priors and refuted positions, the council memos, the five
        slots, the Pareto scatter, and the deterministic tool tables the council
        was looking at.
        """
        total_rounds = SETTINGS.rounds
        if round_number < 0 or round_number > total_rounds:
            raise ValueError(f"round must be in 0..{total_rounds}")
        current = self.current_round()
        if round_number > current:
            raise ValueError(
                f"round {round_number} has not run yet (ledger holds {current})"
            )
        version = self.snapshot().version
        cache_key = (version, round_number, include_tool_outputs)
        cached = self._state_cache.get(cache_key)
        if cached is not None:
            return cached

        state = empty_state(self.benchmark)
        state["current_round"] = round_number
        state["round_number"] = round_number
        state["phase"] = "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff"
        revealed = self._round_reveals(round_number)
        qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]] if not revealed.empty else revealed
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        reference = resolve_reference_binding(revealed, self.protocol["tau_dev"])
        curve = quality_curve_from_revealed(revealed, self.protocol["tau_dev"])
        hypotheses, _falsified = self._round_verdicts(round_number)
        slots = self._latest_slots(round_number)
        memos = self._latest_council(round_number)
        declared = self._declared_priors_for(round_number)

        state["kpis"] = {
            "best_qualified_binding": round(best, 2),
            "delta_vs_wt": round(best - reference, 3) if reference is not None else 0.0,
            "query_budget_spent": int(len(revealed)),
            "total_budget": SETTINGS.total_budget,
            "wet_lab_cost_saved_usd": max(
                0, (SETTINGS.total_budget - len(revealed)) * SETTINGS.assay_cost_usd
            ),
            "hypotheses_supported": sum(h["verdict"] == "SUPPORTED" for h in hypotheses),
            "hypotheses_falsified": sum(h["verdict"] == "FALSIFIED" for h in hypotheses),
        }
        state["voi_diagnostics"] = self._voi_diagnostics(round_number, slots, revealed)
        state["hcdr3_ribbon"] = self._ribbon_for(round_number)
        state["declared_priors"] = declared
        state["agent_council"] = self._annotate_council_memos(
            memos, state["voi_diagnostics"]["active_tools_metadata"]
        )
        state["batch_slots"] = slots
        state["pareto_data"] = self._pareto_data(revealed)
        state["hypotheses"] = hypotheses
        state["quality_curve"] = curve
        state["aubqc"] = aubqc_from_curve(curve)
        if include_tool_outputs:
            state["tool_outputs"] = self.tool_outputs_at(round_number)
        self._state_cache[cache_key] = state
        return state

    def timeline(self) -> list[dict]:
        """One compact entry per round ``0..current`` for the UI stepper.

        Deliberately cheap (no tool tables, no surrogate refit): it is the payload
        that draws ``[Seed] -> [R1] -> ...``, and the tool pills are fetched for
        whichever round the operator actually clicks.
        """
        current = self.current_round()
        entries = []
        for round_number in range(current + 1):
            state = self.state_at(round_number, include_tool_outputs=False)
            hypotheses = state.get("hypotheses") or []
            last = hypotheses[-1] if hypotheses else {}
            slots = state.get("batch_slots") or []
            kpis = state["kpis"]
            diagnostics = state["voi_diagnostics"]
            active_metadata = diagnostics.get("active_tools_metadata") or {}
            entries.append({
                "round": round_number,
                "label": "Seed" if round_number == 0 else f"R{round_number}",
                "phase": state["phase"],
                "query_count": kpis["query_budget_spent"],
                "best_qualified_binding": kpis["best_qualified_binding"],
                "delta_vs_wt": kpis["delta_vs_wt"],
                "voi": diagnostics["voi"],
                "sar_maturity": diagnostics["sar_maturity"],
                "exploit_weight": diagnostics["exploit_weight"],
                "hypotheses_supported": kpis["hypotheses_supported"],
                "hypotheses_falsified": kpis["hypotheses_falsified"],
                "hypothesis_id": last.get("hypothesis_id"),
                "verdict": last.get("verdict"),
                "statement": last.get("statement"),
                "council_seats": len(state.get("agent_council") or []),
                "slots": len(slots),
                "slot_candidate_ids": [slot.get("candidate_id") for slot in slots],
                # Option C: the compact per-round tool summary, so the stepper can
                # show a drill switching on/off between two adjacent entries without
                # fetching either round's full state.
                "active_tools": list(active_metadata.get("active") or []),
                "inactive_tools": [
                    tool["name"] for tool in active_metadata.get("tools") or []
                    if not tool["active"]
                ],
                "tool_tiers": {
                    tool["name"]: tool["tier"]
                    for tool in active_metadata.get("tools") or []
                },
            })
        return entries

    # ── Rewind / restore ─────────────────────────────────────────────────────
    def restore_golden(self) -> dict:
        """Replace ``results/campaign.sqlite3`` with the frozen Seed-21 ledger.

        The golden file is **read**, never written: it is the only copy of the
        verified run, and everything here is designed so that no live round and no
        rewind can cost the stage its campaign. The restore is a single atomic
        ``os.replace`` of a byte copy, so a crash mid-restore cannot leave a
        half-file behind.
        """
        golden = self.golden_ledger_path
        if not golden.exists():
            raise FileNotFoundError(
                f"golden ledger {golden} is missing; restore it from the verified run first"
            )
        if not is_ledger(golden):
            raise ValueError(f"golden ledger {golden} is not a readable ledger")
        destination = Path(self.ledger.path)
        staging = destination.with_name(f"{destination.name}.restore.tmp")
        shutil.copyfile(golden, staging)
        if not is_ledger(staging):
            staging.unlink(missing_ok=True)
            raise ValueError(f"staging copy of {golden} did not open as a ledger")
        shutil.move(str(staging), str(destination))
        self.rebind_ledger(destination)
        return {
            "restored": True,
            "source": str(golden),
            "ledger": str(destination),
            "current_round": self.current_round(),
            "entries": self.snapshot().version[1],
            "verified": self.ledger.verify(),
        }

    def rewind_to(self, round_number: int) -> dict:
        """Truncate the live ledger to the end of ``round_number``.

        If the live ledger is shorter than the requested round (typical after a
        botched live round), the golden ledger is restored first — otherwise
        "rewind to R5" would silently do nothing on a ledger that only holds R2.

        The truncation itself is structural, not a read-time filter: the file on
        disk becomes the shorter one, so the next :meth:`execute_round` appends to
        the truncated chain exactly as if the later rounds had never happened.
        """
        total_rounds = SETTINGS.rounds
        if round_number < 0 or round_number > total_rounds:
            raise ValueError(f"round must be in 0..{total_rounds}")
        restored_from_golden = False
        if self.current_round() < round_number:
            golden = self.golden_ledger_path
            if not golden.exists():
                raise ValueError(
                    f"ledger holds round {self.current_round()}, cannot rewind forward to "
                    f"{round_number} and no golden ledger is available"
                )
            self.restore_golden()
            restored_from_golden = True
            if self.current_round() < round_number:
                raise ValueError(
                    f"golden ledger only holds round {self.current_round()}, "
                    f"cannot rewind to {round_number}"
                )
        truncation = self.ledger.truncate_to_round(round_number)
        self.rebind_ledger(self.ledger.path)
        state = self.state_at(round_number, include_tool_outputs=False)
        return {
            "rewound": True,
            "round": round_number,
            "restored_from_golden": restored_from_golden,
            "ledger": str(self.ledger.path),
            "verified": self.ledger.verify(),
            "entries_dropped": truncation["entries_dropped"],
            "entries_remaining": truncation["entries_after"],
            "current_round": self.current_round(),
            "queries_spent": state["kpis"]["query_budget_spent"],
            "best_qualified_binding": state["kpis"]["best_qualified_binding"],
            "voi": state["voi_diagnostics"]["voi"],
            "state": state,
        }

    def _record_quality(self, revealed: pd.DataFrame) -> None:
        """Record one ``query_count -> best qualified binding`` point per budget level.

        The live campaign spends in batches (5, 10, ...), so every level maps to
        exactly one checkpoint of the matched benchmark curve.
        """
        qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        best = round(float(qualified.y_bind.max()) if len(qualified) else 0.0, 3)
        count = int(len(revealed))
        self.quality_curve = [point for point in self.quality_curve if int(point["query_count"]) != count]
        self.quality_curve.append({"query_count": count, "best_qualified_binding": best})
        self.quality_curve.sort(key=lambda point: int(point["query_count"]))

    def aubqc(self) -> float:
        """Live AUBQC over ``q = 5, 10, ..., 35``, scaled by the frozen ``1/30``.

        Mirrors :func:`core.baselines.run_method` (which uses the identical
        ``1/30`` normalisation over seven checkpoints) through the shared
        :func:`aubqc_from_curve`, so the live number and the same-seed benchmark
        row are computed by one function rather than two that must agree.
        """
        return aubqc_from_curve(self.quality_curve)

    async def execute_round(self) -> dict:
        # Reset rather than replace: replacing the bus would silently drop the
        # synchronous listeners a CLI attaches to watch the round live.
        self.runner.events.reset()
        revealed = self.revealed()
        if revealed.empty:
            # ``SETTINGS.random_seed`` is the benchmark seed this campaign is the
            # same-seed twin of: ``run_benchmark`` draws the identical five
            # initial ids with ``np.random.default_rng(seed)``, so the report can
            # compare like-for-like instead of against a seed *median*.
            initial = np.random.default_rng(SETTINGS.random_seed).choice(self.private.candidate_id.to_numpy(), SETTINGS.initial_queries, replace=False).tolist()
            self.binding_oracle.query(initial, round_number=0, actor="seed_protocol", batch_tag="initial")
            self.dev_oracle.query(initial, round_number=0, actor="seed_protocol", batch_tag="initial")
            revealed = self.revealed()
        # The seed batch is the q=5 checkpoint every matched method shares, so it
        # must be recorded before the first round's reveals move the curve.
        self._record_quality(revealed)
        round_number = int(self.state["current_round"]) + 1
        if round_number > SETTINGS.rounds:
            raise RuntimeError("Campaign already completed")
        X = encode_sequences(self.private.sequence)
        index = {cid: i for i, cid in enumerate(self.private.candidate_id)}
        train = np.asarray([index[x] for x in revealed.candidate_id])
        unqueried = self.private[~self.private.candidate_id.isin(revealed.candidate_id)].copy()
        # ``y_dev`` is a sequence-only proxy (SPEC §2: available at t=0, no label
        # leak) and ``select_five_slots`` gates on it through
        # :func:`core.acquisition.qualified_pool`. That gate silently no-ops when
        # the column is absent, so the live selector was choosing from the *whole*
        # pool while the benchmark — whose work frame always carries ``y_dev`` —
        # chose from the qualified 210. Same portfolio, different candidate set:
        # that is why live M5 diverged from benchmark M5 on the same seed, and it
        # was invisible because neither path errors when the column is missing.
        unqueried = unqueried.assign(
            y_dev=[self.normalizer.score(sequence)["y_dev"] for sequence in unqueried.sequence]
        )
        pred = np.asarray([index[x] for x in unqueried.candidate_id])
        # Surrogate RNG keyed to the campaign seed, not the round number.
        # ``run_benchmark`` calls ``BindingSurrogate(seed)`` /
        # ``DevelopabilitySurrogate(seed + 1)``; keying the live loop on
        # ``round_number`` instead built different ExtraTrees on the *same*
        # revealed rows, so the live campaign was not the same-seed twin of the
        # benchmark row it is reported against — it was an independent draw that
        # merely shared five initial queries. With this key, seed 21's live run and
        # benchmark M5 see identical forests, which is what makes the same-seed
        # comparison in ``scripts/run_campaign.py`` meaningful.
        bs = BindingSurrogate(SETTINGS.random_seed).fit(X[train], revealed.y_bind.to_numpy())
        ds = DevelopabilitySurrogate(SETTINGS.random_seed + 1).fit(X[train], revealed.y_dev.to_numpy())
        mb, sb = bs.predict(X[pred]); md, sd = ds.predict(X[pred])
        linked = link_predictions(mb, sb, md, sd, self.protocol["tau_dev"])
        unqueried = unqueried.assign(mu_bind=mb, sigma_bind=sb, mu_dev=md, sigma_dev=sd,
                                     a_base=linked.a_base, sigma_joint=linked.sigma_joint)
        tau_dev = self.protocol["tau_dev"]

        # Surrogate residuals for the revealed rows drive the SAR term: the
        # surrogate already absorbs additive multi-mutation structure, so only a
        # *surprising* candidate says anything about its individual substitutions.
        mu_revealed, _ = bs.predict(X[train])
        residuals = {
            str(cid): float(y) - float(mu)
            for cid, y, mu in zip(revealed.candidate_id, revealed.y_bind, mu_revealed)
        }
        falsified = self.falsified_positions

        decision, memos = await self.runner.plan_round(
            round_number, revealed, unqueried,
            tau_dev=tau_dev, residuals=residuals, falsified_positions=falsified,
        )
        slots = select_five_slots(
            unqueried, decision,
            ledger=revealed,
            tau_dev=tau_dev,
            residuals=residuals,
            falsified_positions=falsified,
            queries_spent=int(len(revealed)),
            total_budget=SETTINGS.total_budget,
            batch_size=SETTINGS.batch_size,
        )
        ids = [s["candidate_id"] for s in slots]
        # Stamp the decision with what the quota *actually* measures, plus the
        # pre-query surrogate expectation, so the audit tool has a decidable
        # baseline even when the planner's declared targets were consumed by
        # slots 1-3 of this same batch. Slot 4/5 ids are rewritten to the picked
        # candidates, so the audit can never reference a rejected id.
        self._annotate_decision(decision, slots, dict(zip(
            unqueried.candidate_id.astype(str), unqueried.mu_bind.astype(float))))
        for slot in slots: await self.runner.events.emit("slot_selected", slot)
        # Persist the batch design *before* the reveals. ``state['batch_slots']``
        # lives only in memory, so without this event a restarted process would
        # present an empty batch and the VoI/SAR diagnostics behind the five picks
        # — the whole point of the annealed objective — would be lost.
        self.ledger.append(
            "round_slots",
            {"slots": [self._slot_diagnostics(s) for s in slots]},
            actor="virtual_lab",
            round_number=round_number,
            batch_tag=f"round-{round_number}",
        )
        # The council memos are the *provenance* of the declared priors, and they
        # were previously in-memory only: a restarted API served
        # ``agent_council: []`` for a campaign that had run all six rounds with
        # four live models. Persisting them keeps the debate on the record.
        self.ledger.append(
            "round_council",
            {"memos": [memo.model_dump() for memo in memos]},
            actor="virtual_lab",
            round_number=round_number,
            batch_tag=f"round-{round_number}",
        )
        bind = self.binding_oracle.query(ids, round_number=round_number, actor="virtual_lab", batch_tag=f"round-{round_number}")
        dev = self.dev_oracle.query(ids, round_number=round_number, actor="virtual_lab", batch_tag=f"round-{round_number}")
        for b, d in zip(bind, dev): await self.runner.events.emit("oracle_reveal", {"candidate_id": b["candidate_id"], "y_bind": b["value"], "y_dev": d["value"]})
        self.hypotheses.append({"hypothesis_id": decision.hypothesis_id, "round_number": round_number, "statement": decision.hypothesis_statement, "verdict": "INCONCLUSIVE"})
        self._refresh_hypothesis_verdicts()
        self._refresh_state(round_number, slots, memos)
        await self.runner.events.emit("round_complete", {"state": self.state, "round_summary": {"queried": ids}})
        await self.runner.events.emit("stream_end", {"ok": True})
        return self.state

    async def run_autonomous(self) -> dict:
        while self.state["current_round"] < SETTINGS.rounds:
            await self.execute_round()
        return self.state

    def _annotate_decision(self, decision, slots: list[dict], expectations: dict[str, float]) -> None:
        """Reconcile the decision with what the quota actually queried.

        ``VirtualLabRunner.plan_round`` appends the planner's decision *before*
        the quota runs. The selector then re-derives Slots 1-2 (top ``A_H1``),
        Slot 3 (VoI exploration), Slot 4 (third ``A_H1`` pick) and Slot 5 (the H0
        control), so the declared targets can be replaced or consumed by Slots
        1-3. Two things follow, and both are fixed here:

        * the audit needs the ids that were **measured**, not the declared ones,
          otherwise every historical verdict is ``INCONCLUSIVE``;
        * the plan shown to the operator must not claim a Slot 4 target (e.g. a
          low-scoring ``CAND-312``) that was never queried. The recorded decision
          is rewritten to the selected ids, so the ledger, the audit and the
          printed round summary all agree.

        The **whole H1 exploitation arm** is stamped, not just Slot 4. Slots 1, 2
        and 4 all rank by ``A_H1``, so slot 4 is structurally the *fourth* choice;
        grading only that leftover against the control stamped ``FALSIFIED`` in
        rounds where slot 1 or 2 found the elite binder, and the resulting
        inversion flipped the verdict back the next round. The arm excludes any
        exploitation pick whose ``Δ_agent`` is not positive, because a candidate
        the hypothesis did not boost cannot be evidence *for* it.
        """
        by_type = {slot["slot_type"]: str(slot["candidate_id"]) for slot in slots}
        slot3 = by_type.get("Max Joint Uncertainty")
        slot4 = by_type.get("Agent Hypothesis Test")
        slot5 = by_type.get("Skeptic Falsification Probe")
        # The pair the planner *claimed*. It is snapped into the prospective batch
        # by ``_reconcile_decision``, so by construction both members are measured
        # in this round — which is what lets the audit grade the claim it was
        # actually given rather than the selector's leftover.
        declared_h1 = str(decision.slot4_hypothesis_target_id)
        declared_h0 = str(decision.slot5_falsification_target_id)
        # The council's own declaration, kept as provenance now that the selector
        # is scored on the data-derived set. It goes into the ledger, not just
        # memory: ``declared_priors_by_round`` is rebuilt from these payloads on
        # restart, so ``/api/state`` can show what the planner argued for even
        # after the process that heard it has exited.
        declared = self.runner.declared_priors_by_round.get(decision.round_number) or []
        # The H1 arm is what the hypothesis actually bought: the exploitation
        # picks (slots 1, 2, 4) whose Δ_agent is non-negative. Slot 3 is the VoI
        # exploration probe and slot 5 is the control, so neither is in the arm.
        arm = [
            str(slot["candidate_id"])
            for slot in slots
            if int(slot.get("slot_id", 0)) in H1_EXPLOIT_SLOT_IDS
            and float(slot.get("delta_agent", 0.0) or 0.0) >= 0.0
        ]
        # Re-validate (not ``model_copy``) so the contract's ``unique_targets``
        # invariant is still enforced on the reconciled record.
        reconciled = type(decision).model_validate({
            **decision.model_dump(),
            **{key: value for key, value in (
                ("slot3_uncertainty_target_id", slot3),
                ("slot4_hypothesis_target_id", slot4),
                ("slot5_falsification_target_id", slot5),
            ) if value},
        })
        measured = {
            "slot4_measured_id": slot4,
            "slot5_measured_id": slot5,
            H1_ARM_KEY: arm,
            # The claimed comparison pair, kept distinct from the slot-4 leftover
            # that ``_annotate_decision`` rewrites for the operator's benefit.
            "declared_h1_id": declared_h1,
            "declared_h0_id": declared_h0,
            "pre_query_expectation": {cid: round(float(mu), 3) for cid, mu in expectations.items()},
            "declared_priors": declared,
        }
        for entry in reversed(self.runner.previous_decisions):
            if entry.get("round_number") == decision.round_number:
                entry.update(reconciled.model_dump())
                entry.update(measured)
                self.ledger.append(
                    "round_decision",
                    {"decision": reconciled.model_dump(), **measured},
                    actor="virtual_lab",
                    round_number=decision.round_number,
                    batch_tag=f"round-{decision.round_number}",
                )
                return

    def _refresh_hypothesis_verdicts(self) -> None:
        """Re-derive every verdict from the ledger; KPI counters read these strings.

        Auditing *all* rounds each time (rather than only the round that just
        closed) keeps the counters monotone and lets a verdict flip once the
        falsification probe is measured.
        """
        audit = audit_past_hypotheses(self.revealed(), self.runner.previous_decisions)
        verdicts = {v.get("round_number"): v for v in audit.get("verdicts", [])}
        for entry in self.hypotheses:
            record = verdicts.get(entry["round_number"])
            if record is None:
                continue
            entry["verdict"] = record["verdict"]
            entry["evidence"] = record

    def _queries_before_round(self, round_number: int) -> int:
        """Queries the council had seen when it designed ``round_number``.

        The **planning prefix**, not the round's own spend: round ``k`` was designed
        from the reveals through ``k - 1``. Feeding the Option C selector the
        round's own spend would open ``detect_epistasis_clashes`` in R1 (10 queries)
        instead of R2, which is exactly the off-by-one-batch the drill's
        ``queries_spent >= 10`` trigger is written against.
        """
        if round_number < 1:
            return 0
        return int(len(self._round_reveals(round_number - 1)))

    @staticmethod
    def _delta_agent_spread(slots: list[dict]) -> float | None:
        """Spread of the batch's measured ``Δ_agent``, or ``None`` if unavailable.

        ``slot_graduation`` takes this as its "has the prior stopped moving?"
        input. Returning ``None`` rather than ``0.0`` when no slot carries the
        field matters: ``0.0`` would read as *fully converged* and mark any batch
        graduated, so a missing measurement would silently switch the slot law.
        """
        values = [
            float(slot["delta_agent"])
            for slot in slots
            if slot.get("delta_agent") is not None
        ]
        if len(values) < 2:
            return None
        return float(max(values) - min(values))

    def _voi_diagnostics(self, round_number: int, slots: list[dict], revealed: pd.DataFrame) -> dict:
        """Batch-level VoI state plus the position-level SAR that fed ``Δ_agent``.

        Every slot in one batch shares the same ``VoI``/``horizon``/``sar_maturity``
        (they are computed once from the qualified pool), so the batch-level view
        simply reads them off slot 1. ``exploit_weight`` is spelled out because it
        *is* the architecture change: 0.35 in Round 1, 1.0 by the final batch.

        ``active_tools_metadata`` (Option C) is derived from the **already-rounded**
        VoI and SAR maturity this dict exports, never the raw floats. The live path
        and the ``state_at`` reconstruction must agree field for field —
        ``tests/test_stage_readiness.py`` compares the two by exact dict equality —
        and a raw ``voi`` of ``-1e-17`` on the final batch would deactivate the
        hotspot drill live while the reconstruction, reading ``0.0`` off the
        persisted batch design, deactivated it via the terminal-harvest branch.
        """
        from .acquisition import VOI_EXPLOIT_WEIGHT, slot_graduation
        from .state import voi_objective_provenance

        lead = slots[0] if slots else {}
        voi = float(lead.get("voi", 0.0) or 0.0)
        sar = sar_effect_table(revealed, tau_dev=self.protocol["tau_dev"])
        position_effects = getattr(sar, "position_effects", {}) or {}
        sar_maturity = round(float(lead.get("sar_maturity", 0.0) or 0.0), 4)
        exported_voi = round(voi, 4)
        # ``eta_sar`` / ``sar_maturity_target`` come off the *same* campaign
        # protocol the selector used, so the badge and the picks cannot disagree.
        # ``slot_graduation`` is read straight from ``core.acquisition`` with this
        # campaign's own round count, and the spread is taken from the batch's
        # measured ``Δ_agent`` — the identical inputs the selector saw.
        provenance = voi_objective_provenance(
            voi=exported_voi, sar_maturity=sar_maturity, protocol=self.protocol
        )
        spread = self._delta_agent_spread(slots)
        if spread is not None and self.protocol is not None:
            provenance["slot_graduation"] = bool(
                slot_graduation(
                    round_number,
                    spread,
                    total_rounds=int(self.protocol.get("rounds", 6)),
                )
            )
        return {
            "round_number": round_number,
            "voi": exported_voi,
            "horizon": round(float(lead.get("horizon", 0.0) or 0.0), 4),
            "sar_maturity": sar_maturity,
            "exploit_weight": round(1.0 - VOI_EXPLOIT_WEIGHT * voi, 4),
            "exploration_weight": round(VOI_EXPLOIT_WEIGHT * voi, 4),
            "slot_split": "3-exploit / 2-experimental",
            "sar_basis": sar.basis,
            "sar_support": int(sar.support),
            "position_tolerance_effects": [
                {"position": position, "n": int(n), "shrunk_effect": round(float(effect), 4)}
                for position, (n, effect) in sorted(position_effects.items())
            ],
            **provenance,
            "active_tools_metadata": active_tools_for_state(
                round_number=round_number,
                queries_spent=self._queries_before_round(round_number),
                voi=exported_voi,
                sar_maturity=sar_maturity,
            ),
        }

    def _refresh_state(self, round_number: int, slots: list[dict], memos: list) -> None:
        revealed = self.revealed(); qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        reference = self.reference_binding(revealed)
        # ``best_qualified_binding`` climbs live, so a 2-dp delta would quantise
        # visible movement away; 3 dp keeps ``delta_vs_wt`` tracking it.
        delta = round(best - reference, 3) if reference is not None else 0.0
        self._record_quality(revealed)
        diagnostics = self._voi_diagnostics(round_number, slots, revealed)
        self.state.update({
            "current_round": round_number,
            "phase": "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff",
            "voi_diagnostics": diagnostics,
            # The ribbon is the *declared* provenance of this batch's priors and
            # the audit's refuted positions, read back from the ledger entry
            # ``_annotate_decision`` has just written. Built by the same helper as
            # ``state_at``, so the live view and the timeline entry cannot drift.
            "hcdr3_ribbon": self._ribbon_for(round_number),
            "kpis": {
                "best_qualified_binding": round(best, 2),
                "delta_vs_wt": delta,
                "query_budget_spent": int(len(revealed)),
                "total_budget": SETTINGS.total_budget,
                "wet_lab_cost_saved_usd": max(0, (SETTINGS.total_budget - len(revealed)) * SETTINGS.assay_cost_usd),
                "hypotheses_supported": sum(h["verdict"] == "SUPPORTED" for h in self.hypotheses),
                "hypotheses_falsified": sum(h["verdict"] == "FALSIFIED" for h in self.hypotheses),
            },
            "agent_council": self._annotate_council_memos(
                [m.model_dump() for m in memos],
                diagnostics["active_tools_metadata"],
            ),
            "batch_slots": [self._slot_diagnostics(s) for s in slots],
            "pareto_data": self._pareto_data(revealed),
        })

    def _slot_diagnostics(self, slot: dict) -> dict:
        """One batch slot as the API reports it, including the VoI/SAR diagnostics.

        The additions are all *inside* the existing ``batch_slots`` objects, so the
        top-level ``/api/state`` contract keys are unchanged. They matter because
        the VoI-annealed objective and the 3-exploit split are only legible if the
        UI can show *why* each slot was chosen: which arm it belongs to
        (``a_null`` vs ``a_h1``), how much of the decision was exploration
        (``voi``/``horizon``/``sar_maturity``), and what the agent prior contributed
        (``delta_agent``). Without them a judge sees five candidate ids and no
        evidence that the architecture changed.
        """
        score = self.normalizer.score(slot["sequence"])
        return {
            "slot_id": slot["slot_id"],
            "slot_type": slot["slot_type"],
            "candidate_id": slot["candidate_id"],
            "sequence": slot["sequence"],
            "mu_bind": round(float(slot["mu_bind"]), 2),
            "sigma_bind": round(float(slot["sigma_bind"]), 2),
            "risk_score": round(float(score["risk"]), 2),
            "risk_passed": bool(score["y_dev"] >= self.protocol["tau_dev"]),
            # ── Paired H1/H0 arms and the VoI-annealed objective ──────────────
            "a_null": round(float(slot["a_null"]), 4) if "a_null" in slot else None,
            "a_h1": round(float(slot["a_h1"]), 4) if "a_h1" in slot else None,
            "delta_agent": round(float(slot["delta_agent"]), 4) if "delta_agent" in slot else None,
            "sigma_joint": round(float(slot["sigma_joint"]), 4) if "sigma_joint" in slot else None,
            # The two selection scores that are not simply ``a_h1``:
            # ``optimistic`` is slot 3's VoI-weighted exploration score, and
            # ``agent_promoted`` is slot 4's (now identical, boost retired).
            "optimistic": round(float(slot["optimistic"]), 4) if "optimistic" in slot else None,
            "agent_promoted": (
                round(float(slot["agent_promoted"]), 4) if "agent_promoted" in slot else None
            ),
            "voi": round(float(slot["voi"]), 4) if "voi" in slot else None,
            "horizon": round(float(slot["horizon"]), 4) if "horizon" in slot else None,
            "sar_maturity": round(float(slot["sar_maturity"]), 4) if "sar_maturity" in slot else None,
        }
