"""Stateful closed-loop campaign: surrogates -> agents -> quota -> oracles."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .developability import RiskNormalizer
from .state import empty_state
from .features import encode_sequences
from .ledger import ImmutableLedger
from .oracle import BindingOracle, DevelopabilityOracle
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions
from .acquisition import select_five_slots
from .tool_arsenal import audit_past_hypotheses
from .constants import WT_SEQUENCE
from .virtual_lab_runner import VirtualLabRunner
from config.settings import SETTINGS


class Campaign:
    def __init__(self, private: pd.DataFrame, normalizer: RiskNormalizer, protocol: dict, benchmark: dict | None = None,
                 ledger_path=None):
        self.private, self.normalizer, self.protocol, self.benchmark = private, normalizer, protocol, benchmark or {}
        self.state = empty_state(self.benchmark)
        self.hypotheses: list[dict] = []
        #: Query-count -> best qualified binding, so AUBQC can be computed for the
        #: live run exactly the way ``core/baselines.py`` computes it.
        self.quality_curve: list[dict] = []
        #: Frozen reference for ``delta_vs_wt``; resolved once, then reused.
        self._seed_reference: float | None = None
        self.ledger = ImmutableLedger(ledger_path or SETTINGS.ledger_path, metadata={"m5_policy_mode": SETTINGS.m5_policy_mode, "is_synthetic": protocol["is_synthetic"]})
        self.binding_oracle = BindingOracle(private, self.ledger)
        self.dev_oracle = DevelopabilityOracle(private, self.ledger, normalizer)
        self.runner = VirtualLabRunner(hitl=SETTINGS.hitl)
        self._hydrate_from_ledger()

    # ── Restart safety ────────────────────────────────────────────────────────
    def _decision_events(self) -> list[dict]:
        """Read ``round_decision`` payloads verbatim (payloads are nested JSON)."""
        with self.ledger.connect() as con:
            rows = con.execute(
                "SELECT round_number, payload FROM entries WHERE event_type='round_decision' "
                "ORDER BY sequence_no"
            ).fetchall()
        events = []
        for row in rows:
            payload = json.loads(row["payload"])
            if payload.get("decision"):
                events.append({"round_number": int(row["round_number"]), **payload})
        return events

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
        # A re-executed round writes a second decision; keep the latest per round.
        latest: dict[int, dict] = {}
        for event in self._decision_events():
            latest[int(event["round_number"])] = event
        for round_number in sorted(latest):
            event = latest[round_number]
            decision = event["decision"]
            self.runner.previous_decisions.append({
                **decision,
                "slot4_measured_id": event.get("slot4_measured_id"),
                "slot5_measured_id": event.get("slot5_measured_id"),
                "pre_query_expectation": event.get("pre_query_expectation") or {},
            })
            self.hypotheses.append({
                "hypothesis_id": decision["hypothesis_id"],
                "round_number": round_number,
                "statement": decision.get("hypothesis_statement", ""),
                "verdict": "INCONCLUSIVE",
            })

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
        self.quality_curve = []
        for round_index in range(last_round + 1):
            prefix = revealed[revealed.round_number <= round_index]
            if prefix.empty:
                continue
            qualified = prefix[prefix.y_dev >= self.protocol["tau_dev"]]
            best = float(qualified.y_bind.max()) if len(qualified) else 0.0
            self.quality_curve.append({
                "query_count": int(len(prefix)),
                "best_qualified_binding": round(best, 3),
            })

        # Reveals, not decisions, define a completed round: a decision is written
        # before its batch is measured, so trusting it would mark an interrupted
        # round complete on restart.
        self.state["current_round"] = last_round
        self._refresh_hypothesis_verdicts()
        qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        reference = self.reference_binding(revealed)
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

    def revealed(self) -> pd.DataFrame:
        events = self.ledger.dataframe("oracle_reveal")
        if events.empty:
            return pd.DataFrame(columns=["candidate_id", "sequence", "y_bind", "y_dev", "round_number"])
        rows = []
        for cid, group in events.groupby("candidate_id"):
            values = dict(zip(group.objective, group.value))
            if {"y_bind", "y_dev"} <= values.keys():
                rows.append({"candidate_id": cid, "sequence": group.sequence.iloc[0], "y_bind": values["y_bind"], "y_dev": values["y_dev"], "round_number": int(group.round_number.max())})
        return pd.DataFrame(rows)

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
        if revealed.empty:
            return None
        wt = revealed.loc[revealed.sequence == WT_SEQUENCE, "y_bind"]
        if len(wt):
            self._seed_reference = float(wt.iloc[0])
            return self._seed_reference
        seed = revealed.loc[revealed.round_number == 0] if "round_number" in revealed else revealed
        if seed.empty:
            return None
        qualified = seed[seed.y_dev >= self.protocol["tau_dev"]]
        pool = qualified if len(qualified) else seed
        self._seed_reference = float(pool.y_bind.max())
        return self._seed_reference

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
        ``1/30`` normalisation over seven checkpoints): unmeasured checkpoints
        carry the last observed value forward, and a checkpoint *below* the first
        measurement carries the first one **backward** — otherwise a campaign
        whose first record is at q=10 would score 0.0 for q=5 and understate its
        own AUBQC.
        """
        if not self.quality_curve:
            return float("nan")
        recorded = {int(point["query_count"]): float(point["best_qualified_binding"]) for point in self.quality_curve}
        counts = sorted(recorded)
        values, last = [], recorded[counts[0]]
        for checkpoint in range(5, 36, 5):
            for count in counts:
                if count <= checkpoint:
                    last = recorded[count]
            values.append(last)
        return float(sum(values) / 30.0)

    async def execute_round(self) -> dict:
        # Reset rather than replace: replacing the bus would silently drop the
        # synchronous listeners a CLI attaches to watch the round live.
        self.runner.events.reset()
        revealed = self.revealed()
        if revealed.empty:
            initial = np.random.default_rng(0).choice(self.private.candidate_id.to_numpy(), SETTINGS.initial_queries, replace=False).tolist()
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
        pred = np.asarray([index[x] for x in unqueried.candidate_id])
        bs = BindingSurrogate(round_number).fit(X[train], revealed.y_bind.to_numpy())
        ds = DevelopabilitySurrogate(round_number + 100).fit(X[train], revealed.y_dev.to_numpy())
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
        the quota runs. The selector then re-derives Slot 4 (max ``a_agent`` among
        ``Δ_agent > 0``) and Slot 5 (frontier falsification probe), so the
        declared targets can be replaced or consumed by Slots 1-3. Two things
        follow, and both are fixed here:

        * the audit needs the ids that were **measured**, not the declared ones,
          otherwise every historical verdict is ``INCONCLUSIVE``;
        * the plan shown to the operator must not claim a Slot 4 target (e.g. a
          low-scoring ``CAND-312``) that was never queried. The recorded decision
          is rewritten to the selected ids, so the ledger, the audit and the
          printed round summary all agree.
        """
        by_type = {slot["slot_type"]: str(slot["candidate_id"]) for slot in slots}
        slot3 = by_type.get("Max Joint Uncertainty")
        slot4 = by_type.get("Agent Hypothesis Test")
        slot5 = by_type.get("Skeptic Falsification Probe")
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
            "pre_query_expectation": {cid: round(float(mu), 3) for cid, mu in expectations.items()},
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

    def _refresh_state(self, round_number: int, slots: list[dict], memos: list) -> None:
        revealed = self.revealed(); qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        reference = self.reference_binding(revealed)
        # ``best_qualified_binding`` climbs live, so a 2-dp delta would quantise
        # visible movement away; 3 dp keeps ``delta_vs_wt`` tracking it.
        delta = round(best - reference, 3) if reference is not None else 0.0
        self._record_quality(revealed)
        frontier = revealed.sort_values("y_bind", ascending=False).head(10)
        self.state.update({"current_round": round_number, "phase": "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff",
            "kpis": {"best_qualified_binding": round(best,2), "delta_vs_wt": delta, "query_budget_spent": int(len(revealed)), "total_budget": SETTINGS.total_budget, "wet_lab_cost_saved_usd": max(0,(SETTINGS.total_budget-len(revealed))*SETTINGS.assay_cost_usd), "hypotheses_supported": sum(h["verdict"]=="SUPPORTED" for h in self.hypotheses), "hypotheses_falsified": sum(h["verdict"]=="FALSIFIED" for h in self.hypotheses)},
            "agent_council": [m.model_dump() for m in memos],
            "batch_slots": [{"slot_id":s["slot_id"],"slot_type":s["slot_type"],"candidate_id":s["candidate_id"],"sequence":s["sequence"],"mu_bind":round(float(s["mu_bind"]),2),"sigma_bind":round(float(s["sigma_bind"]),2),"risk_score":round(float(self.normalizer.score(s["sequence"])["risk"]),2),"risk_passed":bool(self.normalizer.score(s["sequence"])["y_dev"]>=self.protocol["tau_dev"])} for s in slots],
            "pareto_data": [{"candidate_id":r.candidate_id,"risk":round(1-float(r.y_dev),2),"binding":round(float(r.y_bind),2),"status":"wild_type" if r.candidate_id=="WT" else ("pareto_frontier" if r.candidate_id in set(frontier.candidate_id) else f"queried_round_{int(r.round_number)}")} for r in revealed.itertuples()]})
