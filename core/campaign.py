"""Stateful closed-loop campaign: surrogates -> agents -> quota -> oracles."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .developability import RiskNormalizer
from .state import empty_state
from .features import encode_sequences
from .ledger import ImmutableLedger
from .oracle import BindingOracle, DevelopabilityOracle
from .surrogate import BindingSurrogate, DevelopabilitySurrogate, link_predictions
from .acquisition import select_five_slots
from .virtual_lab_runner import VirtualLabRunner, RoundEventBus
from config.settings import SETTINGS

class Campaign:
    def __init__(self, private: pd.DataFrame, normalizer: RiskNormalizer, protocol: dict, benchmark: dict | None = None,
                 ledger_path=None):
        self.private, self.normalizer, self.protocol, self.benchmark = private, normalizer, protocol, benchmark or {}
        self.state = empty_state(self.benchmark)
        self.hypotheses: list[dict] = []
        self.ledger = ImmutableLedger(ledger_path or SETTINGS.ledger_path, metadata={"m5_policy_mode": SETTINGS.m5_policy_mode, "is_synthetic": protocol["is_synthetic"]})
        self.binding_oracle = BindingOracle(private, self.ledger)
        self.dev_oracle = DevelopabilityOracle(private, self.ledger, normalizer)
        self.runner = VirtualLabRunner(hitl=SETTINGS.hitl)
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

    async def execute_round(self) -> dict:
        revealed = self.revealed()
        if revealed.empty:
            initial = np.random.default_rng(0).choice(self.private.candidate_id.to_numpy(), SETTINGS.initial_queries, replace=False).tolist()
            self.binding_oracle.query(initial, round_number=0, actor="seed_protocol", batch_tag="initial")
            self.dev_oracle.query(initial, round_number=0, actor="seed_protocol", batch_tag="initial")
            revealed = self.revealed()
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
        self.runner.events = RoundEventBus()
        decision, memos = await self.runner.plan_round(round_number, revealed, unqueried)
        slots = select_five_slots(unqueried, decision)
        ids = [s["candidate_id"] for s in slots]
        for slot in slots: await self.runner.events.emit("slot_selected", slot)
        bind = self.binding_oracle.query(ids, round_number=round_number, actor="virtual_lab", batch_tag=f"round-{round_number}")
        dev = self.dev_oracle.query(ids, round_number=round_number, actor="virtual_lab", batch_tag=f"round-{round_number}")
        for b, d in zip(bind, dev): await self.runner.events.emit("oracle_reveal", {"candidate_id": b["candidate_id"], "y_bind": b["value"], "y_dev": d["value"]})
        self.hypotheses.append({"hypothesis_id": decision.hypothesis_id, "round_number": round_number, "statement": decision.hypothesis_statement, "verdict": "INCONCLUSIVE"})
        self._refresh_state(round_number, slots, memos)
        await self.runner.events.emit("round_complete", {"state": self.state, "round_summary": {"queried": ids}})
        await self.runner.events.emit("stream_end", {"ok": True})
        return self.state

    async def run_autonomous(self) -> dict:
        while self.state["current_round"] < SETTINGS.rounds:
            await self.execute_round()
        return self.state

    def _refresh_state(self, round_number: int, slots: list[dict], memos: list) -> None:
        revealed = self.revealed(); qualified = revealed[revealed.y_dev >= self.protocol["tau_dev"]]
        wt = revealed.loc[revealed.candidate_id == "WT", "y_bind"]
        best = float(qualified.y_bind.max()) if len(qualified) else 0.0
        wt_value = float(wt.iloc[0]) if len(wt) else 0.0
        frontier = revealed.sort_values("y_bind", ascending=False).head(10)
        self.state.update({"current_round": round_number, "phase": "Cold-Start Tuition" if round_number <= 2 else "Self-Accelerating Payoff",
            "kpis": {"best_qualified_binding": round(best,2), "delta_vs_wt": round(best-wt_value,2) if wt_value else 0.0, "query_budget_spent": int(len(revealed)), "total_budget": SETTINGS.total_budget, "wet_lab_cost_saved_usd": max(0,(SETTINGS.total_budget-len(revealed))*SETTINGS.assay_cost_usd), "hypotheses_supported": sum(h["verdict"]=="SUPPORTED" for h in self.hypotheses), "hypotheses_falsified": sum(h["verdict"]=="FALSIFIED" for h in self.hypotheses)},
            "agent_council": [m.model_dump() for m in memos],
            "batch_slots": [{"slot_id":s["slot_id"],"slot_type":s["slot_type"],"candidate_id":s["candidate_id"],"sequence":s["sequence"],"mu_bind":round(float(s["mu_bind"]),2),"sigma_bind":round(float(s["sigma_bind"]),2),"risk_score":round(float(self.normalizer.score(s["sequence"])["risk"]),2),"risk_passed":bool(self.normalizer.score(s["sequence"])["y_dev"]>=self.protocol["tau_dev"])} for s in slots],
            "pareto_data": [{"candidate_id":r.candidate_id,"risk":round(1-float(r.y_dev),2),"binding":round(float(r.y_bind),2),"status":"wild_type" if r.candidate_id=="WT" else ("pareto_frontier" if r.candidate_id in set(frontier.candidate_id) else f"queried_round_{int(r.round_number)}")} for r in revealed.itertuples()]})
