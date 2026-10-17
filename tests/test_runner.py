import pandas as pd
import pytest
from core.virtual_lab_runner import VirtualLabRunner

@pytest.mark.asyncio
async def test_tool_first_runner_emits_schema_valid_decision():
    ledger=pd.DataFrame(columns=["candidate_id","sequence","y_bind","round_number"])
    unqueried=pd.DataFrame([{"candidate_id":f"C{i}","sigma_bind":i/10,"sigma_dev":i/11,"sigma_joint":i/9} for i in range(20)])
    runner=VirtualLabRunner(); decision,memos=await runner.plan_round(1,ledger,unqueried)
    assert len(memos)==3 and decision.slot3_uncertainty_target_id=="C19" and runner.events.history[0]["event"]=="round_started"
