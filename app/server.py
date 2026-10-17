"""Backend API and live SSE stream for the separately built React frontend."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import SETTINGS
from core.campaign import Campaign
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol, load_protocol
from core.tool_arsenal import configure_candidate_lookup
from core.ledger import ImmutableLedger
from data.datasets import load_primary, public_pool

campaign: Campaign | None = None


def initialize() -> Campaign:
    private, synthetic = load_primary()
    try:
        protocol = load_protocol()
        # Re-freeze automatically when the sequence pool changed.
        from core.preregistration import sequence_hash
        if protocol["dataset_sequence_sha256"] != sequence_hash(public_pool(private)):
            protocol = freeze_protocol(public_pool(private), is_synthetic=synthetic)
    except FileNotFoundError:
        protocol = freeze_protocol(public_pool(private), is_synthetic=synthetic)
    normalizer = RiskNormalizer.from_dict(protocol["normalizer"])
    benchmark = json.loads(SETTINGS.benchmark_path.read_text()) if SETTINGS.benchmark_path.exists() else {}
    configure_candidate_lookup(dict(zip(private.candidate_id, private.sequence)))
    return Campaign(private, normalizer, protocol, benchmark)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global campaign
    campaign = initialize()
    yield


app = FastAPI(title="Self-Accelerating Antibody Virtual Lab", version="1.2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

def current() -> Campaign:
    global campaign
    if campaign is None: campaign = initialize()
    return campaign

@app.get("/api/health")
def health(): return {"status": "ok", "version": "1.2"}

@app.get("/api/state")
def state(): return current().state

@app.get("/api/config")
def config(): return {**current().protocol, "total_budget": SETTINGS.total_budget, "hitl": SETTINGS.hitl, "m5_policy_mode": SETTINGS.m5_policy_mode}

@app.get("/api/candidates")
def candidates(offset: int = 0, limit: int = 100): return {"total": len(current().private), "items": current().candidates()[offset:offset + min(limit, 419)]}

@app.get("/api/benchmark")
def benchmark(): return current().benchmark or {"m5_policy_mode": SETTINGS.m5_policy_mode, "seeds": 0, "curves": [], "aubqc": {}}

@app.get("/api/ledger")
def ledger():
    store = current().ledger
    return {"verified": store.verify(), "metadata": store.metadata(), "entries": store.dataframe().to_dict("records")}

@app.get("/api/hypotheses")
def hypotheses(): return {"items": current().hypotheses}

@app.get("/api/stream-round")
async def stream_round():
    async def generate():
        lab = current()
        task = asyncio.create_task(lab.execute_round())
        await asyncio.sleep(0)
        async for message in lab.runner.events.subscribe(replay=True):
            yield f"event: {message['event']}\ndata: {json.dumps(message['data'], default=str)}\n\n"
        await task
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/api/round/approve")
def approve(): return {"approved": True}

@app.post("/api/round/reject")
def reject(): return {"approved": False, "rejected": True}

@app.post("/api/round/run")
def run_round():
    if current().state["current_round"] >= SETTINGS.rounds:
        raise HTTPException(409, "Campaign already completed")
    return {"accepted": True, "stream": "/api/stream-round"}

@app.post("/api/campaign/reset")
def reset():
    global campaign; campaign = initialize(); return campaign.state
