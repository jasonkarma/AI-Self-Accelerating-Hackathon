"""Backend API and live SSE stream for the separately built React frontend."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import SETTINGS
from core.baselines import METHODS, ZERO_SHOT_REFERENCES, same_seed_comparison
from core.campaign import Campaign
from core.developability import RiskNormalizer
from core.preregistration import freeze_protocol, load_protocol
from core.tool_arsenal import configure_candidate_lookup
from core.ledger import ImmutableLedger
from data.datasets import load_primary, public_pool

campaign: Campaign | None = None

#: Enriched ``/api/benchmark`` body, memoized on everything it depends on.
#: ``same_seed_comparison`` re-runs five 35-query campaigns (~1.5 s), which is far
#: too slow to pay on every poll from the UI — but it *must* be recomputed after a
#: rewind, because the live trajectory it is compared against just changed.
_benchmark_cache: dict | None = None
_benchmark_key: tuple | None = None


def initialize(ledger_path=None, golden_ledger_path=None) -> Campaign:
    """Build the campaign the API serves.

    ``ledger_path`` / ``golden_ledger_path`` exist so a rehearsal (the
    stage-readiness suite, or a scratch sandbox) can drive the real app against
    its *own* copy of the verified run. ``SETTINGS`` is frozen, so threading the
    paths through is the only way to test rewind without the test being able to
    touch the file the demo depends on.
    """
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
    return Campaign(
        private, normalizer, protocol, benchmark,
        ledger_path=ledger_path, golden_ledger_path=golden_ledger_path,
    )


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


def invalidate_benchmark_cache() -> None:
    """Force the next ``/api/benchmark`` to rebuild the same-seed comparison."""
    global _benchmark_cache, _benchmark_key
    _benchmark_cache = None
    _benchmark_key = None


def benchmark_payload() -> dict:
    """The frozen 100-seed artifact plus its Seed-21 same-seed references.

    Two additions, both required by the report and neither present in the file
    written by ``scripts/run_benchmark.py`` (which has no live campaign to compare
    against):

    * ``zero_shot_references`` — the published AbBiBench numbers, injected here as
      well as from ``run_benchmark`` so an artifact frozen *before* they existed
      still serves them (an operator must never be able to ship a benchmark
      endpoint that silently omits its only external reference points);
    * ``same_seed_comparison`` — the recorded Seed-21 campaign against M1-M5
      re-run on the *identical* five initial queries.
    """
    global _benchmark_cache, _benchmark_key
    lab = current()
    benchmark = dict(lab.benchmark or {})
    benchmark.setdefault(
        "m5_policy_mode", SETTINGS.m5_policy_mode
    )
    benchmark.setdefault("seeds", 0)
    benchmark.setdefault("curves", [])
    benchmark.setdefault("aubqc", {})
    benchmark["zero_shot_references"] = dict(
        benchmark.get("zero_shot_references") or ZERO_SHOT_REFERENCES
    )

    key = (id(lab), lab.snapshot().version, SETTINGS.random_seed)
    if _benchmark_cache is not None and _benchmark_key == key:
        return _benchmark_cache

    seed = SETTINGS.random_seed
    matched = (benchmark.get("matched_initial_seeds") or {}).get(str(seed))
    try:
        comparison = same_seed_comparison(
            lab.private,
            lab.normalizer,
            lab.protocol["tau_dev"],
            seed=seed,
            initial_ids=[str(candidate_id) for candidate_id in matched] if matched else None,
            live_curve=lab.quality_curve,
            live_aubqc=lab.aubqc(),
            m5_policy_mode=benchmark.get("m5_policy_mode") or SETTINGS.m5_policy_mode,
        )
    except Exception as exc:  # noqa: BLE001 - a report must never 500 the stage
        comparison = {
            "seed": seed,
            "methods": list(METHODS),
            "trajectory": [],
            "summary": {},
            "ranking": [],
            "live": {"label": "live_m5", "q5": None, "q35": None, "aubqc": None},
            "error": f"{type(exc).__name__}: {exc}",
        }
    benchmark["same_seed_comparison"] = comparison
    _benchmark_cache, _benchmark_key = benchmark, key
    return benchmark


@app.get("/api/health")
def health(): return {"status": "ok", "version": "1.2"}

@app.get("/api/state")
def state(round: int | None = Query(default=None, ge=0, le=SETTINGS.rounds)):
    """Contract state, optionally **as of** an earlier round.

    ``GET /api/state?round=k`` reconstructs the round-``k`` view from the ledger
    prefix and never mutates the database, so the UI timeline stepper can inspect
    every round's memos, tool tables, VoI/maturity and 5-slot batch while the
    campaign itself sits at round 6.
    """
    lab = current()
    if round is None:
        return lab.state
    try:
        return lab.state_at(round)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

@app.get("/api/rounds")
def rounds():
    """One compact entry per round ``0..current`` for the timeline stepper."""
    lab = current()
    return {
        "seed": SETTINGS.random_seed,
        "current_round": lab.current_round(),
        "total_rounds": SETTINGS.rounds,
        "total_budget": SETTINGS.total_budget,
        "batch_size": SETTINGS.batch_size,
        "rounds": lab.timeline(),
    }

@app.get("/api/config")
def config(): return {**current().protocol, "total_budget": SETTINGS.total_budget, "hitl": SETTINGS.hitl, "m5_policy_mode": SETTINGS.m5_policy_mode}

@app.get("/api/candidates")
def candidates(offset: int = 0, limit: int = 100): return {"total": len(current().private), "items": current().candidates()[offset:offset + min(limit, 419)]}

@app.get("/api/benchmark")
def benchmark(): return benchmark_payload()


#: ``results/cross_target_benchmark.json`` memoized on its mtime+size. The artifact
#: is written by a script, not by the campaign, so it can be regenerated while the
#: API is up; keying on the stat means the endpoint picks that up without a restart
#: and without re-reading a multi-MB JSON on every poll.
_cross_target_cache: dict | None = None
_cross_target_key: tuple | None = None


def cross_target_payload() -> dict:
    """The second-target generalization report, as written by the script.

    Served verbatim rather than recomputed: the benchmark takes minutes and the
    endpoint is a *report* of a run that was already made. The one addition is
    ``available``, so a checkout where the script has not run yet gets an explicit
    ``false`` instead of a 500 or a misleadingly empty report.
    """
    global _cross_target_cache, _cross_target_key
    path = SETTINGS.cross_target_benchmark_path
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "detail": (
                "cross-target artifact not generated; run "
                "`python scripts/run_cross_target_benchmark.py --seeds 50`"
            ),
        }
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cross_target_cache is not None and _cross_target_key == key:
        return _cross_target_cache
    try:
        payload = json.loads(path.read_text())
        payload.setdefault("cross_target", {})
        payload["available"] = True
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "path": str(path),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    _cross_target_cache, _cross_target_key = payload, key
    return payload


@app.get("/api/cross-target")
def cross_target(): return cross_target_payload()

@app.get("/api/ledger")
def ledger():
    store = current().ledger
    # Ledger payloads are heterogeneous by design: an ``oracle_reveal`` carries
    # ``objective``/``value``, a ``round_decision`` carries ``declared_h0_id``.
    # ``DataFrame`` normalisation therefore fills the missing half with NaN, which
    # is not JSON — and a campaign with seed reveals has no decision columns at
    # all, so this endpoint used to 500 on its own golden ledger. Missing means
    # "not recorded for this event", which is exactly JSON ``null``.
    frame = store.dataframe()
    entries = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return {"verified": store.verify(), "metadata": store.metadata(), "entries": entries}

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
        invalidate_benchmark_cache()
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

class RewindRequest(BaseModel):
    """``POST /api/rewind`` body. Bounded here as well as in the campaign."""

    round: int = Field(ge=0, le=SETTINGS.rounds, description="Round to rewind to (0 = seed batch only)")


@app.post("/api/restore-golden")
def restore_golden():
    """Copy the frozen Seed-21 ledger back over the live one.

    The stage-recovery button: whatever a live demo did to
    ``results/campaign.sqlite3``, one call re-installs the verified 6-round
    campaign (``8.580`` best / ``1.963`` AUBQC) and re-hydrates the in-memory
    campaign from it in a few milliseconds.
    """
    try:
        result = current().restore_golden()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    invalidate_benchmark_cache()
    result["state"] = current().state_at(result["current_round"], include_tool_outputs=False)
    return result


@app.post("/api/rewind")
def rewind(payload: RewindRequest = Body(...)):
    """Rewind the live ledger to the end of ``payload.round``.

    Restores from the golden ledger first when the live one is shorter, so
    ``{"round": 5}`` after a botched Round 6 still lands on the Seed-21 R5 state
    (``30`` queries spent) and the next ``/api/stream-round`` runs Round 6 live
    over SSE from there.
    """
    lab = current()
    try:
        result = lab.rewind_to(payload.round)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    invalidate_benchmark_cache()
    return result


@app.post("/api/campaign/reset")
def reset():
    global campaign; campaign = initialize(); invalidate_benchmark_cache(); return campaign.state
