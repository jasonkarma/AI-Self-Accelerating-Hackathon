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
    # The served file is ``benchmark_10seeds.json`` but holds a 100-seed run. The
    # UI must label the panel from the artifact's own ``seeds`` field, so that
    # fact is published as data instead of left to whoever reads the filename.
    benchmark["source"] = {
        "path": str(SETTINGS.benchmark_path),
        "filename": SETTINGS.benchmark_path.name,
        "seeds": benchmark.get("seeds"),
        "label": (
            f"{benchmark.get('seeds')}-seed benchmark"
            if benchmark.get("seeds")
            else "benchmark (seed count not recorded)"
        ),
        "filename_is_historical": SETTINGS.benchmark_path.name
        != f"benchmark_{benchmark.get('seeds')}seeds.json",
        "note": (
            "The filename is a historical artifact of the write path; the seed "
            "count in 'seeds' is authoritative. Label the UI from 'label', never "
            "from 'filename'."
        ),
    }
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


def _stat_key(*paths) -> tuple:
    """Cache key from ``(path, mtime_ns, size)`` triples, skipping absent files."""
    parts: list = []
    for path in paths:
        try:
            stat = path.stat()
            parts.extend([str(path), stat.st_mtime_ns, stat.st_size])
        except OSError:
            parts.extend([str(path), None, None])
    return tuple(parts)


def _load_json(path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _lockbox_metadata() -> dict:
    """The 17-target pre-registration funnel, as written by the audit script.

    Embedded in ``/api/cross-target`` because a single-target ``headline`` block
    is actively misleading on its own: it reads as though one target was chosen
    and run, when the point is that 16 others were measured and rejected by a
    rule fixed in advance. Absent file degrades to ``available: False`` rather
    than pretending the funnel does not exist.
    """
    path = SETTINGS.root / "results/lockbox_metadata.json"
    payload = _load_json(path)
    if payload is None:
        return {
            "available": False,
            "path": str(path),
            "detail": (
                "lockbox metadata not generated; run "
                "`python scripts/audit_lockbox_metadata.py`"
            ),
        }
    payload["available"] = True
    return payload


def _learnability_gate() -> dict:
    """The baseline-only learnability gate, as written by the gate script.

    Served alongside the funnel because the two answer different halves of the
    same question. ``lockbox_metadata`` says *which targets exist and what they
    contain*; this says *which of them can be learned from at all*, measured with
    M2/M3 only so the answer is independent of M5.

    Without it, a reader sees `aayl51` ranked 3/5 and cannot tell whether M5 lost
    or whether there was nothing to win.
    """
    path = SETTINGS.root / "results/learnability_gate.json"
    payload = _load_json(path)
    if payload is None:
        return {
            "available": False,
            "path": str(path),
            "detail": (
                "learnability gate not generated; run "
                "`python scripts/baseline_learnability_gate.py`"
            ),
        }
    payload["available"] = True
    return payload


def _one_8z_reference_row() -> dict | None:
    """1N8Z as a *reference* row for the cross-target matrix.

    The primary 1N8Z numbers live in a different artifact and were produced by a
    different protocol (35 queries over 210 qualified), so they cannot be dropped
    into ``targets[]`` as if they were measured the same way. They are surfaced
    under an explicit ``reference`` block instead, and the 16.7 % protocol
    (35/210) is stated so a reader can see the two are the same *rate*.
    """
    benchmark = _load_json(SETTINGS.benchmark_path)
    if not benchmark:
        return None
    return {
        "dataset": "1n8z",
        "label": "1N8Z (primary benchmark, reference row)",
        "cdr3_length": 13,
        "qualified_count": 210,
        "rows_final": 419,
        "seeds": benchmark.get("seeds"),
        "coverage_label": "16.7pct",
        "reference_only": True,
        "note": (
            "Measured by the primary 1N8Z benchmark (35-query protocol, 35/210 = "
            "16.7% coverage), not by the dynamic-sizing cross-target script. "
            "Reported separately because its budget and abscissae differ."
        ),
        "aubqc": benchmark.get("aubqc") or {},
        "aubqc_se": benchmark.get("aubqc_se") or {},
        "normalized_aubqc": benchmark.get("normalized_aubqc") or {},
        "top5_panel_aubqc": benchmark.get("top5_panel_aubqc") or {},
        "top_decile_hit_rate": benchmark.get("top_decile_hit_rate") or {},
        "spearman_rho_curves": benchmark.get("spearman_rho_curves") or [],
        "curves": benchmark.get("curves") or [],
        "checkpoints": [row["query_count"] for row in benchmark.get("curves") or []],
    }


def cross_target_payload() -> dict:
    """The multi-target generalization report, plus the holdout funnel.

    Served from the artifact rather than recomputed: the benchmark takes ~13
    minutes and the endpoint is a *report* of a run that was already made.

    Three additions, each fixing a way the raw artifact misleads on its own:

    * ``available`` — an explicit ``false`` on a checkout where the script has
      not run, instead of a 500 or a silently empty report;
    * ``lockbox_metadata`` — the 17-target funnel, so the frontend can show *why*
      the holdout is the holdout rather than presenting one target as a choice;
    * ``reference_targets`` — 1N8Z, which was measured by a different protocol
      and must not be compared row-for-row with the dynamic-sized cells.
    """
    global _cross_target_cache, _cross_target_key
    path = SETTINGS.cross_target_benchmark_path
    lockbox_path = SETTINGS.root / "results/lockbox_metadata.json"
    gate_path = SETTINGS.root / "results/learnability_gate.json"
    key = _stat_key(path, lockbox_path, gate_path, SETTINGS.benchmark_path)
    if _cross_target_cache is not None and _cross_target_key == key:
        return _cross_target_cache

    if not path.exists():
        payload = {
            "available": False,
            "path": str(path),
            "detail": (
                "cross-target artifact not generated; run "
                "`python scripts/run_cross_target_benchmark.py --seeds 50`"
            ),
        }
    else:
        loaded = _load_json(path)
        if loaded is None:
            payload = {
                "available": False,
                "path": str(path),
                "detail": "artifact present but not valid JSON",
            }
        else:
            payload = loaded
            payload.setdefault("cross_target", {})
            payload["available"] = True

    # The funnel, the learnability gate and the reference row are attached to
    # every response, including the unavailable one: a frontend that only has the
    # funnel should still be able to render the pre-registration story.
    payload["lockbox_metadata"] = _lockbox_metadata()
    payload["learnability_gate"] = _learnability_gate()
    reference = _one_8z_reference_row()
    payload["reference_targets"] = [reference] if reference else []
    _cross_target_cache, _cross_target_key = payload, key
    return payload


@app.get("/api/cross-target")
def cross_target(): return cross_target_payload()


#: ``results/attribution_ablation.json`` memoized on mtime+size, same pattern as
#: the cross-target artifact: a script writes it, so it can be regenerated while
#: the API is up.
_attribution_cache: dict | None = None
_attribution_key: tuple | None = None


def attribution_payload() -> dict:
    """The mechanism ablation: which part of M5's design earns its keep.

    Served verbatim from the artifact. The endpoint exists because the *negative*
    results are the scientifically load-bearing ones — a variant that ties
    production is evidence the mechanism is not doing the work, and there is no
    honest way to show that in a UI that only renders the winning configuration.
    """
    global _attribution_cache, _attribution_key
    path = SETTINGS.root / "results/attribution_ablation.json"
    key = _stat_key(path)
    if _attribution_cache is not None and _attribution_key == key:
        return _attribution_cache
    payload = _load_json(path)
    if payload is None:
        result = {
            "available": False,
            "path": str(path),
            "detail": (
                "attribution artifact not generated; run "
                "`python scripts/run_attribution_ablation.py --family all`"
            ),
        }
    else:
        payload["available"] = True
        result = payload
    _attribution_cache, _attribution_key = result, key
    return result


@app.get("/api/attribution")
def attribution(): return attribution_payload()

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
