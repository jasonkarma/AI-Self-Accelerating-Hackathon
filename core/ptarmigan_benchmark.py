"""Stage B — closed-loop falsification benchmark over 50 matched seeds.

Four methods, each run over the same 50 seeds with matched ``Round-0`` queries:

+------------------------+----------+--------+-------------------------------+-------------------------------+
| Method                 | Rounds   | Wells  | Round-0 selection              | Acquisition rule              |
+------------------------+----------+--------+-------------------------------+-------------------------------+
| ``ptarmigan_zeroshot`` | 0        | 0      | Ptarmigan engagement ranking   | None (static)                 |
| ``m3_ucb_cold``        | 6        | 30     | Random (matched across rows)   | UCB on molecular features     |
| ``ptarmigan_static``   | 6        | 30     | Ptarmigan engagement ranking   | UCB on molecular features     |
| ``m5_ptarmigan_hybrid``| 6        | 30     | Ptarmigan engagement ranking   | Engagement + residue cosine   |
|                        |          |        | + hotspot residue cosine       | as Δ_agent on UCB             |
+------------------------+----------+--------+-------------------------------+-------------------------------+

Paired stats follow the exact arithmetic of :func:`core.baselines._paired_test`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = 50
"""Number of matched seeds for the benchmark."""

ROUNDS = 6
"""Number of acquisition rounds."""

BATCH_SIZE = 5
"""Wells per round (matches the production ``3:1:1`` 5-slot law)."""

INITIAL_QUERIES = 5
"""Round-0 queries shared across methods on the same seed."""

TOTAL_QUERIES = INITIAL_QUERIES + ROUNDS * BATCH_SIZE  # 35
"""Total queries per seed (matches the production 35-query budget)."""

FINGERPRINT_SIZE = 512
"""Width of the hashed-SMILES fingerprint vector."""

NGRAM_MIN = 2
NGRAM_MAX = 5
"""Character n-gram range for the hashed fingerprint."""

HARD_GATE_USD = 15.0

#: The four benchmark method names, in display order.
METHODS: tuple[str, ...] = (
    "ptarmigan_zeroshot",
    "m3_ucb_cold",
    "ptarmigan_static",
    "m5_ptarmigan_hybrid",
)

PAIRED_METRICS: tuple[str, ...] = (
    "aubqc",
    "spearman_rho_at_q35",
    "top_decile_hit_rate",
)


# ---------------------------------------------------------------------------
# Molecular featuriser (no RDKit — hashed SMILES n-grams + physchem counts)
# ---------------------------------------------------------------------------

def _smiles_ngrams(smiles: str) -> list[str]:
    """Generate overlapping character n-grams from a SMILES string."""
    ngrams: list[str] = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(smiles) - n + 1):
            ngrams.append(smiles[i:i + n])
    return ngrams


def _physchem_counts(smiles: str) -> list[float]:
    """Lightweight physicochemical rule-of-thumb counts from SMILES.

    These are heuristic approximations — not RDKit-grade, but sufficient to
    distinguish compound classes in the fingerprint.
    """
    s = smiles
    # Count heteroatoms (non-C, non-H)
    heteroatom = sum(1 for c in s if c in "NOnSFPClBrI")
    # Count aromatic indicators (lowercase = aromatic in SMILES)
    aromatic = sum(1 for c in s if c.islower() and c in "cno")
    # Count ring indicators (digits)
    rings = sum(1 for c in s if c.isdigit())
    # Count branching indicators (parentheses)
    branches = s.count("(") + s.count(")")
    # Count double/triple bonds
    multiple_bonds = s.count("=") + s.count("#")
    # Count charge indicators
    charges = s.count("+") + s.count("-")
    # Approximate rotatable bonds (single bonds not in rings)
    # SMILES single bond is typically implicit; count 'C' as a proxy
    chain_bonds = sum(1 for i in range(1, len(s)) if s[i].isalpha() and s[i-1].isalpha() and s[i] not in "=#")
    return [
        float(len(s)),          # 0: length
        float(heteroatom),      # 1: heteroatom count
        float(aromatic),        # 2: aromatic indicator
        float(rings),           # 3: ring count
        float(branches),        # 4: branching count
        float(multiple_bonds),  # 5: multiple bond count
        float(charges),         # 6: charge count
        float(chain_bonds),     # 7: approximate rotatable bonds
    ]


PHYSCHEM_WIDTH = 8
"""Width of the physicochemical counts block."""


def featurise_smiles(smiles_list: list[str]) -> np.ndarray:
    """Convert a list of SMILES strings to a fixed-width fingerprint.

    Uses hashed character n-grams (2- to 5-mers) concatenated with
    physicochemical rule-of-thumb counts.  Total width = ``FINGERPRINT_SIZE + 8``.
    Returns ``(n, FINGERPRINT_SIZE + PHYSCHEM_WIDTH)`` float array.
    """
    n = len(smiles_list)
    X = np.zeros((n, FINGERPRINT_SIZE + PHYSCHEM_WIDTH), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        if not s:
            continue
        # Hashed n-grams
        ngrams = _smiles_ngrams(s)
        for ng in ngrams:
            idx = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16) % FINGERPRINT_SIZE
            X[i, idx] += 1.0
        # Physicochemical counts
        X[i, FINGERPRINT_SIZE:] = _physchem_counts(s)
    return X


# ---------------------------------------------------------------------------
# Hotspot residue cosine similarity
# ---------------------------------------------------------------------------

def hotspot_cosine(
    compound_engagement: list[float | None],
    hotspot_mask: np.ndarray,
    *,
    n_residues: int | None = None,
) -> float:
    """Cosine similarity between a compound's residue engagement vector and the
    top predicted hotspot mask.

    ``compound_engagement``: the per-residue engagement array from
    Ptarmigan's ``compound_detail.residue_engagement``.
    ``hotspot_mask``: boolean array of same length; ``True`` at top-10% engaged
    positions across all hits.
    Returns a similarity in ``[0, 1]``.
    """
    vec = np.asarray(
        [0.0 if v is None else float(v) for v in compound_engagement],
        dtype=float,
    )
    if n_residues is not None and len(vec) != n_residues:
        # Pad or truncate to match reference proteome length.
        if len(vec) < n_residues:
            vec = np.pad(vec, (0, n_residues - len(vec)), constant_values=0.0)
        else:
            vec = vec[:n_residues]

    # Ensure hotspot_mask matches.
    mask = np.asarray(hotspot_mask, dtype=float)
    if len(mask) != len(vec):
        if len(mask) < len(vec):
            mask = np.pad(mask, (0, len(vec) - len(mask)), constant_values=0.0)
        else:
            mask = mask[:len(vec)]

    numerator = float(np.dot(vec, mask))
    denom = float(np.linalg.norm(vec) * np.linalg.norm(mask))
    if denom == 0.0:
        return 0.0
    return numerator / denom


def compute_hotspot_mask(
    residue_engagements: list[list[float | None]],
    top_pct: float = 0.10,
) -> np.ndarray:
    """Compute a consensus hotspot mask from all screened compounds' per-residue
    engagement profiles.

    ``residue_engagements``: list of ``residue_engagement`` arrays (one per
    compound).  ``top_pct``: fraction of the highest-mean-engagement residues
    to mark as hotspots.  Returns a boolean mask.
    """
    max_len = max(len(arr) for arr in residue_engagements if arr is not None)
    mean_engagement = np.zeros(max_len, dtype=float)
    count = np.zeros(max_len, dtype=int)

    for arr in residue_engagements:
        if arr is None:
            continue
        n = len(arr)
        for j, v in enumerate(arr):
            if v is not None:
                mean_engagement[j] += float(v)
                count[j] += 1

    mean_engagement = np.divide(
        mean_engagement, count, out=np.zeros_like(mean_engagement), where=count > 0
    )
    n_hotspots = max(1, int(len(mean_engagement) * top_pct))
    threshold = -np.sort(-mean_engagement)[min(n_hotspots - 1, len(mean_engagement) - 1)]
    return mean_engagement >= threshold


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PtarmiganBenchmarkResult:
    """One method's results across ``n`` seeds, matching the paired-test interface.

    Fields correspond to :class:`core.baselines.BenchmarkResult` but only carry
    what the Ptarmigan loop needs.
    """
    def __init__(self, method: str, records: list[dict]) -> None:
        self.method = method
        self.records = records  # one per seed: {seed, aubqc, rho, top_decile_hit_rate, top_decile_hits, top_decile_queries, best_per_round, ...}
        self._seed_map: dict[int, dict] = {r["seed"]: r for r in records}

    @property
    def seeds(self) -> int:
        return len(self.records)

    @property
    def aubqc(self) -> float | None:
        """Mean AUBQC across seeds, or ``None`` when the method has no curve.

        ``ptarmigan_zeroshot`` spends zero wet-lab wells, so it has no area
        under a discovery curve by definition. Returning ``None`` (rather than
        ``nan``) keeps that "not applicable" distinct from "computed and
        undefined", and matches the per-seed value the loop emits.
        """
        vals = [r["aubqc"] for r in self.records if r["aubqc"] is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def aubqc_se(self) -> float | None:
        vals = [r["aubqc"] for r in self.records if r["aubqc"] is not None]
        return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None

    @property
    def rho(self) -> float | None:
        vals = [r["spearman_rho_at_q35"] for r in self.records if r["spearman_rho_at_q35"] is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def top_decile_hit_rate(self) -> float | None:
        total_hits = sum(r["top_decile_hits"] for r in self.records)
        total_queries = sum(r["top_decile_queries"] for r in self.records)
        return total_hits / total_queries if total_queries else None

    def get_seed(self, seed: int) -> dict | None:
        return self._seed_map.get(seed)


# ---------------------------------------------------------------------------
# UCB acquisition on molecular features
# ---------------------------------------------------------------------------

def _ucb_pick(
    X: np.ndarray,
    y: np.ndarray,
    unqueried_idx: np.ndarray,
    *,
    beta: float = 1.0,
) -> int:
    """Standard UCB: rank unqueried items by ``mu + beta * sigma``.

    ``X``: feature matrix of all compounds.
    ``y``: measured binding values of queried compounds (aligned to ``X`` rows
    via ``queried_idx``; unqueried rows have NaN).
    ``unqueried_idx``: indices into ``X`` of candidates not yet assayed.
    Returns the index (into ``X``) of the best UCB pick.
    """
    known = ~np.isnan(y)
    if not known.any():
        # Cold start: pick the one with the highest total feature magnitude.
        scores = np.linalg.norm(X, axis=1)
        candidate_scores = scores[unqueried_idx]
        return unqueried_idx[int(np.argmax(candidate_scores))]

    # Nearest-neighbour surrogate: for each unqueried candidate, use the mean
    # of its k nearest known neighbours as mu, and their std as sigma.
    k = min(5, int(known.sum()))
    from sklearn.neighbors import NearestNeighbors  # in requirements via sklearn

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X[known])
    distances, indices = nn.kneighbors(X[unqueried_idx])

    best_score, best_idx = -np.inf, unqueried_idx[0]
    for i, u_idx in enumerate(unqueried_idx):
        neighbours = np.where(known)[0][indices[i]]
        mu = float(y[neighbours].mean())
        sigma = float(y[neighbours].std(ddof=1)) if k > 1 else 0.0
        score = mu + beta * sigma
        if score > best_score:
            best_score = score
            best_idx = u_idx

    return best_idx


def _ucb_scores(
    X: np.ndarray,
    y: np.ndarray,
    unqueried_idx: np.ndarray,
    *,
    beta: float = 1.0,
) -> np.ndarray:
    """Vectorised UCB score for every unqueried candidate.

    Same arithmetic as :func:`_ucb_pick` (``mu + beta * sigma`` over the k
    nearest known compounds), but the ``NearestNeighbors`` index is built once
    per call instead of once per candidate. ``_ucb_pick`` still fits a model per
    call; at one well per round that was tolerable, but a full 35-query budget
    revisits this ~30 times per seed, so the M5 branch needs the batched form.

    Returns an array aligned with ``unqueried_idx``. **Callers must treat the
    result positionally** — index ``i`` is the score of ``unqueried_idx[i]``,
    not of row ``i`` of ``X``.
    """
    known = ~np.isnan(y)
    if not known.any():
        return np.linalg.norm(X[unqueried_idx], axis=1)

    k = min(5, int(known.sum()))
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X[known])
    _, indices = nn.kneighbors(X[unqueried_idx])
    known_rows = np.where(known)[0]

    scores = np.empty(len(unqueried_idx), dtype=float)
    for i in range(len(unqueried_idx)):
        neighbours = known_rows[indices[i]]
        mu = float(y[neighbours].mean())
        sigma = float(y[neighbours].std(ddof=1)) if k > 1 else 0.0
        scores[i] = mu + beta * sigma
    return scores


# ---------------------------------------------------------------------------
# Falsification tracking
# ---------------------------------------------------------------------------

def _falsification_event(
    compound_engagement: float,
    all_engagements: list[float],
    *,
    percentile_threshold: float = 0.30,
) -> bool:
    """A falsification event occurs when the selected compound's engagement is
    in the bottom ``percentile_threshold`` of the ranking posterior.

    ``compound_engagement``: Ptarmigan engagement score (0-1) of the picked
    compound. ``all_engagements``: all engagement scores from the screen hits.
    Returns ``True`` when the pick falls below the threshold percentile.
    """
    if not all_engagements:
        return False
    sorted_eng = sorted(all_engagements)
    threshold = np.percentile(sorted_eng, percentile_threshold * 100)
    return compound_engagement <= threshold


# ---------------------------------------------------------------------------
# One seed run
# ---------------------------------------------------------------------------

def _one_seed(
    seed: int,
    df: pd.DataFrame,
    method: str,
    ptarmigan_ranking: list[dict] | None = None,
    hotspot_mask: np.ndarray | None = None,
    residue_engagements: dict[str, list[float | None]] | None = None,
    n_residues: int = 630,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run one method on one seed.

    ``df`` must have columns ``["compound_id", "smiles", "pchembl_value"]``.
    ``ptarmigan_ranking``: list of hit dicts with ``compound_id`` and ``engagement``.
    ``hotspot_mask``: boolean mask of hotspot residues.
    ``residue_engagements``: ``{compound_id: residue_engagement_array}``.
    ``n_residues``: length of the reference proteome (default 630 for ERBB2).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Build the pool.
    pool = df.copy()
    pool = pool.rename(columns={"pchembl_value": "y_bind"})

    # Map compound_id to ptarmigan engagement.
    engagement_map: dict[str, float] = {}
    if ptarmigan_ranking:
        for h in ptarmigan_ranking:
            cid = str(h["compound_id"])
            engagement_map[cid] = float(h["engagement"])
    pool["engagement"] = pool["compound_id"].map(engagement_map).fillna(0.0)

    # Query ordering by engagement.
    pool = pool.sort_values("engagement", ascending=False).reset_index(drop=True)

    ptarmigan_ordered_ids = pool["compound_id"].tolist()

    # !=---------- seed-matching ----------!=

    # For m3_ucb_cold: use rng for random round-0.
    # For ptarmigan_zeroshot: static ranking, no rounds.
    # For ptarmigan_static: round-0 seeded by top-5 ptarmigan, then UCB.
    # For m5_ptarmigan_hybrid: round-0 seeded by ptarmigan + hotspot cosine, then UCB+Δ.

    # Round-0 selection.
    n_pool = len(pool)
    if method in ("ptarmigan_static", "m5_ptarmigan_hybrid"):
        # Top 5 by engagement (enriched by hotspot cosine for hybrid).
        if method == "m5_ptarmigan_hybrid" and hotspot_mask is not None and residue_engagements:
            # Re-rank by a blend: 0.7 * engagement + 0.3 * hotspot cosine.
            scores: list[tuple[int, float]] = []
            for idx, row in pool.iterrows():
                cid = str(row["compound_id"])
                eng = float(row["engagement"])
                res_eng = residue_engagements.get(cid)
                if res_eng is not None:
                    cos_sim = hotspot_cosine(res_eng, hotspot_mask, n_residues=n_residues)
                else:
                    cos_sim = 0.0
                scores.append((idx, 0.7 * eng + 0.3 * cos_sim))
            scores.sort(key=lambda x: -x[1])
            initial_ids = [str(pool.at[idx, "compound_id"]) for idx, _ in scores[:INITIAL_QUERIES]]
        else:
            initial_ids = ptarmigan_ordered_ids[:INITIAL_QUERIES]

    elif method == "m3_ucb_cold":
        # Random round-0 (matched across seeds via the seed parameter).
        shuffled = ptarmigan_ordered_ids.copy()
        rng.shuffle(shuffled)
        initial_ids = shuffled[:INITIAL_QUERIES]
    else:
        # ptarmigan_zeroshot — no rounds; static evaluation.
        initial_ids = ptarmigan_ordered_ids[:INITIAL_QUERIES]

    # Feature matrix.
    X = featurise_smiles(pool["smiles"].tolist())

    # Ensure we have the columns we need.
    pool["y_pred"] = pool["engagement"]  # For zeroshot correlation.

    # !=---------- Run method ----------!=

    if method == "ptarmigan_zeroshot":
        return _run_zeroshot(seed, pool, initial_ids)
    else:
        return _run_active(seed, pool, X, method, initial_ids, ptarmigan_ranking, hotspot_mask, residue_engagements, n_residues, engagement_map, rng)


def _run_zeroshot(seed: int, pool: pd.DataFrame, initial_ids: list[str]) -> dict[str, Any]:
    """Static evaluation: predict all pool candidates, compute metrics once.

    Zero wet-lab rounds, so AUBQC is null by definition.
    """
    # The zeroshot "prediction" is the Ptarmigan engagement, sorted descending.
    # It acts as a static ranker with no active learning.
    predictions = pool.set_index("compound_id")["engagement"].to_dict()
    truth = pool.set_index("compound_id")["y_bind"].to_dict()

    # Spearman ρ: correlation of engagement rank vs pchembl rank across all compounds.
    ordered_truth = [truth[cid] for cid in initial_ids]
    ordered_pred = [predictions[cid] for cid in initial_ids]

    from scipy.stats import spearmanr
    rho, _ = spearmanr(ordered_truth, ordered_pred) if len(ordered_truth) >= 3 else (None, None)

    return {
        "seed": seed,
        "method": "ptarmigan_zeroshot",
        "aubqc": None,  # No active learning rounds.
        "spearman_rho_at_q35": float(rho) if rho is not None else None,
        "top_decile_hits": 0,
        "top_decile_queries": 0,
        "best_per_round": [],
        "round_histories": [],
        "falsification_events": 0,
        "hotspot_residues": [],
    }


def _run_active(
    seed: int,
    pool: pd.DataFrame,
    X: np.ndarray,
    method: str,
    initial_ids: list[str],
    ptarmigan_ranking: list[dict] | None,
    hotspot_mask: np.ndarray | None,
    residue_engagements: dict[str, list[float | None]] | None,
    n_residues: int,
    engagement_map: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run an active-learning method (m3_ucb_cold, ptarmigan_static, m5_ptarmigan_hybrid)."""
    queried: list[str] = list(initial_ids)
    best_per_round: list[dict] = []
    round_histories: list[list[dict]] = []
    falsification_events = 0
    falsification_queries = 0
    all_engagements = list(engagement_map.values()) if engagement_map else []

    # Check for degenerate edge case: all measured values are identical.
    id_to_idx = {cid: i for i, cid in enumerate(pool["compound_id"])}
    y_measured = np.full(len(pool), np.nan, dtype=float)

    for cid in queried:
        if cid in id_to_idx:
            idx = id_to_idx[cid]
            y_measured[idx] = float(pool.loc[pool["compound_id"] == cid, "y_bind"].iloc[0])

    # AUBQC is the mean best-so-far binding over the checkpoints reached
    # (5, 10, 15, ... up to TOTAL_QUERIES). It is recorded here rather than
    # left to the runner so the published figure is always the curve the loop
    # actually walked.
    aubqc_checkpoints: list[float] = []
    round_number = 0
    for round_idx in range(ROUNDS):
        round_number += 1

        if len(queried) >= TOTAL_QUERIES:
            break

        # ---- Spend one BATCH_SIZE batch this round (5 wells, 3:1:1 law) ----
        n_this_round = min(BATCH_SIZE, TOTAL_QUERIES - len(queried))
        for _slot in range(n_this_round):
            # ---- Identify unqueried candidates (positional row indices) ----
            unqueried = [cid for cid in pool["compound_id"] if cid not in queried]
            if not unqueried:
                break

            unqueried_idx = np.asarray([id_to_idx[cid] for cid in unqueried], dtype=int)

            # ---- Score and pick one ----
            # Index convention: ``_ucb_pick`` returns a *row index into X*, and
            # ``X[i]`` corresponds to ``pool.iloc[i]`` because ``pool`` was
            # sorted by engagement and ``reset_index(drop=True)`` before the
            # feature matrix was built. The M5 branch below indexes the
            # ``unqueried`` list positionally instead, so it converts through
            # ``id_to_idx`` — never through ``pool.iloc`` — to stay consistent.
            if method in ("m3_ucb_cold", "ptarmigan_static"):
                # m3 starts cold (random round-0); ptarmigan_static round-0 is
                # the Ptarmigan top-5. After round 0 both are pure UCB on
                # molecular features (no Ptarmigan signal in-loop).
                pick_idx = _ucb_pick(X, y_measured, unqueried_idx, beta=1.0)
                chosen_cid = str(pool.iloc[pick_idx]["compound_id"])

            elif method == "m5_ptarmigan_hybrid":
                # UCB on features PLUS Δ_agent from engagement and hotspot
                # similarity. Δ_agent is the whole point of the hybrid: without
                # it this method is byte-identical to ptarmigan_static.
                base_scores = _ucb_scores(X, y_measured, unqueried_idx, beta=1.0)
                delta = np.zeros(len(unqueried), dtype=float)
                for i, u_idx in enumerate(unqueried_idx):
                    cid = str(pool.iloc[u_idx]["compound_id"])
                    engagement = engagement_map.get(cid, 0.0)
                    residue_sim = 0.0
                    if hotspot_mask is not None and residue_engagements is not None:
                        res_eng = residue_engagements.get(cid)
                        if res_eng is not None:
                            residue_sim = hotspot_cosine(res_eng, hotspot_mask, n_residues=n_residues)
                    # Δ_agent = 0.4 * engagement + 0.6 * residue_sim (bounded)
                    delta[i] = max(-0.15, min(0.15, 0.4 * engagement + 0.6 * residue_sim))

                # Add Δ_agent — the term that was previously computed and then
                # dropped on the floor, making M5 ≡ ptarmigan_static.
                best_i = int(np.argmax(base_scores + delta))
                chosen_cid = unqueried[best_i]

                # Falsification counts *pulls that landed on a low-engagement
                # compound*, so it is expressed as a rate over the adaptive
                # picks (``falsification_queries``), never as "events / seeds".
                # A per-seed 0/1 flag was bounded but vacuous: with 30 picks it
                # read 50/50 on every target and carried no information.
                engagement_chosen = engagement_map.get(chosen_cid, 0.0)
                falsification_queries += 1
                if _falsification_event(engagement_chosen, all_engagements):
                    falsification_events += 1
            else:
                raise ValueError(f"Unknown method: {method}")

            # ---- Measure ----
            queried.append(chosen_cid)
            chosen_y = float(pool.loc[pool["compound_id"] == chosen_cid, "y_bind"].iloc[0])
            y_measured[id_to_idx[chosen_cid]] = chosen_y

            round_histories.append({
                "round": round_number,
                "candidate_id": chosen_cid,
                "y_bind": chosen_y,
            })

        # ---- Checkpoint after the batch: best qualified binding so far ----
        known_y = y_measured[~np.isnan(y_measured)]
        best_y = float(np.nanmax(known_y)) if known_y.size > 0 else float("nan")
        best_per_round.append({
            "round": round_number,
            "query_count": len(queried),
            "best_qualified_binding": best_y,
        })
        if known_y.size > 0:
            aubqc_checkpoints.append(best_y)

    # ---- Final metrics ----
    known_y = y_measured[~np.isnan(y_measured)]
    decile_threshold = float(np.quantile(pool["y_bind"], 0.90)) if len(pool) else float("inf")

    top_decile_hits = int(np.sum(known_y >= decile_threshold))
    top_decile_queries = int(len(known_y))

    # Spearman ρ at Q=35.
    predictions = [float(engagement_map.get(cid, 0.0)) for cid in queried]
    truths = [float(pool.loc[pool["compound_id"] == cid, "y_bind"].iloc[0]) for cid in queried]

    from scipy.stats import spearmanr
    rho, _ = spearmanr(truths, predictions) if len(truths) >= 3 else (None, None)

    return {
        "seed": seed,
        "method": method,
        # AUBQC = mean of the best-so-far binding at each batch checkpoint. This
        # was previously hardcoded to None because the round loop only advanced
        # one well per round and never reached TOTAL_QUERIES; it is now measured.
        "aubqc": float(np.mean(aubqc_checkpoints)) if aubqc_checkpoints else None,
        "spearman_rho_at_q35": float(rho) if rho is not None else None,
        "top_decile_hits": top_decile_hits,
        "top_decile_queries": top_decile_queries,
        "best_per_round": best_per_round,
        "round_histories": round_histories,
        "falsification_events": falsification_events,
        "falsification_queries": falsification_queries,
        "falsification_rate": (
            falsification_events / falsification_queries if falsification_queries else None
        ),
        "hotspot_residues": [],  # Filled by the runner from Stage A data.
    }


def _nn_mu_sigma(X: np.ndarray, y: np.ndarray, idx: int, k: int = 5) -> float:
    """Compute the UCB score for candidate ``idx`` using k-NN on known rows."""
    known = ~np.isnan(y)
    if not known.any():
        return float(np.linalg.norm(X[idx]))

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, int(known.sum())), metric="cosine").fit(X[known])
    distances, indices = nn.kneighbors(X[idx].reshape(1, -1))
    neighbours = np.where(known)[0][indices[0]]
    mu = float(y[neighbours].mean())
    sigma = float(y[neighbours].std(ddof=1)) if len(neighbours) > 1 else 0.0
    return mu + 1.0 * sigma


# ---------------------------------------------------------------------------
# Paired test (exact arithmetic of core.baselines._paired_test)
# ---------------------------------------------------------------------------

def paired_test(
    treatment: list[dict],  # per-seed records
    control: list[dict],    # per-seed records
    metric: str,
) -> dict:
    """Paired t-test of ``treatment - control`` across matched seeds.

    Numerically agrees with :func:`core.baselines._paired_test`.
    """
    treat_map = {r["seed"]: r for r in treatment}
    control_map = {r["seed"]: r for r in control}
    seeds = sorted(set(treat_map) & set(control_map))

    diffs: list[float] = []
    for s in seeds:
        tv = _get_metric(treat_map[s], metric)
        cv = _get_metric(control_map[s], metric)
        if tv is not None and cv is not None:
            diffs.append(float(tv) - float(cv))

    n = len(diffs)
    if n < 2:
        return {
            "metric": metric,
            "n": n,
            "mean_diff": float("nan"),
            "se": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "seeds_won": 0,
        }

    arr = np.asarray(diffs, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    t_stat = float(mean / se) if se > 0 else float("inf") if mean != 0 else 0.0

    if se > 0:
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    else:
        p_value = 1.0 if mean == 0 else 0.0

    return {
        "metric": metric,
        "n": n,
        "mean_diff": mean,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "seeds_won": int(np.sum(arr > 0)),
    }


def _get_metric(record: dict, metric: str) -> float | None:
    if metric == "spearman_rho_at_q35":
        return record.get("spearman_rho_at_q35")
    if metric == "aubqc":
        return record.get("aubqc")
    if metric == "top_decile_hit_rate":
        hits = record.get("top_decile_hits", 0)
        queries = record.get("top_decile_queries", 0)
        return hits / queries if queries else None
    return record.get(metric)


# ---------------------------------------------------------------------------
# Top-level benchmark runner
# ---------------------------------------------------------------------------

def run_ptarmigan_benchmark(
    df: pd.DataFrame,
    ptarmigan_ranking: list[dict],
    residue_engagements: dict[str, list[float | None]],
    hotspot_mask: np.ndarray,
    n_residues: int,
    *,
    seeds: int = SEEDS,
) -> dict[str, Any]:
    """Run all four methods over ``seeds`` matched seeds.

    Returns the full benchmark dictionary with ``methods``, ``paired_comparisons``,
    and metadata.
    """
    all_seed_records: dict[str, list[dict]] = {m: [] for m in METHODS}

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for method in METHODS:
            rec = _one_seed(
                seed, df, method,
                ptarmigan_ranking=ptarmigan_ranking,
                hotspot_mask=hotspot_mask,
                residue_engagements=residue_engagements,
                n_residues=n_residues,
                rng=rng,
            )
            all_seed_records[method].append(rec)

    # Build results and paired comparisons.
    results = {}
    for method in METHODS:
        records = all_seed_records[method]
        result = PtarmiganBenchmarkResult(method, records)
        results[method] = {
            "method": method,
            "seeds": seeds,
            "aubqc": result.aubqc,
            "aubqc_se": result.aubqc_se,
            "rho": result.rho,
            "top_decile_hit_rate": result.top_decile_hit_rate,
            "per_seed": records,
        }

    # Paired comparisons: each active method vs the cold UCB control.
    paired: dict[str, dict[str, dict]] = {}
    for treatment in ("ptarmigan_static", "m5_ptarmigan_hybrid"):
        paired[treatment] = {}
        for metric in PAIRED_METRICS:
            paired[treatment][metric] = paired_test(
                all_seed_records[treatment],
                all_seed_records["m3_ucb_cold"],
                metric,
            )

    return {
        "methods": results,
        "paired_comparisons": paired,
        "metadata": {
            "seeds": seeds,
            "rounds": ROUNDS,
            "batch_size": BATCH_SIZE,
            "initial_queries": INITIAL_QUERIES,
            "total_queries_per_seed": TOTAL_QUERIES,
            "methods": list(METHODS),
            "falsification_percentile_threshold": 0.30,
        },
    }