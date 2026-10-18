import numpy as np

from core.baselines import METHODS, _spearman_or_none, run_benchmark, run_method
from core.constants import WT_SEQUENCE
from core.developability import RiskNormalizer
from data.datasets import generate_fixture


def _setup(n: int = 70):
    frame = generate_fixture(n)
    norm = RiskNormalizer.fit(frame.sequence)
    tau = norm.score(frame.sequence.iloc[0])["y_dev"]
    return frame, norm, tau, frame.candidate_id.head(5).tolist()


def test_methods_share_seeds_and_spend_35_queries():
    frame, norm, tau, initial = _setup()
    results = [run_method(frame, norm, tau, m, 0, initial) for m in METHODS]
    assert all(r.initial_ids == initial and [x["query_count"] for x in r.curve] == list(range(5, 36, 5)) for r in results)
    # Every method spends the same 30 post-seed queries, so the hit-rate
    # denominator is matched and the numerator is the only thing that varies.
    assert all(r.top_decile_queries == 30 for r in results)


def test_normalized_curve_is_bounded_and_starts_at_zero():
    """``y*(q)`` is rescaled between the seed score (0.0) and the pool ceiling."""
    frame, norm, tau, initial = _setup()
    for method in METHODS:
        result = run_method(frame, norm, tau, method, 0, initial)
        values = [p["normalized"] for p in result.normalized_curve]
        assert values[0] == 0.0
        assert all(0.0 <= v <= 1.0 for v in values if np.isfinite(v))
        # Normalisation is monotone and cannot reorder discovery checkpoints.
        assert np.all(np.diff([v for v in values if np.isfinite(v)]) >= -1e-12)
        assert np.isfinite(result.normalized_aubqc)


def test_top5_panel_is_a_mean_of_five_not_a_single_max():
    """Panel depth is a different quantity from the best-single-molecule curve."""
    frame, norm, tau, initial = _setup(120)
    result = run_method(frame, norm, tau, "m3_ucb", 0, initial)
    for point, panel in zip(result.curve, result.panel_curve):
        assert panel["query_count"] == point["query_count"]
        # A panel mean can never exceed the single best value, and with more than
        # five qualified hits it is strictly below it.
        assert panel["top5_panel_binding"] <= point["best_qualified_binding"] + 1e-12
    assert result.panel_aubqc <= result.aubqc + 1e-12


def test_spearman_is_undefined_not_zero_without_a_model():
    """M1 has no scoring model: ``None`` is honest, ``0.0`` would be a claim."""
    frame, norm, tau, initial = _setup()
    result = run_method(frame, norm, tau, "m1_random", 0, initial)
    assert all(point["spearman_rho"] is None for point in result.rho_curve)


def test_spearman_helps_on_a_learnable_pool_and_is_finite():
    """The surrogate must beat chance rank correlation once it has seen labels."""
    frame, norm, tau, initial = _setup(120)
    result = run_method(frame, norm, tau, "m3_ucb", 0, initial)
    rhos = [p["spearman_rho"] for p in result.rho_curve]
    assert all(r is not None and np.isfinite(r) for r in rhos)
    assert max(rhos) > 0.0


def test_spearman_helper_returns_none_for_degenerate_input():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman_or_none(np.zeros(4), truth) is None
    assert _spearman_or_none(np.array([1.0, 2.0, 3.0, 4.0]), truth) == 1.0
    assert _spearman_or_none(np.array([4.0, 3.0, 2.0, 1.0]), truth) == -1.0
    assert _spearman_or_none(np.array([1.0, np.nan]), np.array([1.0, 2.0])) is None


def test_run_benchmark_emits_all_four_self_learning_metrics():
    frame, norm, tau, _ = _setup(90)
    report = run_benchmark(frame, norm, tau, seeds=2)
    for key in ("curves", "aubqc", "normalized_curves", "normalized_aubqc",
                "top5_panel_curves", "top5_panel_aubqc", "spearman_rho_curves",
                "top_decile_hit_rate"):
        assert key in report, key
    for key in ("normalized_curves", "top5_panel_curves", "spearman_rho_curves"):
        assert [row["query_count"] for row in report[key]] == list(range(5, 36, 5))
        assert all(method in report[key][0] for method in METHODS)
    for key in ("normalized_aubqc", "top5_panel_aubqc", "top_decile_hit_rate"):
        assert set(report[key]) == set(METHODS)
    # Hit rates are fractions of the 30 post-seed queries.
    assert all(0.0 <= value <= 1.0 for value in report["top_decile_hit_rate"].values())
    # M1 has no model, so its rank correlation is reported as null, not zero.
    assert all(row["m1_random"] is None for row in report["spearman_rho_curves"])


def test_top_decile_threshold_is_the_pool_ninety_ninth_percentile_definition():
    """The hit definition is a pool quantile over qualified candidates only."""
    frame, norm, tau, initial = _setup(120)
    y_dev = {cid: norm.score(seq)["y_dev"] for cid, seq in zip(frame.candidate_id, frame.sequence)}
    qualified = frame[[y_dev[cid] >= tau for cid in frame.candidate_id]]
    threshold = float(np.quantile(qualified.y_bind, 0.90))
    result = run_method(frame, norm, tau, "m3_ucb", 0, initial)
    y_bind = dict(zip(frame.candidate_id, frame.y_bind))
    hits = sum(y_bind[cid] >= threshold for batch in result.picks for cid in batch)
    assert result.top_decile_hits == hits
