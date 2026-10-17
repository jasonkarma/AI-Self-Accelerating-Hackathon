"""Dual bootstrapped ExtraTrees surrogate models and linked acquisition."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .constants import BINDING_WEIGHT, DEVELOPABILITY_WEIGHT, N_BOOTSTRAPS, QUALIFICATION_PENALTY


def normalize_uncertainty(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


class BootstrappedExtraTrees:
    """50 independently bootstrapped ExtraTrees models; ensemble SD is epistemic heuristic."""
    def __init__(self, random_state: int = 0, n_estimators: int = N_BOOTSTRAPS, n_jobs: int = -1):
        self.random_state, self.n_estimators, self.n_jobs = random_state, n_estimators, n_jobs
        self.model: ExtraTreesRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrappedExtraTrees":
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        if len(X) != len(y) or not len(y):
            raise ValueError("X/y must be aligned and non-empty")
        # ExtraTrees supports per-tree bootstrap directly. One 50-tree fit is equivalent to
        # 50 independently bootstrapped trees and is dramatically faster than 50 fit calls.
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators, bootstrap=True,
                                         max_features="sqrt", random_state=self.random_state,
                                         n_jobs=self.n_jobs)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, *, normalize_sigma: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Surrogate is not fitted")
        predictions = np.vstack([tree.predict(X) for tree in self.model.estimators_])
        sigma = predictions.std(axis=0, ddof=1)
        return predictions.mean(axis=0), normalize_uncertainty(sigma) if normalize_sigma else sigma


class BindingSurrogate(BootstrappedExtraTrees):
    pass


class DevelopabilitySurrogate(BootstrappedExtraTrees):
    pass


@dataclass(frozen=True)
class LinkedPredictions:
    mu_bind: np.ndarray
    sigma_bind: np.ndarray
    mu_dev: np.ndarray
    sigma_dev: np.ndarray
    a_base: np.ndarray
    sigma_joint: np.ndarray


def link_predictions(mu_bind: np.ndarray, sigma_bind: np.ndarray, mu_dev: np.ndarray,
                     sigma_dev: np.ndarray, tau_dev: float, *, beta_0: float = 1.0) -> LinkedPredictions:
    arrays = tuple(np.asarray(v, dtype=float) for v in (mu_bind, sigma_bind, mu_dev, sigma_dev))
    mu_b, sig_b, mu_d, sig_d = arrays
    penalty = QUALIFICATION_PENALTY * np.maximum(0, tau_dev - mu_d)
    score = BINDING_WEIGHT * (mu_b + beta_0 * sig_b) + DEVELOPABILITY_WEIGHT * (mu_d + beta_0 * sig_d) - penalty
    joint = np.sqrt(BINDING_WEIGHT * sig_b ** 2 + DEVELOPABILITY_WEIGHT * sig_d ** 2)
    return LinkedPredictions(mu_b, sig_b, mu_d, sig_d, score, joint)
