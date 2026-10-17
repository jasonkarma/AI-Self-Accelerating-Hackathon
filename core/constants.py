"""Scientific constants frozen by SPEC v1.2."""
from __future__ import annotations

import numpy as np

WT_SEQUENCE = "SRWGGDGFYAMDY"
SEQUENCE_LENGTH = 13
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ANCHOR_POSITIONS = (3, 11, 12, 13)

HYDROPATHY = dict(zip(AMINO_ACIDS, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-4.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
CHARGE = {aa: 0.0 for aa in AMINO_ACIDS}
CHARGE.update({"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0})
# Approximate side-chain van der Waals volumes (A^3), normalized once globally.
_VOLUME_RAW = dict(zip(AMINO_ACIDS, [67,86,91,109,135,48,118,124,135,124,124,96,90,148,129,73,93,105,163,141]))
_v = np.asarray(list(_VOLUME_RAW.values()), dtype=float)
VOLUME = {aa: float((value - _v.min()) / (_v.max() - _v.min())) for aa, value in _VOLUME_RAW.items()}

RISK_COMPONENTS = ("hydro", "window", "aromatic", "charge", "cluster", "liability")
RISK_WEIGHTS = np.asarray([0.25, 0.25, 0.20, 0.10, 0.10, 0.10], dtype=float)
BINDING_WEIGHT = 0.65
DEVELOPABILITY_WEIGHT = 0.35
BETA_0 = 1.0
QUALIFICATION_PENALTY = 2.0
N_BOOTSTRAPS = 50
SOFT_PRIOR_BOUND = 0.15
