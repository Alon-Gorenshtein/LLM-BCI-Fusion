"""Map an estimated neural-reliability probability to a fusion weight.

Beta = 1.0 is the field's fixed-weight default (see the sibling attribution study's
`authorship.attribution.fuse`). The sibling analysis already swept beta over
{0.25, 0.5, 1.0, 2.0, 4.0}; this mapping is anchored to that same range so the adaptive
arm is directly comparable to the existing grid rather than an unrelated new scale.
"""
from __future__ import annotations

import numpy as np

BETA_MIN = 0.25
BETA_MAX = 4.0


def reliability_to_beta(g_hat) -> np.ndarray:
    """g_hat=0 -> BETA_MIN, g_hat=0.5 -> 1.0 (the geometric mean), g_hat=1 -> BETA_MAX."""
    g = np.clip(np.asarray(g_hat, dtype=float), 1e-6, 1 - 1e-6)
    return BETA_MIN * (BETA_MAX / BETA_MIN) ** g
