"""Stage 0 of the Route A controller (design spec section 15.2): derive actionability and per-beta
counterfactual correctness labels from data that already exists in `attribution.parquet` -- no new
fusion or posterior computation. `fused_argmax` and `fused_correct` are already columns of the
long, beta-indexed attribution table for every selection at every beta in `fusion.rescue.BETA_GRID`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID

_KEY = list(KEY_COLUMNS)


def _restrict_to_beta_grid(attribution: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(attribution), dtype=bool)
    for beta in BETA_GRID:
        mask |= np.isclose(attribution["beta"], beta)
    return attribution[mask]


def actionability_by_key(attribution: pd.DataFrame) -> pd.DataFrame:
    """One row per selection: `actionable` is True iff `fused_argmax` differs across at least two of
    the BETA_GRID rows for that selection.
    """
    grid = _restrict_to_beta_grid(attribution)

    def _varies(group: pd.DataFrame) -> bool:
        return bool(group["fused_argmax"].nunique() > 1)

    actionable = grid.groupby(_KEY, sort=False).apply(_varies, include_groups=False)
    out = actionable.reset_index()
    out = out.rename(columns={0: "actionable"})
    out["actionable"] = out["actionable"].astype(bool)
    return out


def beta_correctness_long(attribution: pd.DataFrame) -> pd.DataFrame:
    """KEY_COLUMNS + ["beta", "fused_correct"], restricted to beta in BETA_GRID, one row per
    (selection, beta) pair. This is Y_i(beta) in long format.
    """
    grid = _restrict_to_beta_grid(attribution)
    return grid[_KEY + ["beta", "fused_correct"]].reset_index(drop=True)
