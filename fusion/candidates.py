"""Candidate collapse (design spec section 17.2 Step 1): every selection's achievable candidate
set is {neural-only argmax} union {fused_argmax at each beta in BETA_GRID} -- pure aggregation over
data that already exists in attribution.parquet, no new fusion computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID

_KEY = list(KEY_COLUMNS)


def achievable_candidates_by_key(attribution: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(attribution), dtype=bool)
    for beta in BETA_GRID:
        mask |= np.isclose(attribution["beta"], beta)
    grid = attribution[mask]

    def _collapse(group: pd.DataFrame) -> tuple[int, ...]:
        values = set(group["fused_argmax"].tolist())
        values.add(int(group["neural_argmax"].iloc[0]))
        return tuple(sorted(values))

    candidates = grid.groupby(_KEY, sort=False).apply(_collapse, include_groups=False)
    out = candidates.reset_index().rename(columns={0: "candidates"})
    out["n_distinct_candidates"] = out["candidates"].apply(len)
    return out
