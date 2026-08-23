"""Assemble the four fusion arms' per-selection outcomes: no-LLM floor, fixed-weight
fusion (beta=1, the field default), adaptive fusion (this study's estimated beta), and
an oracle upper bound told the true archive outcome for that single trial only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.data import AUTHORSHIP_ROOT, SELECTION_KEY

from authorship.attribution import fuse, attribution_row  # noqa: E402

KEY_COLUMNS = SELECTION_KEY
OUTCOME_COLUMNS = list(KEY_COLUMNS) + [
    "arm", "fused_correct", "neural_correct", "phantom_agreement", "prior_capture",
]


def compute_adaptive_arm(frame: pd.DataFrame) -> pd.DataFrame:
    """frame needs p_neural, p_lm, target_code, beta_adaptive (one row per selection)."""
    rows = []
    for row in frame.itertuples(index=False):
        target_index = row.target_code - 1
        fused = fuse(row.p_neural, row.p_lm, beta=row.beta_adaptive)
        rows.append(attribution_row(row.p_neural, row.p_lm, fused, target_index))
    outcomes = pd.DataFrame(rows)
    outcomes["arm"] = "adaptive"
    keys = frame[list(KEY_COLUMNS)].reset_index(drop=True)
    return pd.concat([keys, outcomes], axis=1)


def select_floor_arm(attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """No-LLM baseline: whatever the neural posterior alone would have picked.

    Reused directly from the sibling attribution table -- neural_argmax/neural_correct
    do not depend on beta or on which prior model produced the row.
    """
    one_beta = attribution_frame[np.isclose(attribution_frame["beta"], 1.0)]
    floor = one_beta[list(KEY_COLUMNS) + ["neural_argmax", "neural_correct"]].copy()
    floor = floor.rename(columns={"neural_argmax": "fused_argmax", "neural_correct": "fused_correct"})
    # Keep neural_correct too (in addition to the fused_correct rename above, not instead of it --
    # other callers rely on that rename): for the floor arm, no LM is used, so fused_correct IS
    # neural_correct by construction. This is what makes add_net_correct_gain's
    # fused_correct - neural_correct exactly 0.0 for every floor row.
    floor["neural_correct"] = floor["fused_correct"]
    floor["arm"] = "floor"
    floor["phantom_agreement"] = False
    floor["prior_capture"] = False
    return floor.reset_index(drop=True)


def select_fixed_arm(attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """Field-default fixed-weight fusion (beta=1.0), reused without recomputation."""
    fixed = attribution_frame[np.isclose(attribution_frame["beta"], 1.0)].copy()
    fixed["arm"] = "fixed"
    return fixed.reset_index(drop=True)


def select_oracle_arm(attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-selection best-of-existing-grid choice, using only that selection's own known
    outcome (never an aggregate or a neighboring trial's outcome) -- an upper bound, not
    a deployable policy.
    """
    grid_sizes = attribution_frame.groupby(list(KEY_COLUMNS))["beta"].nunique()
    if grid_sizes.min() < 2:
        raise ValueError(
            "select_oracle_arm requires the full beta grid per selection "
            "(got a frame with fewer than 2 distinct beta values for at least one selection) "
            "-- pass the unfiltered attribution frame, not a single-beta slice"
        )
    ranked = attribution_frame.sort_values(["fused_correct", "p_fused_target"], ascending=[False, False])
    oracle = ranked.groupby(list(KEY_COLUMNS), as_index=False).first()
    oracle["arm"] = "oracle"
    return oracle.reset_index(drop=True)


def assemble_all_arms(working_frame_with_beta: pd.DataFrame, attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """One combined long frame across all four arms with matching outcome columns."""
    adaptive = compute_adaptive_arm(working_frame_with_beta)
    parts = [
        select_floor_arm(attribution_frame)[OUTCOME_COLUMNS],
        select_fixed_arm(attribution_frame)[OUTCOME_COLUMNS],
        adaptive[OUTCOME_COLUMNS],
        select_oracle_arm(attribution_frame)[OUTCOME_COLUMNS],
    ]
    return pd.concat(parts, ignore_index=True)
