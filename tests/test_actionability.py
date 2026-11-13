import numpy as np
import pandas as pd
import pytest

from fusion.actionability import actionability_by_key, beta_correctness_long
from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID


def _toy_attribution():
    # Two selections. Selection 1 ("s1") has the SAME fused_argmax at every beta -> not actionable.
    # Selection 2 ("s2") has a DIFFERENT fused_argmax at beta=4.0 -> actionable.
    rows = []
    key = dict(study="StudyF", participant_id="p1", session_id="sess1", trial_number=1,
               relative_path="s1.csv")
    for beta in BETA_GRID:
        rows.append({**key, "beta": beta, "fused_argmax": 5, "fused_correct": beta == 1.0})
    key2 = dict(study="StudyF", participant_id="p1", session_id="sess1", trial_number=2,
                relative_path="s2.csv")
    for beta in BETA_GRID:
        argmax = 7 if beta == 4.0 else 5
        rows.append({**key2, "beta": beta, "fused_argmax": argmax, "fused_correct": argmax == 7})
    return pd.DataFrame(rows)


def test_actionability_by_key_flags_only_the_varying_selection():
    result = actionability_by_key(_toy_attribution())
    assert set(result.columns) == set(KEY_COLUMNS) | {"actionable"}
    assert len(result) == 2
    by_trial = result.set_index("trial_number")["actionable"]
    assert by_trial.loc[1] == False
    assert by_trial.loc[2] == True


def test_beta_correctness_long_has_one_row_per_selection_beta_pair():
    result = beta_correctness_long(_toy_attribution())
    assert set(result.columns) == set(KEY_COLUMNS) | {"beta", "fused_correct"}
    assert len(result) == 2 * len(BETA_GRID)
    s2_beta4 = result[(result["trial_number"] == 2) & np.isclose(result["beta"], 4.0)]
    assert bool(s2_beta4["fused_correct"].iloc[0]) is True


def test_beta_correctness_long_restricts_to_beta_grid():
    attribution = _toy_attribution()
    extra = attribution.iloc[[0]].copy()
    extra["beta"] = 999.0  # not in BETA_GRID
    frame = pd.concat([attribution, extra], ignore_index=True)
    result = beta_correctness_long(frame)
    assert not (np.isclose(result["beta"], 999.0)).any()
