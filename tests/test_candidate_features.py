import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.candidate_features import candidate_feature_table
from fusion.rescue import BETA_GRID


def _key(trial):
    return dict(study="StudyF", participant_id="p1", session_id="s1", trial_number=trial,
                relative_path=f"t{trial}.csv")


def _one_hot_ish(index, n=36, peak=0.6):
    vector = np.full(n, (1 - peak) / (n - 1))
    vector[index] = peak
    return vector


def test_candidate_feature_table_has_one_row_per_selection_candidate_pair():
    p_neural = _one_hot_ish(3)
    p_lm = _one_hot_ish(5)
    frame = pd.DataFrame([{**_key(1), "p_neural": p_neural, "p_lm": p_lm, "target_code": 4}])

    rows = []
    for beta in BETA_GRID:
        rows.append({**_key(1), "beta": beta, "fused_argmax": 5, "neural_argmax": 3})
    attribution = pd.DataFrame(rows)

    result = candidate_feature_table(frame, attribution, fixed_beta=1.0)

    assert set(KEY_COLUMNS).issubset(result.columns)
    assert "candidate_index" in result.columns and "is_target" in result.columns
    # candidates = {3, 5} (neural_argmax=3 union {fused_argmax=5 at every grid beta})
    assert set(result["candidate_index"]) == {3, 5}
    assert len(result) == 2
    # target_code=4 -> target_index=3, matching the neural-only candidate here.
    row_3 = result[result["candidate_index"] == 3].iloc[0]
    row_5 = result[result["candidate_index"] == 5].iloc[0]
    assert bool(row_3["is_target"]) is True
    assert bool(row_5["is_target"]) is False
    assert bool(row_3["is_neural_winner"]) is True
    assert bool(row_5["is_neural_winner"]) is False


def test_candidate_feature_table_boundary_distance_is_zero_for_the_fixed_winner():
    p_neural = _one_hot_ish(3)
    p_lm = _one_hot_ish(3)  # both sources agree on 3 -> 3 is the fixed-beta winner
    frame = pd.DataFrame([{**_key(1), "p_neural": p_neural, "p_lm": p_lm, "target_code": 4}])
    rows = [{**_key(1), "beta": beta, "fused_argmax": 3, "neural_argmax": 3} for beta in BETA_GRID]
    attribution = pd.DataFrame(rows)

    result = candidate_feature_table(frame, attribution, fixed_beta=1.0)
    assert len(result) == 1  # only one achievable candidate here
    assert result.iloc[0]["boundary_distance"] == 0.0
    assert bool(result.iloc[0]["is_fixed_winner"]) is True
