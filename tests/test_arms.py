import numpy as np
import pandas as pd
import pytest

from fusion.arms import (
    assemble_all_arms, compute_adaptive_arm, select_floor_arm, select_fixed_arm, select_oracle_arm,
)


def _one_selection_attribution_grid():
    """One selection, beta grid {0.25,0.5,1,2,4}; correct only when beta trusts neural."""
    rows = []
    for beta, fused_correct, p_target in [
        (0.25, False, 0.10), (0.5, False, 0.15), (1.0, False, 0.20),
        (2.0, True, 0.60), (4.0, True, 0.80),
    ]:
        rows.append({
            "study": "StudyB", "participant_id": "P1", "session_id": "S1", "trial_number": 1,
            "relative_path": "StudyB/P1/S1/file1.edf",
            "beta": beta, "fused_correct": fused_correct, "p_fused_target": p_target,
            "neural_argmax": 3, "neural_correct": True,
            "phantom_agreement": False, "prior_capture": not fused_correct,
        })
    return pd.DataFrame(rows)


def test_oracle_arm_picks_the_best_beta_for_each_selection():
    oracle = select_oracle_arm(_one_selection_attribution_grid())
    assert len(oracle) == 1
    assert bool(oracle.loc[0, "fused_correct"]) is True
    assert oracle.loc[0, "p_fused_target"] == pytest.approx(0.80)


def test_select_oracle_arm_rejects_a_single_beta_slice():
    """A pre-filtered single-beta frame (e.g. load_attribution(model, beta=1.0)) has no grid
    to pick the best-of over -- select_oracle_arm must raise rather than silently degenerate
    into a copy of the fixed arm.
    """
    grid = _one_selection_attribution_grid()
    single_beta = grid[np.isclose(grid["beta"], 1.0)]
    with pytest.raises(ValueError):
        select_oracle_arm(single_beta)


def test_fixed_arm_is_exactly_the_beta_one_rows():
    fixed = select_fixed_arm(_one_selection_attribution_grid())
    assert len(fixed) == 1
    assert bool(fixed.loc[0, "fused_correct"]) is False


def test_floor_arm_uses_neural_only_outcome_and_has_no_lm_effects():
    floor = select_floor_arm(_one_selection_attribution_grid())
    assert len(floor) == 1
    assert bool(floor.loc[0, "fused_correct"]) is True
    assert bool(floor.loc[0, "phantom_agreement"]) is False
    assert bool(floor.loc[0, "prior_capture"]) is False


def test_compute_adaptive_arm_runs_fuse_with_the_given_beta():
    p_neural = np.zeros(36)
    p_neural[2] = 1.0
    p_lm = np.full(36, 1 / 36)
    frame = pd.DataFrame([{
        "study": "StudyB", "participant_id": "P1", "session_id": "S1", "trial_number": 1,
        "relative_path": "StudyB/P1/S1/file1.edf",
        "p_neural": p_neural, "p_lm": p_lm, "target_code": 3, "beta_adaptive": 4.0,
    }])
    result = compute_adaptive_arm(frame)
    assert len(result) == 1
    assert result.loc[0, "arm"] == "adaptive"
    assert bool(result.loc[0, "fused_correct"]) is True


def test_assemble_all_arms_combines_all_four_arms_into_one_long_frame():
    """Direct test for assemble_all_arms -- the one function that combines all four arms into
    the long-frame shape every downstream stage (stats, ladder, transport) consumes.
    """
    attribution_frame = _one_selection_attribution_grid()
    p_neural = np.zeros(36)
    p_neural[2] = 1.0
    p_lm = np.full(36, 1 / 36)
    working_frame_with_beta = pd.DataFrame([{
        "study": "StudyB", "participant_id": "P1", "session_id": "S1", "trial_number": 1,
        "relative_path": "StudyB/P1/S1/file1.edf",
        "p_neural": p_neural, "p_lm": p_lm, "target_code": 3, "beta_adaptive": 4.0,
    }])

    combined = assemble_all_arms(working_frame_with_beta, attribution_frame)

    assert len(combined) == 4
    assert set(combined["arm"]) == {"floor", "fixed", "adaptive", "oracle"}


def test_select_oracle_arm_does_not_collapse_distinct_selections_sharing_trial_number():
    """Two DIFFERENT selections that happen to share (study, participant_id, session_id,
    trial_number) but differ in relative_path (exactly as real bigP3BCI data does --
    trial_number resets per condition/file within a session) must be treated as separate
    oracle picks, not collapsed into one.
    """
    rows = []
    for beta, fused_correct, p_target in [(0.25, False, 0.10), (4.0, True, 0.80)]:
        rows.append({
            "study": "StudyB", "participant_id": "P1", "session_id": "S1", "trial_number": 1,
            "relative_path": "fileA.edf",
            "beta": beta, "fused_correct": fused_correct, "p_fused_target": p_target,
        })
    for beta, fused_correct, p_target in [(0.25, True, 0.95), (4.0, False, 0.05)]:
        rows.append({
            "study": "StudyB", "participant_id": "P1", "session_id": "S1", "trial_number": 1,
            "relative_path": "fileB.edf",
            "beta": beta, "fused_correct": fused_correct, "p_fused_target": p_target,
        })
    frame = pd.DataFrame(rows)
    oracle = select_oracle_arm(frame)
    assert len(oracle) == 2, "two distinct selections must produce two oracle rows, not one collapsed row"
    by_path = oracle.set_index("relative_path")
    assert by_path.loc["fileA.edf", "p_fused_target"] == pytest.approx(0.80)
    assert by_path.loc["fileB.edf", "p_fused_target"] == pytest.approx(0.95)
