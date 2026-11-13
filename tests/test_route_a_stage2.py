import numpy as np
import pandas as pd
import pytest

from fusion.rescue import BETA_GRID
from fusion.route_a_stage2 import fit_beta_utility_models, fit_beta_utility_models_frozen

_KEY_COLS = ["study", "participant_id", "session_id", "trial_number", "relative_path"]


def _toy_dev(n_participants=6, per_participant=10):
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_participants):
        for t in range(per_participant):
            p_neural = rng.dirichlet(np.ones(36))
            p_lm = rng.dirichlet(np.ones(36))
            rows.append({
                "study": "StudyF", "participant_id": f"p{p}", "session_id": "s1",
                "trial_number": t, "relative_path": f"p{p}_{t}.csv",
                "p_neural": p_neural, "p_lm": p_lm,
            })
    return pd.DataFrame(rows)


def _toy_beta_correctness(dev_frame):
    rng = np.random.default_rng(2)
    rows = []
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    for _, row in dev_frame[key_cols].iterrows():
        for beta in BETA_GRID:
            rows.append({**row.to_dict(), "beta": beta, "fused_correct": bool(rng.integers(0, 2))})
    return pd.DataFrame(rows)


def test_fit_beta_utility_models_returns_one_row_per_selection_beta_pair():
    dev_frame = _toy_dev()
    dev_beta_correctness = _toy_beta_correctness(dev_frame)
    q_hat = fit_beta_utility_models(dev_frame, dev_beta_correctness, n_folds=3)
    assert len(q_hat) == len(dev_frame) * len(BETA_GRID)
    assert set(q_hat["beta"].unique()) == set(BETA_GRID)
    assert ((q_hat["q_hat"] >= 0) & (q_hat["q_hat"] <= 1)).all()


def test_fit_beta_utility_models_frozen_returns_one_row_per_test_selection_beta_pair():
    dev_frame = _toy_dev()
    dev_beta_correctness = _toy_beta_correctness(dev_frame)
    test_frame = _toy_dev(n_participants=2, per_participant=4)
    q_hat = fit_beta_utility_models_frozen(dev_frame, dev_beta_correctness, test_frame)
    assert len(q_hat) == len(test_frame) * len(BETA_GRID)
    assert set(q_hat["beta"].unique()) == set(BETA_GRID)


def test_fit_beta_utility_models_raises_when_a_beta_label_join_drops_rows():
    dev_frame = _toy_dev()
    dev_beta_correctness = _toy_beta_correctness(dev_frame)
    dropped_beta = BETA_GRID[0]
    is_dropped_row = (
        np.isclose(dev_beta_correctness["beta"], dropped_beta)
        & (dev_beta_correctness["participant_id"] == "p0")
        & (dev_beta_correctness["trial_number"] == 0)
    )
    dev_beta_correctness = dev_beta_correctness[~is_dropped_row].reset_index(drop=True)
    with pytest.raises(ValueError):
        fit_beta_utility_models(dev_frame, dev_beta_correctness, n_folds=3)


def test_fit_beta_utility_models_frozen_aligns_to_test_frame_keys_with_nondefault_index():
    dev_frame = _toy_dev()
    dev_beta_correctness = _toy_beta_correctness(dev_frame)
    test_frame = _toy_dev(n_participants=2, per_participant=4)
    test_frame = test_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    test_frame.index = test_frame.index + 1000

    q_hat = fit_beta_utility_models_frozen(dev_frame, dev_beta_correctness, test_frame)

    expected_keys = pd.concat([test_frame[_KEY_COLS]] * len(BETA_GRID), ignore_index=True)
    pd.testing.assert_frame_equal(
        q_hat[_KEY_COLS].reset_index(drop=True), expected_keys.reset_index(drop=True)
    )
