import numpy as np
import pandas as pd
import pytest

from fusion.route_a_stage1 import fit_actionability_model, fit_actionability_model_frozen


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


def _toy_actionability(dev_frame):
    rng = np.random.default_rng(1)
    keys = dev_frame[["study", "participant_id", "session_id", "trial_number", "relative_path"]].copy()
    keys["actionable"] = rng.integers(0, 2, size=len(keys)).astype(bool)
    return keys


def test_fit_actionability_model_returns_one_probability_per_dev_row():
    dev_frame = _toy_dev()
    dev_actionability = _toy_actionability(dev_frame)
    p_hat = fit_actionability_model(dev_frame, dev_actionability, n_folds=3)
    assert len(p_hat) == len(dev_frame)
    assert ((p_hat >= 0) & (p_hat <= 1)).all()


def test_fit_actionability_model_frozen_returns_one_probability_per_test_row():
    dev_frame = _toy_dev()
    dev_actionability = _toy_actionability(dev_frame)
    test_frame = _toy_dev(n_participants=2, per_participant=4)
    p_hat = fit_actionability_model_frozen(dev_frame, dev_actionability, test_frame)
    assert len(p_hat) == len(test_frame)
    assert ((p_hat >= 0) & (p_hat <= 1)).all()


def test_fit_actionability_model_raises_when_label_join_drops_rows():
    dev_frame = _toy_dev()
    dev_actionability = _toy_actionability(dev_frame).iloc[1:]
    with pytest.raises(ValueError, match="dropped rows"):
        fit_actionability_model(dev_frame, dev_actionability, n_folds=3)


def test_fit_actionability_model_frozen_raises_when_label_join_drops_rows():
    dev_frame = _toy_dev()
    dev_actionability = _toy_actionability(dev_frame).iloc[1:]
    test_frame = _toy_dev(n_participants=2, per_participant=4)
    with pytest.raises(ValueError, match="dropped rows"):
        fit_actionability_model_frozen(dev_frame, dev_actionability, test_frame)


def test_fit_actionability_model_frozen_is_index_aligned_to_test_frame():
    dev_frame = _toy_dev()
    dev_actionability = _toy_actionability(dev_frame)
    test_frame = _toy_dev(n_participants=2, per_participant=4)
    test_frame.index = [10 + i for i in range(len(test_frame))]
    p_hat = fit_actionability_model_frozen(dev_frame, dev_actionability, test_frame)
    assert list(p_hat.index) == list(test_frame.index)
