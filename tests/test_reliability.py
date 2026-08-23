import numpy as np
import pandas as pd
import pytest

from fusion.reliability import (
    fit_reliability_cv, reliability_auroc, reliability_calibration_error,
)


def _synthetic_frame(n_participants=20, per_participant=60, seed=0):
    """Rows where a sharply peaked posterior (high margin) is correct more often."""
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(n_participants):
        for _ in range(per_participant):
            sharp = rng.random() < 0.5
            if sharp:
                p = np.full(36, 0.1 / 35)
                p[0] = 0.9
                correct = rng.random() < 0.9
            else:
                p = np.full(36, 1 / 36)
                correct = rng.random() < 1 / 36
            rows.append({
                "participant_id": f"P{participant}",
                "p_neural": p,
                "neural_correct": bool(correct),
            })
    return pd.DataFrame(rows)


def test_fit_reliability_cv_adds_a_bounded_g_hat_column():
    frame = _synthetic_frame()
    result = fit_reliability_cv(frame, n_folds=5)
    assert "g_hat" in result.columns
    assert result["g_hat"].between(0, 1).all()


def test_higher_margin_predicts_correctness_out_of_fold():
    frame = _synthetic_frame(seed=1)
    result = fit_reliability_cv(frame, n_folds=5)
    assert reliability_auroc(result) > 0.6


def test_n_folds_cannot_exceed_participant_count():
    frame = _synthetic_frame(n_participants=3, per_participant=5)
    with pytest.raises(ValueError):
        fit_reliability_cv(frame, n_folds=5)


def test_calibration_error_is_near_zero_for_well_calibrated_predictions():
    frame = pd.DataFrame({
        "g_hat": [0.9] * 10 + [0.1] * 10,
        "neural_correct": [True] * 9 + [False] * 1 + [False] * 9 + [True] * 1,
    })
    assert reliability_calibration_error(frame, n_bins=10) < 0.05
