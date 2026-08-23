from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fusion.transport import leave_one_study_out_contrast


def _fake_working_frame(n_per_study=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for study in ["StudyB", "StudyF", "StudyL", "StudyN"]:
        for i in range(n_per_study):
            sharp = rng.random() < 0.5
            if sharp:
                p_neural = np.full(36, 0.1 / 35)
                p_neural[0] = 0.9
                correct = rng.random() < 0.9
            else:
                p_neural = np.full(36, 1 / 36)
                correct = rng.random() < 1 / 36
            target_code = 1 if correct else 2
            p_lm = np.full(36, 1 / 36)
            rows.append({
                "study": study, "participant_id": f"{study}_P{i % 10}", "session_id": "S1",
                "trial_number": i, "relative_path": f"{study}_{i}.edf",
                "p_neural": p_neural, "p_lm": p_lm,
                "target_code": target_code, "neural_correct": bool(correct),
            })
    return pd.DataFrame(rows)


def test_leave_one_study_out_returns_one_bounded_row_per_study():
    working = _fake_working_frame()
    attribution = working.copy()
    attribution["beta"] = 1.0
    attribution["fused_correct"] = attribution["neural_correct"]
    attribution["phantom_agreement"] = False
    attribution["prior_capture"] = False
    result = leave_one_study_out_contrast(working, attribution)
    assert len(result) == 4
    assert set(result["held_out_study"]) == {"StudyB", "StudyF", "StudyL", "StudyN"}
    assert (result["lower"] <= result["point"]).all()
    assert (result["point"] <= result["upper"]).all()


def test_leave_one_study_out_never_trains_on_the_held_out_study():
    working = _fake_working_frame(n_per_study=60, seed=2)
    attribution = working.copy()
    attribution["beta"] = 1.0
    attribution["fused_correct"] = attribution["neural_correct"]
    attribution["phantom_agreement"] = False
    attribution["prior_capture"] = False

    recorded_fit_row_counts = []
    original_fit = LogisticRegression.fit

    def spy_fit(self, X, y):
        recorded_fit_row_counts.append(len(X))
        return original_fit(self, X, y)

    with mock.patch.object(LogisticRegression, "fit", spy_fit):
        leave_one_study_out_contrast(working, attribution)

    n_per_study = 60
    total_rows = len(working)
    assert len(recorded_fit_row_counts) == 4
    for count in recorded_fit_row_counts:
        assert count == total_rows - n_per_study, (
            f"expected each fold's training set to exclude exactly one study's "
            f"{n_per_study} rows, got a training set of size {count}"
        )
