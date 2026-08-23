"""Online reliability estimator: neural-posterior shape -> P(neural selection correct),
fit via subject-level (never same-fold) cross-validation so the adaptive fusion arm never
benefits from information about its own evaluation subjects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from fusion.features import reliability_features_frame

RANDOM_STATE = 20260820


def fit_reliability_cv(frame: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Return frame with an added `g_hat` column: out-of-fold P(neural_correct).

    frame must have columns p_neural (36-vector), neural_correct (bool), participant_id.
    Splitting is grouped by participant_id: a participant's rows are never in both the
    training and evaluation side of the same fold.
    """
    features = reliability_features_frame(frame["p_neural"])
    labels = frame["neural_correct"].astype(int).to_numpy()
    groups = frame["participant_id"].to_numpy()

    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")

    splitter = GroupKFold(n_splits=n_folds)
    g_hat = np.full(len(frame), np.nan)
    for train_idx, test_idx in splitter.split(features, labels, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a reliability CV fold")
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(features.iloc[train_idx], labels[train_idx])
        g_hat[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]

    if np.isnan(g_hat).any():
        raise RuntimeError("some rows never received an out-of-fold prediction")

    result = frame.copy()
    result["g_hat"] = g_hat
    return result


def reliability_auroc(frame_with_g_hat: pd.DataFrame) -> float:
    return float(
        roc_auc_score(frame_with_g_hat["neural_correct"].astype(int), frame_with_g_hat["g_hat"])
    )


def reliability_calibration_error(frame_with_g_hat: pd.DataFrame, n_bins: int = 10) -> float:
    """Expected calibration error of g_hat against realized neural_correct."""
    labels = frame_with_g_hat["neural_correct"].astype(int).to_numpy()
    probs = frame_with_g_hat["g_hat"].to_numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_confidence - bin_accuracy)
    return float(ece)
