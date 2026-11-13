"""Route A Stage 1: out-of-fold P-hat(actionable | x) from the same
joint neural+LM feature set the joint gate already validated. This module deliberately
reimplements the small CV/scaling pattern from `fusion.arbitration` rather than importing its
underscore-prefixed internals, matching the DRY exception already used by
`fusion/arbitration_diagnostics.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fusion.arms import KEY_COLUMNS
from fusion.joint_features import joint_features_frame
from fusion.reliability import RANDOM_STATE

_KEY = list(KEY_COLUMNS)


def _new_classifier() -> LogisticRegression:
    return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    features = joint_features_frame(frame["p_neural"], frame["p_lm"])
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()


def _fit_probability_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int) -> np.ndarray:
    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")
    splitter = GroupKFold(n_splits=n_folds)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a Route A Stage 1 CV fold")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = _new_classifier()
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    return oof


def fit_actionability_model(
    dev_frame: pd.DataFrame, dev_actionability: pd.DataFrame, n_folds: int = 5
) -> pd.Series:
    """Out-of-fold P-hat(actionable | x), participant-grouped cross-validation, indexed to the
    merged frame's positional index (same convention as `fusion.arbitration.fit_net_benefit_model`).
    """
    merged = dev_frame.merge(dev_actionability[_KEY + ["actionable"]], on=_KEY, how="inner")
    if len(merged) != len(dev_frame):
        raise ValueError(
            f"actionability label join dropped rows: {len(dev_frame)} dev selections -> "
            f"{len(merged)} labelled rows"
        )
    X = _feature_matrix(merged)
    y = merged["actionable"].astype(int).to_numpy()
    groups = merged["participant_id"].to_numpy()
    oof = _fit_probability_cv(X, y, groups, n_folds)
    return pd.Series(oof, index=merged.index, name="p_actionable")


def fit_actionability_model_frozen(
    dev_frame: pd.DataFrame, dev_actionability: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.Series:
    """The FROZEN actionability classifier: fit on ALL of dev, applied to `test_frame`. No holdout
    label enters the fit path -- `test_frame` supplies features only.
    """
    merged = dev_frame.merge(dev_actionability[_KEY + ["actionable"]], on=_KEY, how="inner")
    if len(merged) != len(dev_frame):
        raise ValueError(
            f"actionability label join dropped rows: {len(dev_frame)} dev selections -> "
            f"{len(merged)} labelled rows"
        )
    dev_features = _feature_matrix(merged)
    scaler = StandardScaler().fit(dev_features)
    X_dev = scaler.transform(dev_features)
    X_test = scaler.transform(_feature_matrix(test_frame))

    model = _new_classifier()
    model.fit(X_dev, merged["actionable"].astype(int).to_numpy())
    p_hat = model.predict_proba(X_test)[:, 1]
    return pd.Series(p_hat, index=test_frame.index, name="p_actionable")
