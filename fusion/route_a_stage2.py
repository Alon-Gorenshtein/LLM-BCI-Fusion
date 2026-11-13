"""Route A Stage 2: out-of-fold P-hat(fused_correct=1 | x, beta) for
each beta in BETA_GRID, fit on ALL dev rows regardless of actionability -- actionability (Stage 1)
gates whether this output is CONSULTED, not whether the model is trained. Same CV/scaling pattern as
Stage 1 (`fusion.route_a_stage1`), reimplemented locally per this plan's DRY exception rather than
imported, since these are two independently testable modules.
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
from fusion.rescue import BETA_GRID

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
            raise RuntimeError("group leakage across a Route A Stage 2 CV fold")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = _new_classifier()
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    return oof


def fit_beta_utility_models(
    dev_frame: pd.DataFrame, dev_beta_correctness: pd.DataFrame, n_folds: int = 5
) -> pd.DataFrame:
    """KEY_COLUMNS + ['beta', 'q_hat'], one row per (selection, beta), out-of-fold. Fits one
    classifier per beta in BETA_GRID, each on all dev selections labelled at that beta.
    """
    X_dev = _feature_matrix(dev_frame)
    groups = dev_frame["participant_id"].to_numpy()
    out_frames = []
    for beta in BETA_GRID:
        labels = dev_beta_correctness[np.isclose(dev_beta_correctness["beta"], beta)]
        merged = dev_frame[_KEY].merge(
            labels[_KEY + ["fused_correct"]], on=_KEY, how="inner", validate="one_to_one"
        )
        if len(merged) != len(dev_frame):
            raise ValueError(
                f"beta={beta} label join dropped rows: {len(dev_frame)} dev selections -> "
                f"{len(merged)} labelled rows"
            )
        y = merged["fused_correct"].astype(int).to_numpy()
        oof = _fit_probability_cv(X_dev, y, groups, n_folds)
        beta_out = merged[_KEY].copy()
        beta_out["beta"] = beta
        beta_out["q_hat"] = oof
        out_frames.append(beta_out)
    return pd.concat(out_frames, ignore_index=True)


def fit_beta_utility_models_frozen(
    dev_frame: pd.DataFrame, dev_beta_correctness: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.DataFrame:
    """The FROZEN per-beta utility estimator: each of the 5 classifiers fit on ALL of dev, applied
    to `test_frame`. No holdout label enters any fit -- `test_frame` supplies features only.
    """
    dev_features = _feature_matrix(dev_frame)
    scaler = StandardScaler().fit(dev_features)
    X_dev = scaler.transform(dev_features)
    X_test = scaler.transform(_feature_matrix(test_frame))

    out_frames = []
    for beta in BETA_GRID:
        labels = dev_beta_correctness[np.isclose(dev_beta_correctness["beta"], beta)]
        merged = dev_frame[_KEY].merge(
            labels[_KEY + ["fused_correct"]], on=_KEY, how="inner", validate="one_to_one"
        )
        if len(merged) != len(dev_frame):
            raise ValueError(
                f"beta={beta} label join dropped rows: {len(dev_frame)} dev selections -> "
                f"{len(merged)} labelled rows"
            )
        model = _new_classifier()
        model.fit(X_dev, merged["fused_correct"].astype(int).to_numpy())
        q_hat = model.predict_proba(X_test)[:, 1]
        beta_out = test_frame[_KEY].copy()
        beta_out["beta"] = beta
        beta_out["q_hat"] = q_hat
        out_frames.append(beta_out)
    return pd.concat(out_frames, ignore_index=True)
