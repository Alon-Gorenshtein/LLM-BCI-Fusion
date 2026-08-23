"""Candidate-ranking model and the conservative, exact-identity-by-default combining rule (design
spec section 17.2 Steps 3-4). Reimplements the small CV/scaling pattern locally rather than
importing route_a_stage1/2's underscore-prefixed internals, per this plan's DRY exception.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fusion.arms import KEY_COLUMNS
from fusion.reliability import RANDOM_STATE

_KEY = list(KEY_COLUMNS)
_FEATURE_COLUMNS = [
    "neural_p_candidate", "lm_p_candidate", "boundary_distance",
    "stability_interval_width", "neural_lm_log_odds_candidate",
]
DELTA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def _new_classifier() -> LogisticRegression:
    return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    features = frame[_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.to_numpy()


def fit_candidate_ranker(train_features: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    X = _feature_matrix(train_features)
    y = train_features["is_target"].astype(int).to_numpy()
    groups = train_features["participant_id"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")
    splitter = GroupKFold(n_splits=n_folds)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a candidate-ranker CV fold")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = _new_classifier()
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    return pd.Series(oof, index=train_features.index, name="p_hat")


def fit_candidate_ranker_frozen(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> pd.Series:
    X_train_raw = _feature_matrix(train_features)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(_feature_matrix(test_features))
    model = _new_classifier()
    model.fit(X_train, train_features["is_target"].astype(int).to_numpy())
    p_hat = model.predict_proba(X_test)[:, 1]
    return pd.Series(p_hat, index=test_features.index, name="p_hat")


def apply_combining_rule(
    features: pd.DataFrame, p_hat: pd.Series, delta: float
) -> pd.DataFrame:
    scored = features[_KEY + ["candidate_index", "is_fixed_winner"]].copy()
    scored["p_hat"] = pd.Series(p_hat).to_numpy()

    rows = []
    for keys, group in scored.groupby(_KEY, sort=False):
        fixed_row = group[group["is_fixed_winner"]].iloc[0]
        best_row = group.loc[group["p_hat"].idxmax()]
        override = (
            not bool(best_row["is_fixed_winner"])
            and float(best_row["p_hat"]) - float(fixed_row["p_hat"]) > delta
        )
        chosen = int(best_row["candidate_index"]) if override else int(fixed_row["candidate_index"])
        rows.append({
            **dict(zip(_KEY, keys)),
            "chosen_candidate": chosen,
            "overridden": override,
            "p_hat_fixed_winner": float(fixed_row["p_hat"]),
            "p_hat_best_alternative": float(best_row["p_hat"]),
        })
    return pd.DataFrame(rows)


def select_delta(
    train_features: pd.DataFrame,
    p_hat: pd.Series,
    delta_grid: tuple[float, ...] = DELTA_GRID,
) -> float:
    """Grid-search delta on out-of-fold p_hat, maximizing the fraction of selections where the
    chosen candidate is correct (equivalently fused selection correctness, since neural_correct is
    candidate-invariant) versus the fixed-winner baseline. Correctness is read directly off
    train_features' own is_target column for the chosen candidate row -- candidate_feature_table
    already carries ground truth per (selection, candidate), so no separate attribution/beta lookup
    is needed here.
    """
    best_score = -np.inf
    best_delta = delta_grid[0]
    for delta in delta_grid:
        applied = apply_combining_rule(train_features, p_hat, delta)
        # A chosen candidate is correct iff it equals the true target -- read this directly off
        # is_target for that (selection, candidate) row, avoiding a second join through beta.
        target_lookup = train_features[_KEY + ["candidate_index", "is_target"]]
        scored = applied.merge(
            target_lookup, left_on=_KEY + ["chosen_candidate"],
            right_on=_KEY + ["candidate_index"], how="inner", validate="one_to_one",
        )
        if len(scored) != len(applied):
            raise ValueError(
                f"delta={delta} scoring join dropped rows: {len(applied)} -> {len(scored)}"
            )
        score = float(scored["is_target"].astype(int).mean())
        if score > best_score:
            best_score = score
            best_delta = delta
    return float(best_delta)
