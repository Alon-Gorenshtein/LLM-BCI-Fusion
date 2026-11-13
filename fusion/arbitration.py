"""The joint arbitration gate: predicts the expected net benefit of
language-model influence directly (V(x) = P(phantom agreement | x) - P(prior capture | x)) from
joint neural+LM features, then makes a small, restricted-grid residual adjustment around the
already-validated best fixed beta, rather than reinventing the fusion policy from an unconstrained
link.

Every model in this module is FIT on dev alone. The one function that touches the holdout at all,
`fit_net_benefit_model_frozen`, reads the holdout's features to predict on them and never its
labels -- so no Study N outcome influences any parameter here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fusion.joint_features import joint_features_frame
from fusion.arms import KEY_COLUMNS
from fusion.reliability import RANDOM_STATE


def _new_classifier() -> LogisticRegression:
    """The one place the gate's classifier is constructed, so the cross-validated and frozen fits
    cannot drift apart in solver, seed, or iteration budget.
    """
    return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Joint neural+LM features as the gate consumes them. entropy_ratio is +inf whenever the LM
    posterior is a point mass, so infinities are neutralized to 0 rather than left to poison the
    fit. Shared by the cross-validated and frozen fits so both see the same representation.
    """
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
            raise RuntimeError("group leakage across an arbitration-gate CV fold")
        # Joint features mix KL/JS divergence (large, wide-ranging) with probabilities/margins
        # in [0, 1]; unscaled inputs caused convergence/overflow issues in unregularized
        # LogisticRegression during exploratory analysis. Here the resulting
        # probabilities feed a numeric threshold comparison downstream (the residual gate), so
        # fit the scaler on the training fold only and apply it to both folds -- never fit on
        # the test fold, to avoid leakage.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = _new_classifier()
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    return oof


def fit_net_benefit_model(dev_frame: pd.DataFrame, dev_attribution_beta1: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """Out-of-fold V(x) = P(phantom_agreement | x) - P(prior_capture | x), both fit at the field-
    default beta = 1 (the reference policy this net-benefit estimate is defined relative to), from
    joint neural+LM features, participant-grouped cross-validation.
    """
    beta1 = dev_attribution_beta1[np.isclose(dev_attribution_beta1["beta"], 1.0)]
    merged = dev_frame.merge(
        beta1[list(KEY_COLUMNS) + ["phantom_agreement", "prior_capture"]],
        on=list(KEY_COLUMNS),
        how="inner",
    )
    X = _feature_matrix(merged)
    groups = merged["participant_id"].to_numpy()

    p_rescue = _fit_probability_cv(X, merged["phantom_agreement"].astype(int).to_numpy(), groups, n_folds)
    p_capture = _fit_probability_cv(X, merged["prior_capture"].astype(int).to_numpy(), groups, n_folds)
    return pd.Series(p_rescue - p_capture, index=merged.index, name="v_hat")


def fit_net_benefit_model_frozen(
    dev_frame: pd.DataFrame, dev_attribution_beta1: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.Series:
    """The FROZEN V(x): both classifiers fit on ALL of dev, then applied to `test_frame`.

    `fit_net_benefit_model`'s out-of-fold cross-validation is exactly right for its own role --
    tuning the residual gate's thresholds on dev, where in-sample predictions would flatter the
    choice being made. Here there is nothing left to choose: the policy is already frozen, and the
    evaluation rows belong to a study the model has never seen. Dev is simply the training set, so
    a single dev-wide fit is both correct and the artifact a deployed gate would actually ship.

    No holdout label enters the fit path at any point. `test_frame` supplies features only -- it is
    never asked for phantom_agreement or prior_capture -- and the scaler is fit on dev and merely
    applied to test, never re-fit on it.

    Returns V(x) = P(phantom agreement | x) - P(prior capture | x) as a Series indexed to align
    with `test_frame`, one value per test row.
    """
    beta1 = dev_attribution_beta1[np.isclose(dev_attribution_beta1["beta"], 1.0)]
    merged = dev_frame.merge(
        beta1[list(KEY_COLUMNS) + ["phantom_agreement", "prior_capture"]],
        on=list(KEY_COLUMNS),
        how="inner",
    )
    if len(merged) != len(dev_frame):
        raise ValueError(
            f"the beta=1.0 label join dropped rows: {len(dev_frame)} dev selections -> "
            f"{len(merged)} labelled rows"
        )

    dev_features = _feature_matrix(merged)
    scaler = StandardScaler().fit(dev_features)
    X_dev = scaler.transform(dev_features)
    X_test = scaler.transform(_feature_matrix(test_frame))

    def _fit_predict(label: str) -> np.ndarray:
        model = _new_classifier()
        model.fit(X_dev, merged[label].astype(int).to_numpy())
        return model.predict_proba(X_test)[:, 1]

    return pd.Series(
        _fit_predict("phantom_agreement") - _fit_predict("prior_capture"),
        index=test_frame.index,
        name="v_hat",
    )


# The gate's three branches, as multipliers of the best fixed beta, in the order
# (neural-favoring, default, LM-favoring). This is the ONE definition: `residual_gate_beta` applies
# it on the holdout and `fusion.final_evaluation` imports it to score candidate thresholds on dev.
# A second copy would let the thresholds be tuned against a policy that is never deployed.
RESIDUAL_MULTIPLIERS = (0.5, 1.0, 2.0)


def residual_gate_beta(best_fixed_beta: float, v_hat: float, high: float, low: float) -> float:
    """Default to best_fixed_beta; move to 2x when v_hat confidently favors the language model
    (v_hat > high), to 0.5x when it confidently favors the neural decoder (v_hat < low), otherwise
    stay at the default. high/low are frozen thresholds tuned by
    `fusion.final_evaluation.run_frozen_gate_selection`.
    """
    neural_favoring, default, lm_favoring = RESIDUAL_MULTIPLIERS
    if v_hat > high:
        return best_fixed_beta * lm_favoring
    if v_hat < low:
        return best_fixed_beta * neural_favoring
    return best_fixed_beta * default
