"""Round 3 addendum, design spec S14.2 item 4: characterize the two component classifiers inside
the arbitration gate (fusion.arbitration.fit_net_benefit_model), which only ever surface as their
difference V(x) elsewhere. Dev-only (Studies B, F, L) -- this module never reads Study N and never
calls fusion.final_evaluation.evaluate_on_holdout or evaluate_frozen_policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from fusion.joint_features import joint_features_frame
from fusion.arms import KEY_COLUMNS
from fusion.arbitration import RESIDUAL_MULTIPLIERS, residual_gate_beta
from fusion.reliability import RANDOM_STATE


def _fit_probability_cv_with_metrics(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int) -> dict:
    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")
    splitter = GroupKFold(n_splits=n_folds)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a diagnostics CV fold")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    return {
        "oof_probabilities": oof,
        "auroc": float(roc_auc_score(y, oof)) if 0 < y.sum() < len(y) else float("nan"),
        "auprc": float(average_precision_score(y, oof)) if 0 < y.sum() < len(y) else float("nan"),
        "n_events": int(y.sum()),
    }


def characterize_gate_components(
    dev_frame: pd.DataFrame,
    dev_attribution_beta1: pd.DataFrame,
    high: float,
    low: float,
    best_fixed_beta: float,
    n_folds: int = 5,
) -> dict:
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

    features = joint_features_frame(merged["p_neural"], merged["p_lm"])
    X = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    groups = merged["participant_id"].to_numpy()

    rescue_result = _fit_probability_cv_with_metrics(
        X, merged["phantom_agreement"].astype(int).to_numpy(), groups, n_folds
    )
    capture_result = _fit_probability_cv_with_metrics(
        X, merged["prior_capture"].astype(int).to_numpy(), groups, n_folds
    )
    v_hat = rescue_result["oof_probabilities"] - capture_result["oof_probabilities"]

    gated_betas = np.array(
        [residual_gate_beta(best_fixed_beta, v, high, low) for v in v_hat]
    )
    deviates = ~np.isclose(gated_betas, best_fixed_beta)
    neural_favoring, _default, lm_favoring = RESIDUAL_MULTIPLIERS
    favors_lm = np.isclose(gated_betas, best_fixed_beta * lm_favoring)
    favors_neural = np.isclose(gated_betas, best_fixed_beta * neural_favoring)

    quartiles = np.percentile(v_hat, [0, 25, 50, 75, 100]).tolist()

    gated_beta_by_selection = merged[list(KEY_COLUMNS)].copy()
    gated_beta_by_selection["gated_beta"] = gated_betas

    return {
        "gated_beta_by_selection": gated_beta_by_selection,
        "phantom_agreement_auroc": rescue_result["auroc"],
        "phantom_agreement_auprc": rescue_result["auprc"],
        "phantom_agreement_n_events": rescue_result["n_events"],
        "prior_capture_auroc": capture_result["auroc"],
        "prior_capture_auprc": capture_result["auprc"],
        "prior_capture_n_events": capture_result["n_events"],
        "v_hat_quartiles": quartiles,
        "fraction_deviating_from_best_fixed": float(deviates.mean()),
        "fraction_favoring_lm": float(favors_lm.mean()),
        "fraction_favoring_neural": float(favors_neural.mean()),
        "n_selections": int(len(merged)),
    }


def _fused_correct_at(dev_attribution: pd.DataFrame, keys: pd.DataFrame, beta: float) -> np.ndarray:
    rows = dev_attribution[np.isclose(dev_attribution["beta"], beta)]
    if rows.empty:
        raise ValueError(f"beta={beta} is not materialized in the attribution table")
    # one_to_one, because the returned array is positionally aligned to `keys`: an inner merge
    # preserves left-key order only when each key matches exactly one right row.
    merged = keys[list(KEY_COLUMNS)].merge(
        rows[list(KEY_COLUMNS) + ["fused_correct"]],
        on=list(KEY_COLUMNS), how="inner", validate="one_to_one",
    )
    if len(merged) != len(keys):
        raise ValueError(
            f"the beta={beta} fused_correct join dropped rows: {len(keys)} selections -> "
            f"{len(merged)} matched rows"
        )
    return merged["fused_correct"].astype(bool).to_numpy()


def classify_gate_deviation_outcomes(
    gated_beta_by_selection: pd.DataFrame,
    dev_attribution: pd.DataFrame,
    best_fixed_beta: float,
) -> dict:
    """Design spec S14.2 item 4's dev-side helped/hurt/no-effect split: for the dev selections where
    the frozen gate moves off the best fixed beta, did that move change the fused selection from
    incorrect to correct (helped), from correct to incorrect (hurt), or neither (no effect)?

    No fusion is recomputed. Every gate beta lands on BETA_GRID for both priors' frozen policies
    (`gate_betas_on_grid` in output/v2_holdout_evaluation.txt), so both the gated and the best-fixed
    outcome for each deviating selection are already materialized rows of the attribution table and
    are read off it. Dev-only: `dev_attribution` must be the dev half of the split.
    """
    gated_betas = gated_beta_by_selection["gated_beta"].to_numpy()
    deviating = gated_beta_by_selection[~np.isclose(gated_betas, best_fixed_beta)]

    baseline_correct = _fused_correct_at(dev_attribution, deviating, best_fixed_beta)
    gated_correct = np.empty(len(deviating), dtype=bool)
    deviating_betas = deviating["gated_beta"].to_numpy()
    for beta in np.unique(deviating_betas):
        at_beta = np.isclose(deviating_betas, beta)
        gated_correct[at_beta] = _fused_correct_at(dev_attribution, deviating[at_beta], beta)

    return {
        "n_deviating": int(len(deviating)),
        "helped": int((~baseline_correct & gated_correct).sum()),
        "hurt": int((baseline_correct & ~gated_correct).sum()),
        "no_effect": int((baseline_correct == gated_correct).sum()),
    }
