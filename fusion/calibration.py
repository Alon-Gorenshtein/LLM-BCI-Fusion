"""Cross-fitted temperature calibration for the neural and language-model posteriors, per the
Version 2 addendum (design spec S13.2). Matches the companion attribution study's own calibration
convention (a scalar temperature per source, fit by minimizing held-out negative log-likelihood of
the true target symbol) -- this study's primary analysis (Tasks 1-10) deliberately left both
sources uncalibrated; this module tests whether calibrating them first changes the fixed-versus-
adaptive picture, before any new gate is built.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TEMPERATURE_GRID = np.geomspace(0.2, 5.0, 25)


def calibrate_posterior(p: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale an already-normalized probability vector: p_T proportional to p ** (1/T).

    Operating on probabilities rather than logits because only the final softmax output is
    materialized upstream (Methods); this is the standard equivalent of logit-scaling for a
    distribution that came from a softmax.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    scaled = p ** (1.0 / temperature)
    return scaled / scaled.sum()


def _mean_nll(frame: pd.DataFrame, column: str, temperature: float) -> float:
    total = 0.0
    for row in frame.itertuples(index=False):
        p = getattr(row, column)
        target_index = row.target_code - 1
        calibrated = calibrate_posterior(p, temperature)
        total += -np.log(max(calibrated[target_index], 1e-12))
    return total / len(frame)


def fit_temperature_cv(frame: pd.DataFrame, column: str, n_folds: int = 5) -> np.ndarray:
    """Return one out-of-fold-fitted temperature per row: for each participant-grouped fold, grid-
    search the scalar temperature minimizing mean NLL on the OTHER folds' participants, then apply
    that fold's winning temperature to this fold's rows. No participant's own rows ever inform the
    temperature applied to that participant.
    """
    groups = frame["participant_id"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")

    splitter = GroupKFold(n_splits=n_folds)
    temperatures = np.full(len(frame), np.nan)
    for train_idx, test_idx in splitter.split(frame, groups=groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a calibration CV fold")
        train_frame = frame.iloc[train_idx]
        nlls = [_mean_nll(train_frame, column, t) for t in TEMPERATURE_GRID]
        best_temperature = TEMPERATURE_GRID[int(np.argmin(nlls))]
        temperatures[test_idx] = best_temperature

    if np.isnan(temperatures).any():
        raise RuntimeError("some rows never received an out-of-fold temperature")
    return temperatures


def run_calibrated_four_arm_comparison(prior_model: str, n_folds: int = 5) -> dict:
    """Dev-only (Study N excluded) re-run of the floor/fixed/adaptive/grid-oracle comparison with
    both posteriors cross-fitted-calibrated before fusion. floor/fixed/oracle are read unchanged
    from the original (uncalibrated) attribution table -- calibration only changes what the
    adaptive arm's reliability estimator sees and what it fuses with, per design spec S13.2.
    """
    import sys

    from fusion.data import AUTHORSHIP_ROOT, build_working_frame, load_attribution
    from fusion.holdout import HOLDOUT_STUDY, split_dev_test
    from fusion.reliability import fit_reliability_cv
    from fusion.weighting import reliability_to_beta
    from fusion.arms import KEY_COLUMNS, OUTCOME_COLUMNS, select_floor_arm, select_fixed_arm, select_oracle_arm
    from fusion.net_gain import add_net_correct_gain
    from fusion.stats import participant_cluster_bootstrap_rate_diff

    if str(AUTHORSHIP_ROOT) not in sys.path:
        sys.path.insert(0, str(AUTHORSHIP_ROOT))
    from authorship.attribution import fuse, attribution_row  # noqa: E402

    working = build_working_frame(prior_model)
    dev, _test = split_dev_test(working)

    neural_temps = fit_temperature_cv(dev, column="p_neural", n_folds=n_folds)
    lm_temps = fit_temperature_cv(dev, column="p_lm", n_folds=n_folds)
    calibrated = dev.copy()
    calibrated["p_neural"] = [calibrate_posterior(p, t) for p, t in zip(dev["p_neural"], neural_temps)]
    calibrated["p_lm"] = [calibrate_posterior(p, t) for p, t in zip(dev["p_lm"], lm_temps)]

    fitted = fit_reliability_cv(calibrated, n_folds=n_folds)
    fitted = fitted.assign(beta_adaptive=reliability_to_beta(fitted["g_hat"]))

    rows = []
    for row in fitted.itertuples(index=False):
        target_index = row.target_code - 1
        fused = fuse(row.p_neural, row.p_lm, beta=row.beta_adaptive)
        rows.append(attribution_row(row.p_neural, row.p_lm, fused, target_index))
    adaptive_calibrated = pd.DataFrame(rows)
    adaptive_calibrated["arm"] = "adaptive_calibrated"
    keys = fitted[list(KEY_COLUMNS)].reset_index(drop=True)
    adaptive_calibrated = pd.concat([keys, adaptive_calibrated], axis=1)
    adaptive_calibrated = add_net_correct_gain(adaptive_calibrated)

    attribution = load_attribution(prior_model)
    attribution_dev = attribution[attribution["study"] != HOLDOUT_STUDY]
    all_arms = pd.concat(
        [
            select_floor_arm(attribution_dev)[OUTCOME_COLUMNS],
            select_fixed_arm(attribution_dev)[OUTCOME_COLUMNS],
            select_oracle_arm(attribution_dev)[OUTCOME_COLUMNS],
        ],
        ignore_index=True,
    )
    all_arms = add_net_correct_gain(all_arms)
    all_arms_with_calibrated = pd.concat(
        [all_arms, adaptive_calibrated[all_arms.columns]], ignore_index=True
    )

    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        all_arms_with_calibrated, "net_correct_gain", "adaptive_calibrated", "fixed"
    )
    return {
        "prior_model": prior_model,
        "adaptive_calibrated_vs_fixed_net_correct_gain": {"point": point, "lower": lower, "upper": upper},
        "mean_neural_temperature": float(np.mean(neural_temps)),
        "mean_lm_temperature": float(np.mean(lm_temps)),
    }
