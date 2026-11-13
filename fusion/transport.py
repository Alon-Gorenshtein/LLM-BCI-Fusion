"""Aim 3: does the adaptive arm's benefit over fixed-weight fusion transport across
cohorts, or is it a pooled-calibration artifact -- directly testing the same question the
companion calibration-transportability study raised for decoder discriminability.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from fusion.features import reliability_features_frame
from fusion.weighting import reliability_to_beta
from fusion.arms import compute_adaptive_arm, select_fixed_arm
from fusion.stats import participant_cluster_bootstrap_rate_diff, N_BOOTSTRAP

RANDOM_STATE = 20260820


def leave_one_study_out_contrast(
    working_frame: pd.DataFrame, attribution_frame: pd.DataFrame, outcome: str = "prior_capture"
) -> pd.DataFrame:
    """Fit the reliability mapping on 3 studies, evaluate the fixed-vs-adaptive contrast
    on the 4th; repeat for every held-out study.
    """
    studies = sorted(working_frame["study"].unique())
    results = []
    for held_out in studies:
        train = working_frame[working_frame["study"] != held_out]
        test = working_frame[working_frame["study"] == held_out].copy()

        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(reliability_features_frame(train["p_neural"]), train["neural_correct"].astype(int))
        test["g_hat"] = model.predict_proba(reliability_features_frame(test["p_neural"]))[:, 1]
        test["beta_adaptive"] = reliability_to_beta(test["g_hat"])

        adaptive = compute_adaptive_arm(test)
        fixed = select_fixed_arm(attribution_frame[attribution_frame["study"] == held_out])
        assert len(adaptive) == len(fixed), (
            f"adaptive arm has {len(adaptive)} rows but fixed arm has {len(fixed)} rows -- "
            f"they must cover the same selections before combining for a contrast"
        )
        long_frame = pd.concat(
            [adaptive[["participant_id", "arm", outcome]], fixed[["participant_id", "arm", outcome]]],
            ignore_index=True,
        )
        point, lower, upper = participant_cluster_bootstrap_rate_diff(
            long_frame, outcome, "fixed", "adaptive", n_bootstrap=N_BOOTSTRAP
        )
        results.append({"held_out_study": held_out, "point": point, "lower": lower, "upper": upper})
    return pd.DataFrame(results)
