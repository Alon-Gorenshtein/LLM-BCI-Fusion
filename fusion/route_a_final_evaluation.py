"""Route A step 4: freeze the Route A controller using only dev
(StudyF + StudyL + StudyN), then evaluate it exactly once against the best-fixed-beta control on
the StudyB holdout. `evaluate_route_a_on_holdout` must be called at most once per prior_model in any
real run of this study -- repeated calls during development would defeat the point of the holdout,
exactly as `fusion.final_evaluation.evaluate_on_holdout` already documents for Study N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.actionability import actionability_by_key, beta_correctness_long
from fusion.arms import KEY_COLUMNS
from fusion.data import build_working_frame, load_attribution
from fusion.net_gain import add_net_correct_gain
from fusion.route_a_holdout import HOLDOUT_STUDY_ROUTE_A, split_dev_test_route_a
from fusion.route_a_policy import combine_route_a_policy, run_route_a_policy_selection
from fusion.route_a_stage1 import fit_actionability_model_frozen
from fusion.route_a_stage2 import fit_beta_utility_models_frozen
from fusion.stats import participant_cluster_bootstrap_rate_diff

_KEY = list(KEY_COLUMNS)


def _require_dev_only(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if "study" not in frame.columns:
            raise ValueError(
                "dev inputs must carry a 'study' column so Route A holdout membership can be verified"
            )
        if HOLDOUT_STUDY_ROUTE_A in set(frame["study"].unique()):
            raise ValueError(
                f"{HOLDOUT_STUDY_ROUTE_A!r} must not be present in Route A's dev-side inputs"
            )


def evaluate_route_a_frozen_policy(
    dev_frame: pd.DataFrame,
    dev_attribution: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_attribution: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """Freeze the Route A controller on `dev_*`, then score it once against the best-fixed-beta
    control on `test_*`. Split out from `evaluate_route_a_on_holdout` so the whole path can be
    exercised on a pseudo-holdout carved out of dev, without spending the real StudyB read.
    """
    _require_dev_only(dev_frame, dev_attribution)

    dev_actionability = actionability_by_key(dev_attribution)
    dev_beta_correctness = beta_correctness_long(dev_attribution)
    policy = run_route_a_policy_selection(
        dev_frame, dev_attribution, dev_actionability, dev_beta_correctness, n_folds=n_folds
    )

    p_actionable_test = fit_actionability_model_frozen(dev_frame, dev_actionability, test_frame)
    q_hat_test = fit_beta_utility_models_frozen(dev_frame, dev_beta_correctness, test_frame)
    applied = combine_route_a_policy(
        test_frame, p_actionable_test, q_hat_test, policy["best_fixed_beta"], policy["tau"]
    )

    gated = applied.merge(
        test_attribution[_KEY + ["beta", "fused_correct", "neural_correct"]],
        left_on=_KEY + ["beta_hat"], right_on=_KEY + ["beta"], how="inner",
    )
    if len(gated) != len(test_frame):
        raise ValueError(
            f"Route A holdout scoring join dropped rows: {len(test_frame)} holdout selections -> "
            f"{len(gated)} scored rows"
        )
    gated["arm"] = "route_a"
    gated = add_net_correct_gain(gated)

    fixed_rows = test_attribution[
        np.isclose(test_attribution["beta"], policy["best_fixed_beta"])
    ].copy()
    if len(fixed_rows) != len(test_frame):
        raise ValueError(
            f"best-fixed arm has {len(fixed_rows)} rows for {len(test_frame)} holdout selections"
        )
    fixed_rows["arm"] = "best_fixed"
    fixed_rows = add_net_correct_gain(fixed_rows)

    long_frame = pd.concat(
        [
            gated[["participant_id", "arm", "net_correct_gain"]],
            fixed_rows[["participant_id", "arm", "net_correct_gain"]],
        ],
        ignore_index=True,
    )
    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        long_frame, "net_correct_gain", "route_a", "best_fixed"
    )

    return {
        "policy": policy,
        "route_a_net_correct_gain": float(gated["net_correct_gain"].mean()),
        "best_fixed_net_correct_gain": float(fixed_rows["net_correct_gain"].mean()),
        "route_a_minus_best_fixed": {"point": point, "lower": lower, "upper": upper},
        "route_a_fused_correct": float(gated["fused_correct"].astype(int).mean()),
        "best_fixed_fused_correct": float(fixed_rows["fused_correct"].astype(int).mean()),
        "n_holdout_selections": int(len(test_frame)),
        "n_holdout_participants": int(test_frame["participant_id"].nunique()),
    }


def evaluate_route_a_on_holdout(prior_model: str, n_folds: int = 5) -> dict:
    """The single, one-shot evaluation Route A has been built toward: apply the frozen controller
    (selected using dev alone -- StudyF + StudyL + StudyN) to StudyB, and compare its net correct
    gain against the dev-selected best fixed beta, both applied to StudyB for the first and only
    time. Call this AT MOST ONCE per prior_model in any real run of this study.
    """
    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)
    dev_frame, test_frame = split_dev_test_route_a(working)
    dev_attribution, test_attribution = split_dev_test_route_a(attribution)

    result = evaluate_route_a_frozen_policy(
        dev_frame, dev_attribution, test_frame, test_attribution, n_folds=n_folds
    )
    return dict(prior_model=prior_model, **result)
