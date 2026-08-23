"""Net correct gain: the redesigned primary endpoint (design spec 3.1), replacing prior-capture
rate after the rescue experiment (Task 9) confirmed it is structurally minimized by the no-LM
floor arm and so cannot fairly credit any fusion policy.

net_correct_gain = int(fused_correct) - int(neural_correct), per selection. Its mean equals
P(phantom_agreement) - P(prior_capture) -- a single honest measure of the LM's net accuracy
contribution. The no-LM floor arm scores exactly 0 by construction (fused_correct == neural_correct
when no LM is used), making it a genuine zero anchor rather than an endpoint-dominating baseline.
"""
from __future__ import annotations

import pandas as pd

from fusion.stats import N_BOOTSTRAP


def add_net_correct_gain(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of `frame` with a `net_correct_gain` column added; does not mutate `frame`.

    frame must have boolean (or 0/1) `fused_correct` and `neural_correct` columns.
    """
    result = frame.copy()
    result["net_correct_gain"] = (
        result["fused_correct"].astype(int) - result["neural_correct"].astype(int)
    )
    return result


def run_net_gain_comparison(prior_model: str, n_folds: int = 5) -> dict:
    """Decisive real-data comparison on the redesigned primary endpoint: does the tuned-adaptive
    arm beat the best out-of-fold fixed beta on net correct gain? The no-LM floor arm's net
    correct gain is exactly 0.0 by construction -- no computation needed to report it.
    """
    from fusion.data import build_working_frame, load_attribution
    from fusion.rescue import select_best_fixed_beta_arm_cv, tune_adaptive_anchor_cv
    from fusion.stats import participant_cluster_bootstrap_rate_diff

    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)

    best_fixed = add_net_correct_gain(
        select_best_fixed_beta_arm_cv(attribution, n_folds=n_folds, outcome="fused_correct")
    )
    tuned_adaptive = add_net_correct_gain(
        tune_adaptive_anchor_cv(working, n_folds=n_folds, outcome="fused_correct")
    )

    assert len(best_fixed) == len(tuned_adaptive), (
        f"best_fixed has {len(best_fixed)} rows but tuned_adaptive has {len(tuned_adaptive)} rows"
    )

    long_frame = pd.concat(
        [
            best_fixed[["participant_id", "arm", "net_correct_gain"]],
            tuned_adaptive[["participant_id", "arm", "net_correct_gain"]],
        ],
        ignore_index=True,
    )
    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        long_frame, "net_correct_gain", "best_fixed", "tuned_adaptive", n_bootstrap=N_BOOTSTRAP
    )
    return {
        "prior_model": prior_model,
        "floor_net_gain": 0.0,
        "best_fixed_net_gain": float(best_fixed["net_correct_gain"].mean()),
        "tuned_adaptive_net_gain": float(tuned_adaptive["net_correct_gain"].mean()),
        "net_gain_diff_point": point,
        "net_gain_diff_lower": lower,
        "net_gain_diff_upper": upper,
    }
