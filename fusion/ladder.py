"""Aim 4: repeat the primary fixed-vs-adaptive contrast across the sibling ladder's full
set of priors (classical n-gram through frontier-scale LLMs, 10 families) to check
whether the adaptive arm's benefit is ladder-wide or concentrated in a few families.
"""
from __future__ import annotations

import pandas as pd

from fusion.data import build_working_frame, load_attribution
from fusion.reliability import fit_reliability_cv
from fusion.weighting import reliability_to_beta
from fusion.arms import compute_adaptive_arm, select_fixed_arm
from fusion.stats import participant_cluster_bootstrap_rate_diff, N_BOOTSTRAP

N_FOLDS = 5


def run_one_prior(prior_model: str, outcome: str = "prior_capture") -> dict:
    working = build_working_frame(prior_model)
    fitted = fit_reliability_cv(working, n_folds=N_FOLDS)
    fitted = fitted.assign(beta_adaptive=reliability_to_beta(fitted["g_hat"]))
    adaptive = compute_adaptive_arm(fitted)
    fixed = select_fixed_arm(load_attribution(prior_model, beta=1.0))
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
    return {"prior_model": prior_model, "point": point, "lower": lower, "upper": upper}


def run_ladder_sweep(prior_models: list[str], outcome: str = "prior_capture") -> pd.DataFrame:
    return pd.DataFrame(run_one_prior(model, outcome) for model in prior_models)
