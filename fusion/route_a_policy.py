"""Route A's combining rule and dev-side policy selection:

    beta_hat_i = best_fixed_beta                          if P_hat(actionable_i) < tau
               = argmax_{beta in BETA_GRID} Q_hat_i(beta)  otherwise

best_fixed_beta is Route A's OWN dev-wide best fixed beta, recomputed on Route A's dev set
(StudyF + StudyL + StudyN), never reused from the joint gate's dev computation (Studies
B, F, L) since Route A's dev composition differs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID

_KEY = list(KEY_COLUMNS)
TAU_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)


def route_a_dev_wide_best_fixed_beta(dev_attribution: pd.DataFrame) -> float:
    """The beta in BETA_GRID maximizing mean fused_correct on the given dev attribution table."""
    rates = {
        beta: dev_attribution.loc[np.isclose(dev_attribution["beta"], beta), "fused_correct"].mean()
        for beta in BETA_GRID
    }
    return float(max(rates, key=rates.get))


def combine_route_a_policy(
    dev_frame: pd.DataFrame,
    p_actionable: pd.Series,
    q_hat_long: pd.DataFrame,
    best_fixed_beta: float,
    tau: float,
) -> pd.DataFrame:
    """KEY_COLUMNS + ['beta_hat']. `p_actionable` must be indexed positionally to `dev_frame`
    (same convention as `fusion.route_a_stage1.fit_actionability_model`'s return value merged back
    onto `dev_frame` by position, since both were derived from the same row order).
    """
    keys = dev_frame[_KEY].reset_index(drop=True)
    p_actionable = pd.Series(p_actionable).reset_index(drop=True)

    # argmax q_hat per selection, ties broken by row order (first occurrence of the max), matching
    # the row-order tie-break convention already used by fusion.arms.select_oracle_arm.
    idx_of_max = q_hat_long.groupby(_KEY, sort=False)["q_hat"].idxmax()
    argmax_beta = q_hat_long.loc[idx_of_max, _KEY + ["beta"]].rename(columns={"beta": "argmax_beta"})
    aligned = keys.merge(argmax_beta, on=_KEY, how="left", validate="one_to_one")
    if aligned["argmax_beta"].isna().any():
        raise ValueError("some dev selections have no q_hat rows in q_hat_long")

    is_actionable = p_actionable.to_numpy() >= tau
    beta_hat = np.where(is_actionable, aligned["argmax_beta"].to_numpy(), best_fixed_beta)
    out = keys.copy()
    out["beta_hat"] = beta_hat
    return out


def run_route_a_policy_selection(
    dev_frame: pd.DataFrame,
    dev_attribution: pd.DataFrame,
    dev_actionability: pd.DataFrame,
    dev_beta_correctness: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """Grid-search tau over TAU_GRID using out-of-fold p_actionable and q_hat, maximizing dev net
    correct gain, and return the selected policy plus its dev-side performance.
    """
    from fusion.route_a_stage1 import fit_actionability_model
    from fusion.route_a_stage2 import fit_beta_utility_models

    best_fixed_beta = route_a_dev_wide_best_fixed_beta(dev_attribution)
    best_fixed_rate = dev_attribution.loc[
        np.isclose(dev_attribution["beta"], best_fixed_beta), "fused_correct"
    ].mean()

    p_actionable = fit_actionability_model(dev_frame, dev_actionability, n_folds=n_folds)
    q_hat_long = fit_beta_utility_models(dev_frame, dev_beta_correctness, n_folds=n_folds)

    best_score = -np.inf
    best_tau = TAU_GRID[0]
    for tau in TAU_GRID:
        policy = combine_route_a_policy(dev_frame, p_actionable, q_hat_long, best_fixed_beta, tau)
        # Score each candidate policy against the already-materialized attribution table at the
        # betas it actually picked, exactly as fusion.final_evaluation.run_frozen_gate_selection
        # scores candidate residual-gate thresholds -- the grid is exact, so no re-fusion needed.
        scored = policy.merge(
            dev_attribution[_KEY + ["beta", "fused_correct"]],
            left_on=_KEY + ["beta_hat"], right_on=_KEY + ["beta"], how="inner",
        )
        if len(scored) != len(dev_frame):
            raise ValueError(
                f"tau={tau} scoring join dropped rows: {len(dev_frame)} selections -> "
                f"{len(scored)} scored rows"
            )
        score = float(scored["fused_correct"].astype(int).mean())
        if score > best_score:
            best_score = score
            best_tau = tau

    return {
        "tau": float(best_tau),
        "best_fixed_beta": float(best_fixed_beta),
        "dev_gated_fused_correct": float(best_score),
        "dev_best_fixed_fused_correct": float(best_fixed_rate),
        "dev_gated_minus_best_fixed": float(best_score - best_fixed_rate),
        "n_dev_selections": int(len(dev_frame)),
    }
