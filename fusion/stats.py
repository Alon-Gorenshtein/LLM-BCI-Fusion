"""Primary statistical contrast: does the adaptive arm reduce prior-capture / fused error
relative to the fixed-weight and floor arms, and how much of the oracle's ceiling does it
recover? Uses participant-clustered GEE rather than a full mixed-effects GLMM for the
primary contrast, matching the two sibling papers' own precedent of falling back to a
cluster-robust estimator when a true binary mixed-effects model is fragile to fit --
a convergence flag is not evidence of convergence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

RANDOM_STATE = 20260820
N_BOOTSTRAP = 2000


def cluster_robust_contrast(long_frame: pd.DataFrame, outcome: str, reference_arm: str = "fixed"):
    """GEE logistic regression of `outcome` on arm (categorical), clustered on participant_id."""
    frame = long_frame.copy()
    event_rates = frame.groupby("arm")[outcome].mean()
    degenerate_arms = event_rates[(event_rates == 0) | (event_rates == 1)].index.tolist()
    if degenerate_arms:
        raise ValueError(
            f"cluster_robust_contrast: arm(s) {degenerate_arms} have a 0% or 100% event rate for "
            f"outcome {outcome!r}, which produces a degenerate/fabricated GEE coefficient. "
            f"Exclude these arms from the input frame before calling this function -- for example, "
            f"the floor arm has no LM and is structurally 0% for prior_capture/phantom_agreement."
        )
    frame[outcome] = frame[outcome].astype(int)
    other_arms = [a for a in frame["arm"].unique() if a != reference_arm]
    frame["arm"] = pd.Categorical(frame["arm"], categories=[reference_arm] + other_arms)
    model = smf.gee(
        f"{outcome} ~ C(arm, Treatment(reference='{reference_arm}'))",
        groups="participant_id",
        data=frame,
        family=sm.families.Binomial(),
    )
    result = model.fit()
    return result.summary2().tables[1]


def participant_cluster_bootstrap_rate_diff(
    long_frame: pd.DataFrame, outcome: str, arm_a: str, arm_b: str, n_bootstrap: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for rate(arm_a) - rate(arm_b), resampling participants with replacement."""
    rng = np.random.default_rng(RANDOM_STATE)
    participants = long_frame["participant_id"].unique()
    point = (
        long_frame.loc[long_frame["arm"] == arm_a, outcome].mean()
        - long_frame.loc[long_frame["arm"] == arm_b, outcome].mean()
    )
    by_participant = {p: sub for p, sub in long_frame.groupby("participant_id")}
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(participants, size=len(participants), replace=True)
        resampled = pd.concat([by_participant[p] for p in sample])
        rate_a = resampled.loc[resampled["arm"] == arm_a, outcome].mean()
        rate_b = resampled.loc[resampled["arm"] == arm_b, outcome].mean()
        diffs[i] = rate_a - rate_b
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lower), float(upper)


def oracle_gap_closure(long_frame: pd.DataFrame, outcome: str) -> float:
    """Fraction of the fixed-to-oracle gap the adaptive arm closes for `outcome`.

    Only valid for outcome="fused_correct": the oracle arm (see `fusion.arms.select_oracle_arm`)
    is selected by maximizing fused_correct per selection, so its ceiling is only meaningful as an
    accuracy bound. Calling this with a different outcome (e.g. prior_capture) would compare a
    numerator built on one objective against a denominator built on another.
    """
    if outcome != "fused_correct":
        raise ValueError(
            f"oracle_gap_closure is only meaningful for outcome='fused_correct' (the oracle arm's "
            f"own selection objective), got {outcome!r}. The oracle arm is not optimized for other "
            f"outcomes, so a gap-closure fraction for them would compare mismatched objectives."
        )
    rates = long_frame.groupby("arm")[outcome].mean()
    fixed, adaptive, oracle = rates["fixed"], rates["adaptive"], rates["oracle"]
    gap = fixed - oracle
    if abs(gap) < 1e-12:
        return float("nan")
    return float((fixed - adaptive) / gap)


def benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    _, adjusted, _, _ = multipletests(pvalues, method="fdr_bh")
    return adjusted
