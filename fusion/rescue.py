"""Rescue experiment (post-final-review): does adaptive fusion beat the BEST achievable fixed
beta -- not just the field-default beta=1 -- once both the fixed-beta choice and the adaptive
link's own anchor are selected out-of-fold instead of assumed a priori?

The original four-arm comparison (fusion.arms) found the adaptive arm losing to fixed beta=1 on
real data. That contrast confounds two questions: "does adaptive beat beta=1" (yes/no is
uninformative on its own, since beta=1 was never claimed to be optimal) and "does adaptive beat
the best any fixed beta can do" (the question that actually tests the thesis). This module builds
the second, fairer contrast, with the same participant-grouped leakage discipline as
fusion.reliability and fusion.transport.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fusion.arms import compute_adaptive_arm
from fusion.reliability import RANDOM_STATE
from fusion.stats import N_BOOTSTRAP

BETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
# Derived from BETA_GRID (rather than listed separately) so the two grids cannot silently drift
# apart: every fixed-beta control value has a matching degenerate anchor (beta_min == beta_max ==
# that beta) that reproduces it exactly, so the adaptive arm's search space always fully nests the
# fixed-beta control's search space -- a fair comparison, not one where the control can win with a
# beta the adaptive arm structurally cannot reach.
ANCHOR_GRID = tuple((b, b) for b in BETA_GRID) + (
    (0.9, 1.1),
    (0.8, 1.25),
    (0.5, 2.0),
    (0.25, 4.0),
    (0.1, 4.0),
    (0.1, 20.0),
    (0.5, 8.0),
)

_MINIMIZE_OUTCOMES = frozenset({"prior_capture"})
_MAXIMIZE_OUTCOMES = frozenset({"fused_correct", "phantom_agreement", "net_correct_gain"})


def _is_better(candidate_rate: float, current_best_rate: float | None, outcome: str) -> bool:
    """True if candidate_rate is a better choice than current_best_rate for this outcome."""
    if outcome not in _MINIMIZE_OUTCOMES and outcome not in _MAXIMIZE_OUTCOMES:
        raise ValueError(
            f"unrecognized outcome {outcome!r}: must be one of "
            f"{_MINIMIZE_OUTCOMES | _MAXIMIZE_OUTCOMES}"
        )
    if current_best_rate is None:
        return True
    if outcome in _MINIMIZE_OUTCOMES:
        return candidate_rate < current_best_rate
    return candidate_rate > current_best_rate


def _grouped_folds(frame: pd.DataFrame, n_folds: int):
    groups = frame["participant_id"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique participants")
    splitter = GroupKFold(n_splits=n_folds)
    dummy_x = np.zeros((len(frame), 1))
    for train_idx, test_idx in splitter.split(dummy_x, groups=groups):
        train_participants = set(groups[train_idx])
        test_participants = set(groups[test_idx])
        if train_participants & test_participants:
            raise RuntimeError("group leakage across a rescue CV fold")
        yield train_participants, test_participants


def select_best_fixed_beta_arm_cv(
    attribution_frame: pd.DataFrame, n_folds: int = 5, outcome: str = "prior_capture"
) -> pd.DataFrame:
    """A fixed-beta control where the single beta is itself chosen out-of-fold: for each
    participant-grouped fold, pick whichever grid beta is best for `outcome` (minimizing the harm
    rate prior_capture, maximizing accuracy-style/benefit rates such as
    fused_correct/phantom_agreement/net_correct_gain -- see `_MINIMIZE_OUTCOMES`/
    `_MAXIMIZE_OUTCOMES`) on the OTHER folds' participants, then apply that beta's precomputed
    grid row to this fold's participants.
    """
    parts = []
    for train_participants, test_participants in _grouped_folds(attribution_frame, n_folds):
        train_frame = attribution_frame[attribution_frame["participant_id"].isin(train_participants)]
        rates = {
            beta: train_frame.loc[np.isclose(train_frame["beta"], beta), outcome].mean()
            for beta in BETA_GRID
        }
        best_beta = None
        best_rate = None
        for beta, rate in rates.items():
            if _is_better(rate, best_rate, outcome):
                best_rate = rate
                best_beta = beta
        test_frame = attribution_frame[attribution_frame["participant_id"].isin(test_participants)]
        chosen = test_frame[np.isclose(test_frame["beta"], best_beta)].copy()
        chosen["arm"] = "best_fixed"
        chosen["chosen_beta"] = best_beta
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def tune_adaptive_anchor_cv(
    working_frame: pd.DataFrame,
    n_folds: int = 5,
    outcome: str = "prior_capture",
    anchor_grid: tuple = ANCHOR_GRID,
) -> pd.DataFrame:
    """Like the adaptive arm, but BOTH the reliability estimator and the (BETA_MIN, BETA_MAX)
    anchor are fit out-of-fold per participant-grouped fold. For fold k: the reliability estimator
    is fit fresh on that fold's training participants only (closing the second-order leakage path
    from a globally-shared g_hat), producing predictions for both training rows (for anchor
    selection) and test rows (for final evaluation) of fold k; the anchor is then selected using
    that same fold-local model's in-sample predictions on its own training rows -- not a further
    nested inner-CV split -- and applied to the test rows. This has been measured to differ
    negligibly from a true inner-CV anchor-selection estimate on this data (correlation 0.998), but
    it is not a formally nested selection: describe it as fold-local in-sample anchor selection,
    not "true nested cross-validation". No test participant's own data ever informs its own fold's
    estimator fit or its own fold's anchor choice -- the leakage this function closes is strictly
    the cross-fold (participant-level) kind, not the in-sample-vs-inner-CV kind.

    `working_frame` must have `p_neural`, `p_lm`, `target_code`, `neural_correct`,
    `participant_id` (i.e. the output of `fusion.data.build_working_frame`, NOT a
    `fit_reliability_cv`-fitted frame -- this function fits its own estimator internally).
    """
    from sklearn.linear_model import LogisticRegression
    from fusion.features import reliability_features_frame

    parts = []
    for train_participants, test_participants in _grouped_folds(working_frame, n_folds):
        train_frame = working_frame[working_frame["participant_id"].isin(train_participants)].copy()
        test_frame = working_frame[working_frame["participant_id"].isin(test_participants)].copy()

        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(
            reliability_features_frame(train_frame["p_neural"]),
            train_frame["neural_correct"].astype(int),
        )
        train_frame["g_hat"] = model.predict_proba(
            reliability_features_frame(train_frame["p_neural"])
        )[:, 1]
        test_frame["g_hat"] = model.predict_proba(
            reliability_features_frame(test_frame["p_neural"])
        )[:, 1]

        best_rate = None
        best_anchor = anchor_grid[0]
        for beta_min, beta_max in anchor_grid:
            g = np.clip(train_frame["g_hat"], 1e-6, 1 - 1e-6)
            beta = beta_min * (beta_max / beta_min) ** g
            candidate = train_frame.assign(beta_adaptive=beta)
            candidate_arm = compute_adaptive_arm(candidate)
            rate = candidate_arm[outcome].mean()
            if _is_better(rate, best_rate, outcome):
                best_rate = rate
                best_anchor = (beta_min, beta_max)

        beta_min, beta_max = best_anchor
        g = np.clip(test_frame["g_hat"], 1e-6, 1 - 1e-6)
        test_frame["beta_adaptive"] = beta_min * (beta_max / beta_min) ** g
        test_arm = compute_adaptive_arm(test_frame)
        test_arm["arm"] = "tuned_adaptive"
        test_arm["chosen_beta_min"] = beta_min
        test_arm["chosen_beta_max"] = beta_max
        parts.append(test_arm)
    return pd.concat(parts, ignore_index=True)


def run_rescue_experiment(prior_model: str, outcome: str = "prior_capture", n_folds: int = 5) -> dict:
    """Does tuned adaptive fusion beat the best achievable fixed beta, once neither side gets to
    see its own evaluation fold?
    """
    from fusion.data import build_working_frame, load_attribution
    from fusion.stats import participant_cluster_bootstrap_rate_diff

    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)

    best_fixed = select_best_fixed_beta_arm_cv(attribution, n_folds=n_folds, outcome=outcome)
    tuned_adaptive = tune_adaptive_anchor_cv(working, n_folds=n_folds, outcome=outcome)

    assert len(best_fixed) == len(tuned_adaptive), (
        f"best_fixed has {len(best_fixed)} rows but tuned_adaptive has {len(tuned_adaptive)} rows"
    )

    long_frame = pd.concat(
        [
            best_fixed[["participant_id", "arm", outcome]],
            tuned_adaptive[["participant_id", "arm", outcome]],
        ],
        ignore_index=True,
    )
    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        long_frame, outcome, "best_fixed", "tuned_adaptive", n_bootstrap=N_BOOTSTRAP
    )
    return {
        "prior_model": prior_model,
        "outcome": outcome,
        "best_fixed_rate": float(best_fixed[outcome].mean()),
        "tuned_adaptive_rate": float(tuned_adaptive[outcome].mean()),
        "rate_diff_point": point,
        "rate_diff_lower": lower,
        "rate_diff_upper": upper,
        "chosen_fixed_betas": sorted(best_fixed["chosen_beta"].unique().tolist()),
        "chosen_anchors": sorted(
            set(zip(tuned_adaptive["chosen_beta_min"], tuned_adaptive["chosen_beta_max"]))
        ),
    }
