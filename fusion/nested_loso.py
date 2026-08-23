"""Nested leave-one-study-out orchestration (design spec section 17.3). This is NOT a fresh single
holdout -- neither Study N nor Study B can honestly be called untouched for this architecture, since
both already had a frozen policy evaluated against them and those results shaped this redesign. Every
selection instead gets exactly one out-of-study prediction, from the one outer fold where its own
study was held out; only the pooled four-fold predictions are compared to nested-CV best-fixed
fusion. Disclosed throughout as a rigorous exploratory cross-study evaluation, never as independent
or prospective validation.
"""
from __future__ import annotations

import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.candidate_arbitration import (
    apply_combining_rule,
    fit_candidate_ranker,
    fit_candidate_ranker_frozen,
    select_delta,
)
from fusion.candidate_features import candidate_feature_table
from fusion.route_a_policy import route_a_dev_wide_best_fixed_beta

_KEY = list(KEY_COLUMNS)
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")


def _load_frame_and_attribution(prior_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from fusion.data import build_working_frame, load_attribution

    return build_working_frame(prior_model), load_attribution(prior_model)


def run_nested_loso(prior_model: str, n_inner_folds: int = 5) -> pd.DataFrame:
    frame, attribution = _load_frame_and_attribution(prior_model)

    outer_results = []
    for held_out_study in _STUDIES:
        train_frame = frame[frame["study"] != held_out_study].reset_index(drop=True)
        train_attribution = attribution[attribution["study"] != held_out_study].reset_index(drop=True)
        test_frame = frame[frame["study"] == held_out_study].reset_index(drop=True)
        test_attribution = attribution[attribution["study"] == held_out_study].reset_index(drop=True)

        fixed_beta = route_a_dev_wide_best_fixed_beta(train_attribution)

        train_features = candidate_feature_table(train_frame, train_attribution, fixed_beta)
        test_features = candidate_feature_table(test_frame, test_attribution, fixed_beta)

        p_hat_train_oof = fit_candidate_ranker(train_features, n_folds=n_inner_folds)
        delta = select_delta(train_features, p_hat_train_oof)

        p_hat_test = fit_candidate_ranker_frozen(train_features, test_features)
        applied = apply_combining_rule(test_features, p_hat_test, delta)
        applied["outer_held_out_study"] = held_out_study
        applied["fixed_beta"] = fixed_beta
        applied["delta"] = delta
        outer_results.append(applied)

    result = pd.concat(outer_results, ignore_index=True)
    if len(result) != len(frame):
        raise ValueError(
            f"nested LOSO produced {len(result)} rows for {len(frame)} selections -- "
            f"every selection must appear exactly once, from its own study's outer fold"
        )
    return result
