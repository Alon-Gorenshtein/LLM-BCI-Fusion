"""Assembles, per selection and per outer leave-one-study-out fold, every trivial confidence
baseline the candidate-arbitration ranker's own p_hat_fixed_winner must beat to justify its added
complexity. Reuses the exact same outer-fold split as fusion.nested_loso.run_nested_loso so every
signal is compared on identical train/test partitions and identical fixed_beta per fold.
"""
from __future__ import annotations

import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.candidate_arbitration import (
    apply_combining_rule, fit_candidate_ranker, fit_candidate_ranker_frozen, select_delta,
)
from fusion.candidate_features import candidate_feature_table
from fusion.fused_posterior_features import fused_posterior_features_frame
from fusion.reliability import fit_reliability_frozen
from fusion.route_a_policy import route_a_dev_wide_best_fixed_beta

_KEY = list(KEY_COLUMNS)
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")

# Order matters only for consistent output column ordering; two signals are inverted (lower =
# more confident) and must be handled with higher_is_more_confident=False downstream: this is
# recorded here, not re-derived by callers.
BASELINE_SIGNALS = (
    "candidate_ranker_p_hat",
    "fixed_fused_top1_prob",
    "fixed_fused_margin",
    "fixed_fused_entropy_bits",
    "neural_top1_prob",
    "neural_margin",
    "reliability_g_hat",
    "neural_lm_js_divergence",
)
INVERTED_SIGNALS = ("fixed_fused_entropy_bits", "neural_lm_js_divergence")


def _load_frame_and_attribution(prior_model: str):
    from fusion.data import build_working_frame, load_attribution
    return build_working_frame(prior_model), load_attribution(prior_model)


def build_baseline_confidence_table(prior_model: str, n_inner_folds: int = 5) -> pd.DataFrame:
    frame, attribution = _load_frame_and_attribution(prior_model)

    outer_pieces = []
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
        ranker_piece = applied[_KEY + ["p_hat_fixed_winner"]].rename(
            columns={"p_hat_fixed_winner": "candidate_ranker_p_hat"}
        )

        fixed_fused = fused_posterior_features_frame(
            test_frame["p_neural"], test_frame["p_lm"], fixed_beta
        )
        fixed_fused_piece = pd.concat(
            [test_frame[_KEY].reset_index(drop=True), fixed_fused.reset_index(drop=True)], axis=1
        )
        fixed_fused_piece = fixed_fused_piece.rename(columns={
            "fused_entropy_bits": "fixed_fused_entropy_bits",
            "fused_top1_prob": "fixed_fused_top1_prob",
            "fused_margin": "fixed_fused_margin",
        })

        g_hat_frozen = fit_reliability_frozen(train_frame, test_frame)
        g_hat_piece = test_frame[_KEY].reset_index(drop=True).assign(
            reliability_g_hat=g_hat_frozen.to_numpy()
        )

        selection_joint = (
            test_features[test_features["is_fixed_winner"]][
                _KEY + ["neural_top1_prob", "neural_margin", "js_divergence"]
            ]
            .rename(columns={"js_divergence": "neural_lm_js_divergence"})
        )

        piece = test_frame[_KEY].reset_index(drop=True)
        for other in (ranker_piece, fixed_fused_piece, g_hat_piece, selection_joint):
            n_before = len(piece)
            piece = piece.merge(other, on=_KEY, how="inner", validate="one_to_one")
            if len(piece) != n_before:
                raise ValueError(
                    f"baseline-confidence merge dropped rows for held-out study "
                    f"{held_out_study}: {n_before} -> {len(piece)}"
                )
        piece["outer_held_out_study"] = held_out_study
        piece["fixed_beta"] = fixed_beta
        outer_pieces.append(piece)

    result = pd.concat(outer_pieces, ignore_index=True)
    if len(result) != len(frame):
        raise ValueError(
            f"baseline-confidence table produced {len(result)} rows for {len(frame)} selections"
        )
    return result
