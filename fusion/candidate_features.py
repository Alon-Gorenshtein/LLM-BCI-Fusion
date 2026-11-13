"""Candidate-level feature table: one row per (selection,
candidate) pair, combining the achievable-candidate set, exact boundary-distance math, and the already-validated joint neural+LM selection-level feature set (section 13.3),
plus per-candidate features specific to arbitration (neural/LM probability and rank of THIS
candidate, whether it is the neural-only/LM-only/fixed winner).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.candidates import achievable_candidates_by_key
from fusion.decision_boundary import candidate_boundary_distance, fixed_winner_stability_interval
from fusion.joint_features import joint_features_frame

_KEY = list(KEY_COLUMNS)


def candidate_feature_table(
    frame: pd.DataFrame, attribution: pd.DataFrame, fixed_beta: float
) -> pd.DataFrame:
    candidates = achievable_candidates_by_key(attribution)
    merged = frame.merge(candidates, on=_KEY, how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise ValueError(
            f"candidate-set join dropped rows: {len(frame)} selections -> {len(merged)} joined rows"
        )

    joint = joint_features_frame(merged["p_neural"], merged["p_lm"]).reset_index(drop=True)
    joint_columns = joint.columns.tolist()

    rows = []
    for position, selection in enumerate(merged.itertuples(index=False)):
        p_neural = np.asarray(selection.p_neural)
        p_lm = np.asarray(selection.p_lm)
        target_index = selection.target_code - 1
        fused_log_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
        fused_log_lm = np.log(np.clip(p_lm, 1e-12, 1.0))
        fixed_scores = fixed_beta * fused_log_neural + fused_log_lm
        fixed_winner = int(np.argmax(fixed_scores))
        neural_winner = int(np.argmax(p_neural))
        lm_winner = int(np.argmax(p_lm))
        _, stability_upper = fixed_winner_stability_interval(p_neural, p_lm, fixed_beta)
        stability_lower, _ = fixed_winner_stability_interval(p_neural, p_lm, fixed_beta)
        stability_width = stability_upper - stability_lower

        neural_rank = {
            int(symbol): int(rank)
            for rank, symbol in enumerate(np.argsort(-p_neural))
        }
        lm_rank = {
            int(symbol): int(rank)
            for rank, symbol in enumerate(np.argsort(-p_lm))
        }

        for candidate_index in selection.candidates:
            boundary_distance = candidate_boundary_distance(
                p_neural, p_lm, fixed_beta, candidate_index
            )
            row = {
                **{column: getattr(selection, column) for column in _KEY},
                "candidate_index": int(candidate_index),
                "is_target": bool(candidate_index == target_index),
                "neural_p_candidate": float(p_neural[candidate_index]),
                "neural_rank_candidate": neural_rank[candidate_index],
                "lm_p_candidate": float(p_lm[candidate_index]),
                "lm_rank_candidate": lm_rank[candidate_index],
                "fixed_fusion_p_candidate": float(
                    np.exp(fixed_scores[candidate_index] - fixed_scores.max())
                    / np.exp(fixed_scores - fixed_scores.max()).sum()
                ),
                "neural_lm_log_odds_candidate": float(
                    fused_log_neural[candidate_index] - fused_log_lm[candidate_index]
                ),
                "boundary_distance": boundary_distance,
                "stability_interval_width": stability_width,
                "is_neural_winner": bool(candidate_index == neural_winner),
                "is_lm_winner": bool(candidate_index == lm_winner),
                "is_fixed_winner": bool(candidate_index == fixed_winner),
            }
            for column in joint_columns:
                row[column] = joint.iloc[position][column]
            rows.append(row)

    return pd.DataFrame(rows)
