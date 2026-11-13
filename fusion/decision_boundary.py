"""Exact fusion decision-boundary math. The fused log-score for
candidate j is S_j(beta) = beta*log(p_neural[j]) + log(p_lm[j]), linear in beta. For two candidates
j, k, S_j(beta) = S_k(beta) at beta*_jk = -(log p_lm[j] - log p_lm[k]) / (log p_neural[j] -
log p_neural[k]). This module deliberately restricts itself to PAIRWISE comparisons against the
already-known fixed-beta winner, not a full 36-symbol upper-envelope construction: every candidate
feature this study needs is framed relative to the frozen default action, so the pairwise formula is both simpler and exactly what is needed -- verified by direct
comparison against a brute-force beta scan in this module's own test suite.
"""
from __future__ import annotations

import numpy as np


def pairwise_crossing_beta(p_neural: np.ndarray, p_lm: np.ndarray, j: int, k: int) -> float | None:
    log_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
    log_lm = np.log(np.clip(p_lm, 1e-12, 1.0))
    delta_slope = float(log_neural[j] - log_neural[k])
    delta_intercept = float(log_lm[j] - log_lm[k])
    if delta_slope == 0.0:
        return None
    return -delta_intercept / delta_slope


def fixed_winner_stability_interval(
    p_neural: np.ndarray, p_lm: np.ndarray, fixed_beta: float
) -> tuple[float, float]:
    log_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
    log_lm = np.log(np.clip(p_lm, 1e-12, 1.0))
    scores = fixed_beta * log_neural + log_lm
    winner = int(np.argmax(scores))

    lower_candidates = []
    upper_candidates = []
    for k in range(len(p_neural)):
        if k == winner:
            continue
        crossing = pairwise_crossing_beta(p_neural, p_lm, winner, k)
        if crossing is None:
            continue
        delta_slope = log_neural[winner] - log_neural[k]
        if delta_slope > 0 and crossing < fixed_beta:
            lower_candidates.append(crossing)
        elif delta_slope < 0 and crossing > fixed_beta:
            upper_candidates.append(crossing)

    lower = max([0.0] + lower_candidates)
    upper = min(upper_candidates) if upper_candidates else float("inf")
    return lower, upper


def candidate_boundary_distance(
    p_neural: np.ndarray, p_lm: np.ndarray, fixed_beta: float, candidate_index: int
) -> float:
    log_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
    log_lm = np.log(np.clip(p_lm, 1e-12, 1.0))
    scores = fixed_beta * log_neural + log_lm
    winner = int(np.argmax(scores))
    if candidate_index == winner:
        return 0.0
    crossing = pairwise_crossing_beta(p_neural, p_lm, winner, candidate_index)
    if crossing is None:
        # Parallel lines: candidate_index never overtakes the winner (or always would have, in
        # which case it would already be the winner). Signed by which one is permanently ahead.
        return float("inf") if log_lm[winner] >= log_lm[candidate_index] else float("-inf")
    return crossing - fixed_beta
