"""Baseline confidence features computed from the FIXED-weight fused posterior itself -- the
simplest possible "does the current decoder already know it's unsure" signals, used as trivial
baselines the candidate-arbitration ranker's own learned confidence must beat to justify its added
complexity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.features import posterior_entropy_bits, top1_probability, top1_top2_margin

N_SYMBOLS = 36


def fused_posterior(p_neural: np.ndarray, p_lm: np.ndarray, beta: float) -> np.ndarray:
    """The fixed-beta fused 36-way posterior: softmax(beta * log p_neural + log p_lm)."""
    p_neural = np.asarray(p_neural, dtype=float)
    p_lm = np.asarray(p_lm, dtype=float)
    if p_neural.shape != (N_SYMBOLS,) or p_lm.shape != (N_SYMBOLS,):
        raise ValueError(f"both posteriors must have shape ({N_SYMBOLS},)")
    log_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
    log_lm = np.log(np.clip(p_lm, 1e-12, 1.0))
    scores = beta * log_neural + log_lm
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def fused_posterior_features(p_neural: np.ndarray, p_lm: np.ndarray, beta: float) -> dict[str, float]:
    p_fused = fused_posterior(p_neural, p_lm, beta)
    return {
        "fused_entropy_bits": posterior_entropy_bits(p_fused),
        "fused_top1_prob": top1_probability(p_fused),
        "fused_margin": top1_top2_margin(p_fused),
    }


def fused_posterior_features_frame(p_neural_column, p_lm_column, beta: float) -> pd.DataFrame:
    return pd.DataFrame(
        fused_posterior_features(pn, pl, beta) for pn, pl in zip(p_neural_column, p_lm_column)
    )
