"""Joint neural+language-model features for the joint arbitration gate.

Extends fusion.features (neural-posterior-only) with features from the language-model posterior
and the interaction between the two sources. Like fusion.features, these must never use the true
target index -- they are the runtime input to a deployable gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.features import posterior_entropy_bits, top1_probability, top1_top2_margin

N_SYMBOLS = 36


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def joint_features(p_neural: np.ndarray, p_lm: np.ndarray) -> dict[str, float]:
    p_neural = np.asarray(p_neural, dtype=float)
    p_lm = np.asarray(p_lm, dtype=float)
    if p_neural.shape != (N_SYMBOLS,) or p_lm.shape != (N_SYMBOLS,):
        raise ValueError(f"both posteriors must have shape ({N_SYMBOLS},)")

    neural_winner = int(p_neural.argmax())
    lm_winner = int(p_lm.argmax())
    neural_entropy = posterior_entropy_bits(p_neural)
    lm_entropy = posterior_entropy_bits(p_lm)

    neural_rank_of_lm_winner = int(np.sum(p_neural > p_neural[lm_winner]))
    lm_rank_of_neural_winner = int(np.sum(p_lm > p_lm[neural_winner]))

    return {
        "neural_entropy_bits": neural_entropy,
        "neural_top1_prob": top1_probability(p_neural),
        "neural_margin": top1_top2_margin(p_neural),
        "lm_entropy_bits": lm_entropy,
        "lm_top1_prob": top1_probability(p_lm),
        "lm_margin": top1_top2_margin(p_lm),
        "argmax_agree": float(neural_winner == lm_winner),
        "lm_prob_of_neural_winner": float(p_lm[neural_winner]),
        "neural_prob_of_lm_winner": float(p_neural[lm_winner]),
        "neural_rank_of_lm_winner": neural_rank_of_lm_winner,
        "lm_rank_of_neural_winner": lm_rank_of_neural_winner,
        "kl_neural_from_lm": _kl_divergence(p_neural, p_lm),
        "js_divergence": _js_divergence(p_neural, p_lm),
        "entropy_diff": neural_entropy - lm_entropy,
        "entropy_ratio": neural_entropy / lm_entropy if lm_entropy > 1e-12 else float("inf"),
    }


def joint_features_frame(p_neural_column, p_lm_column) -> pd.DataFrame:
    return pd.DataFrame(
        joint_features(pn, pl) for pn, pl in zip(p_neural_column, p_lm_column)
    )
