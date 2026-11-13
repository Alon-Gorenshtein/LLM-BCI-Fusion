"""Target-agnostic reliability features computed from a neural posterior alone.

These must never use the true target index -- they are the runtime input to the online
weight estimator, and using target information here would make the estimator
undeployable (and would defeat the point of testing it against an oracle that DOES see
the target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SYMBOLS = 36


def posterior_entropy_bits(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def top1_probability(p: np.ndarray) -> float:
    return float(np.max(np.asarray(p, dtype=float)))


def top1_top2_margin(p: np.ndarray) -> float:
    sorted_p = np.sort(np.asarray(p, dtype=float))[::-1]
    return float(sorted_p[0] - sorted_p[1])


def reliability_features(p_neural: np.ndarray) -> dict[str, float]:
    p = np.asarray(p_neural, dtype=float)
    if p.shape != (N_SYMBOLS,):
        raise ValueError(f"p_neural must have shape ({N_SYMBOLS},), got {p.shape}")
    return {
        "entropy_bits": posterior_entropy_bits(p),
        "top1_prob": top1_probability(p),
        "margin": top1_top2_margin(p),
    }


def reliability_features_frame(p_neural_column) -> pd.DataFrame:
    return pd.DataFrame(reliability_features(p) for p in p_neural_column)
