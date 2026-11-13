"""Cluster bootstrap for paired per-selection differences (participant, context and crossed
schemes), max-t simultaneous bands across several estimates, and equivalence counts."""
from __future__ import annotations

import numpy as np

SCHEMES = ("participant", "context", "crossed")


def _codes(values):
    _, inverse = np.unique(np.asarray(values), return_inverse=True)
    return inverse, int(inverse.max()) + 1


def bootstrap_weights(p_idx, n_p, c_idx, n_c, scheme, rng):
    """Row weights for one resample. A cluster drawn k times counts k times (multinomial counts),
    so repeated draws are never merged. 'crossed' multiplies independent participant and context
    draws (pigeonhole bootstrap), which is conservative when the two factors are crossed."""
    w = np.ones(len(p_idx))
    if scheme in ("participant", "crossed"):
        w = w * rng.multinomial(n_p, np.full(n_p, 1.0 / n_p))[p_idx]
    if scheme in ("context", "crossed"):
        w = w * rng.multinomial(n_c, np.full(n_c, 1.0 / n_c))[c_idx]
    return w


def bootstrap_means(diffs, participants, contexts, scheme="crossed", n_boot=2000, seed=20260820):
    """diffs: (n,) or (n, k) paired differences (adaptive minus fixed net correct gain per
    selection, one column per prior). The same resample is applied to every column."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    d = np.asarray(diffs, float)
    d = d[:, None] if d.ndim == 1 else d
    p_idx, n_p = _codes(participants)
    c_idx, n_c = _codes(contexts)
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, d.shape[1]), np.nan)
    for b in range(n_boot):
        w = bootstrap_weights(p_idx, n_p, c_idx, n_c, scheme, rng)
        total = w.sum()
        if total > 0:
            out[b] = (w[:, None] * d).sum(0) / total
    return out


def percentile_ci(boot, level=0.95):
    lo = (1 - level) / 2 * 100
    return np.nanpercentile(boot, [lo, 100 - lo], axis=0)


def simultaneous_band(boot, point, level=0.95):
    """Max-t band over all columns: c is the level quantile of max_k |boot_k - point_k| / se_k.
    Columns with zero spread (exact ties) get zero width and stay out of the maximum."""
    se = np.nanstd(boot, axis=0, ddof=1)
    live = se > 0
    t = np.zeros_like(boot)
    t[:, live] = np.abs(boot[:, live] - point[live]) / se[live]
    c = float(np.nanquantile(t.max(axis=1), level))
    return point - c * se, point + c * se, c


def equivalence_counts(lower, upper, margin):
    lower, upper = np.asarray(lower), np.asarray(upper)
    inside = (lower > -margin) & (upper < margin)
    outside = (lower >= margin) | (upper <= -margin)
    return {"inside_margin": int(inside.sum()), "outside_margin": int(outside.sum()),
            "overlapping_margin": int((~inside & ~outside).sum())}
