"""Selective-prediction summaries: pooled risk-coverage curves, area under the risk-coverage curve
(AURC), excess AURC over the oracle ranking, and a participant-cluster bootstrap of AURC."""
from __future__ import annotations

import numpy as np


def pooled_risk_coverage(scores, errors, groups, n_points=100):
    """Within each group (held-out study), keep the round(c * n_g) most confident selections, pool
    the kept errors across groups, and return (coverage grid, selective risk). Ranking inside each
    group matches the repo's convention and avoids ranking study identity."""
    scores, errors, groups = np.asarray(scores, float), np.asarray(errors, int), np.asarray(groups)
    cums, sizes = {}, {}
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        cums[g] = np.concatenate([[0], np.cumsum(errors[order])])
        sizes[g] = len(idx)
    coverage = np.arange(1, n_points + 1) / n_points
    risk = np.empty(n_points)
    for j, c in enumerate(coverage):
        kept = {g: max(1, int(round(c * sizes[g]))) for g in sizes}
        risk[j] = sum(cums[g][kept[g]] for g in sizes) / sum(kept.values())
    return coverage, risk


def aurc(scores, errors, groups, n_points=100):
    return float(pooled_risk_coverage(scores, errors, groups, n_points)[1].mean())


def aurc_optimal(errors, groups, n_points=100):
    """Oracle ranking: every correct selection before every error."""
    return aurc(-np.asarray(errors, float), errors, groups, n_points)


def excess_aurc(scores, errors, groups, n_points=100):
    return aurc(scores, errors, groups, n_points) - aurc_optimal(errors, groups, n_points)


def participant_resamples(participants, n_boot, seed=20260820):
    """Yield n_boot row-index arrays, each a participant-cluster resample with replacement
    (resampled blocks are concatenated, so repeated draws stay repeated)."""
    participants = np.asarray(participants)
    rng = np.random.default_rng(seed)
    uniq = np.unique(participants)
    rows = {p: np.flatnonzero(participants == p) for p in uniq}
    for _ in range(n_boot):
        yield np.concatenate([rows[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])


def participant_bootstrap_aurc(scores_by_signal, errors, groups, participants,
                               n_boot=2000, seed=20260820):
    """Same participant resample for every signal, so differences between signals are paired.
    Resampled blocks are concatenated (repeated draws stay repeated). Returns (names, (n_boot, k))."""
    errors, groups = np.asarray(errors), np.asarray(groups)
    names = list(scores_by_signal)
    out = np.empty((n_boot, len(names)))
    for b, idx in enumerate(participant_resamples(participants, n_boot, seed)):
        for j, name in enumerate(names):
            out[b, j] = aurc(np.asarray(scores_by_signal[name])[idx], errors[idx], groups[idx])
    return names, out
