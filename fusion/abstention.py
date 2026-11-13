# fusion/abstention.py
"""Selective-abstention risk-coverage analysis -- a further
analysis for the case that candidate-arbitration does not clear its own bar. Retains the
`coverage` fraction of selections with the HIGHEST confidence_scores (most confident), flags the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(
    confidence_scores: pd.Series,
    correct: pd.Series,
    coverage_grid: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
) -> pd.DataFrame:
    order = confidence_scores.sort_values(ascending=False).index
    ordered_correct = correct.loc[order].reset_index(drop=True)
    n_total = len(ordered_correct)

    rows = []
    for coverage in coverage_grid:
        n_retained = int(round(coverage * n_total))
        retained = ordered_correct.iloc[:n_retained]
        rows.append({
            "coverage": coverage,
            "n_retained": n_retained,
            "n_flagged": n_total - n_retained,
            "accuracy_retained": float(retained.astype(int).mean()) if n_retained else float("nan"),
        })
    return pd.DataFrame(rows)


def risk_coverage_within_group(
    frame: pd.DataFrame,
    confidence_column: str,
    correct_column: str,
    group_column: str = "outer_held_out_study",
    coverage_grid: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
    higher_is_more_confident: bool = True,
    return_masks: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[float, pd.Series]]:
    """Rank confidence_column WITHIN each group (default: each outer fold's held-out study) rather
    than pooling raw scores across groups fit by different models. Avoids the cross-fold
    calibration-mismatch failure mode a global pooled rank is vulnerable to: with four outer folds
    each fit by a distinct model, pooling raw probabilities and ranking globally can partly rank
    study identity rather than trial confidence. Returns one row per (coverage, group) from
    risk_coverage_curve applied within each group, plus one row per coverage with
    group == "pooled_within_study" aggregating every group's own within-group flagged set.

    If `return_masks` is True, also returns a dict mapping each coverage level in `coverage_grid`
    to the boolean `is_retained` Series (indexed like `frame`, True = retained/not-flagged) used to
    compute that coverage level's pooled row -- the exact within-study-ranked flagged set the
    pooled row's error_fraction_captured/enrichment/accuracy_retained describe. Callers that need
    to reproduce the SAME flagged set outside this function (e.g. for a bootstrap CI on the pooled
    metrics) should use this instead of recomputing a separate rank, to avoid describing a
    different, methodologically inferior (e.g. globally pooled) flagged set than the one the
    pooled row itself reports on. Default False preserves the original single-DataFrame return.
    """
    frame = frame.copy()
    if not higher_is_more_confident:
        frame[confidence_column] = -frame[confidence_column]

    per_group_rows = []
    flagged_masks = {}
    for group_value, group_frame in frame.groupby(group_column):
        curve = risk_coverage_curve(
            group_frame[confidence_column], group_frame[correct_column], coverage_grid=coverage_grid,
        )
        curve[group_column] = group_value
        per_group_rows.append(curve)
        order = group_frame[confidence_column].sort_values(ascending=False).index
        for coverage in coverage_grid:
            n_retained = int(round(coverage * len(group_frame)))
            retained_index = set(order[:n_retained])
            flagged_masks[(group_value, coverage)] = frame.index.to_series().isin(retained_index)

    per_group = pd.concat(per_group_rows, ignore_index=True)

    pooled_rows = []
    is_retained_by_coverage = {}
    for coverage in coverage_grid:
        is_retained = pd.Series(False, index=frame.index)
        for group_value in frame[group_column].unique():
            is_retained = is_retained | flagged_masks[(group_value, coverage)]
        is_retained_by_coverage[coverage] = is_retained
        retained_correct = frame.loc[is_retained, correct_column].astype(bool)
        flagged_correct = frame.loc[~is_retained, correct_column].astype(bool)
        n_total_errors = int((~frame[correct_column].astype(bool)).sum())
        n_flagged = int((~is_retained).sum())
        n_flagged_errors = int((~flagged_correct).sum())
        overall_error_rate = n_total_errors / len(frame) if len(frame) else float("nan")
        flagged_error_rate = n_flagged_errors / n_flagged if n_flagged else float("nan")
        pooled_rows.append({
            "coverage": coverage,
            "n_retained": int(is_retained.sum()),
            "n_flagged": n_flagged,
            "accuracy_retained": float(retained_correct.astype(int).mean()) if is_retained.any() else float("nan"),
            "error_fraction_captured": n_flagged_errors / n_total_errors if n_total_errors else float("nan"),
            "enrichment": flagged_error_rate / overall_error_rate if overall_error_rate else float("nan"),
            group_column: "pooled_within_study",
        })
    pooled = pd.DataFrame(pooled_rows)

    result = pd.concat([per_group, pooled], ignore_index=True)
    if return_masks:
        return result, is_retained_by_coverage
    return result
