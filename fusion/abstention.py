# fusion/abstention.py
"""Selective-abstention risk-coverage analysis (design spec section 17.6) -- the one further
repo-only fallback licensed if candidate-arbitration itself does not clear its own bar. Retains the
`coverage` fraction of selections with the HIGHEST confidence_scores (most confident), flags the rest.
"""
from __future__ import annotations

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
