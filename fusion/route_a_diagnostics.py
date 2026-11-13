"""Route A diagnostics: characterize Stage 1 and Stage 2's
dev-side discrimination on their own terms, and cross-tabulate Route A's Stage 1 against the OLD
gate's already-computed helped/hurt/no-effect deviation breakdown (fusion.arbitration_diagnostics) -- the central engineering question this controller exists to answer: does Route A
correctly suppress the OLD gate's action-inert deviations while preserving its rare consequential
ones?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion.actionability import actionability_by_key, beta_correctness_long
from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID
from fusion.route_a_policy import run_route_a_policy_selection
from fusion.route_a_stage1 import fit_actionability_model
from fusion.route_a_stage2 import fit_beta_utility_models

_KEY = list(KEY_COLUMNS)


def characterize_route_a_components(
    dev_frame: pd.DataFrame, dev_attribution: pd.DataFrame, n_folds: int = 5
) -> dict:
    dev_actionability = actionability_by_key(dev_attribution)
    dev_beta_correctness = beta_correctness_long(dev_attribution)

    p_actionable = fit_actionability_model(dev_frame, dev_actionability, n_folds=n_folds)
    merged_actionability = dev_frame.merge(
        dev_actionability[_KEY + ["actionable"]], on=_KEY, how="inner"
    )
    y_actionable = merged_actionability["actionable"].astype(int).to_numpy()
    stage1_auroc = float(roc_auc_score(y_actionable, p_actionable.to_numpy()))
    stage1_auprc = float(average_precision_score(y_actionable, p_actionable.to_numpy()))

    q_hat_long = fit_beta_utility_models(dev_frame, dev_beta_correctness, n_folds=n_folds)
    stage2_auroc_by_beta = {}
    stage2_auprc_by_beta = {}
    for beta in BETA_GRID:
        rows = q_hat_long[np.isclose(q_hat_long["beta"], beta)]
        labels = dev_beta_correctness[np.isclose(dev_beta_correctness["beta"], beta)]
        merged = rows.merge(labels[_KEY + ["fused_correct"]], on=_KEY, how="inner")
        y = merged["fused_correct"].astype(int).to_numpy()
        stage2_auroc_by_beta[beta] = float(roc_auc_score(y, merged["q_hat"].to_numpy()))
        stage2_auprc_by_beta[beta] = float(average_precision_score(y, merged["q_hat"].to_numpy()))

    return {
        "stage1_auroc": stage1_auroc,
        "stage1_auprc": stage1_auprc,
        "n_actionable": int(y_actionable.sum()),
        "n_total": int(len(dev_frame)),
        "stage2_auroc_by_beta": stage2_auroc_by_beta,
        "stage2_auprc_by_beta": stage2_auprc_by_beta,
    }


def suppression_cross_tab(
    dev_frame: pd.DataFrame,
    dev_attribution: pd.DataFrame,
    old_gate_deviation_outcomes: pd.DataFrame,
    n_folds: int = 5,
    tau: float | None = None,
) -> dict:
    dev_actionability = actionability_by_key(dev_attribution)
    dev_beta_correctness = beta_correctness_long(dev_attribution)

    if tau is None:
        policy = run_route_a_policy_selection(
            dev_frame, dev_attribution, dev_actionability, dev_beta_correctness, n_folds=n_folds
        )
        tau = policy["tau"]

    p_actionable = fit_actionability_model(dev_frame, dev_actionability, n_folds=n_folds)
    keys = dev_frame[_KEY].reset_index(drop=True)
    keys["p_actionable"] = pd.Series(p_actionable).reset_index(drop=True).to_numpy()

    joined = old_gate_deviation_outcomes.merge(keys, on=_KEY, how="inner", validate="one_to_one")
    if len(joined) != len(old_gate_deviation_outcomes):
        raise ValueError(
            f"suppression cross-tab join dropped rows: {len(old_gate_deviation_outcomes)} "
            f"deviation-outcome selections -> {len(joined)} joined rows"
        )
    joined["suppressed"] = joined["p_actionable"] < tau

    result = {}
    for outcome in ("helped", "hurt", "no_effect"):
        subset = joined[joined["deviation_outcome"] == outcome]
        result[outcome] = {"n": int(len(subset)), "suppressed": int(subset["suppressed"].sum())}
    return result
