"""Round 3 addendum, design spec S14.2 item 2: freeze ONE policy per arm, selected using
fused_correct (the redesigned primary endpoint's own selection objective) only, then report every
outcome under that same frozen policy -- rather than the original per-outcome policy selection
(design spec S3.1 Limitation 6). Reuses fusion.rescue's existing, already-verified functions: both
already return every outcome column for whichever beta/anchor they chose, so no new fitting is
needed here, only reading the other outcome columns off the frame each function already produces.
This is the original Aims 1-4 rescue analysis's own participant-grouped cross-validation across all
four studies -- it has no Study-N-holdout concept and touches nothing the Version 2 addendum
protects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import load_attribution, build_working_frame
from fusion.rescue import select_best_fixed_beta_arm_cv, tune_adaptive_anchor_cv
from fusion.net_gain import add_net_correct_gain
from fusion.stats import participant_cluster_bootstrap_rate_diff

_OUTCOMES = ("prior_capture", "phantom_agreement", "fused_correct", "net_correct_gain")


def run_frozen_single_policy_report(prior_model: str, n_folds: int = 5) -> dict:
    attribution = load_attribution(prior_model)
    working = build_working_frame(prior_model)

    fixed = select_best_fixed_beta_arm_cv(attribution, n_folds=n_folds, outcome="fused_correct")
    fixed = add_net_correct_gain(fixed)
    adaptive = tune_adaptive_anchor_cv(working, n_folds=n_folds, outcome="fused_correct")
    adaptive = add_net_correct_gain(adaptive)

    long_frame = pd.concat(
        [fixed[["participant_id", "arm"] + list(_OUTCOMES)],
         adaptive[["participant_id", "arm"] + list(_OUTCOMES)]],
        ignore_index=True,
    )

    report = {"prior_model": prior_model, "policy_selected_on": "fused_correct", "outcomes": {}}
    for outcome in _OUTCOMES:
        point, lower, upper = participant_cluster_bootstrap_rate_diff(
            long_frame, outcome, "tuned_adaptive", "best_fixed"
        )
        report["outcomes"][outcome] = {
            "fixed_rate": float(fixed[outcome].mean()),
            "adaptive_rate": float(adaptive[outcome].mean()),
            "difference": point,
            "lower_ci": lower,
            "upper_ci": upper,
        }
    return report


if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        result = run_frozen_single_policy_report(prior)
        print(f"--- {prior} (policy frozen on {result['policy_selected_on']}) ---")
        for outcome, entry in result["outcomes"].items():
            print(
                f"  {outcome}: fixed={entry['fixed_rate']:.4f} adaptive={entry['adaptive_rate']:.4f} "
                f"diff={entry['difference']:.4f} (95% CI {entry['lower_ci']:.4f} to {entry['upper_ci']:.4f})"
            )
