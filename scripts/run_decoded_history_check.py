"""Real run of the decoded-history replay robustness check (design spec section 16). No holdout,
no new tuning -- both betas are already frozen from the original Aims 1-4 rescue analysis, recomputed
here only as a read-off via the existing public route_a_dev_wide_best_fixed_beta.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.data import load_attribution, load_selections
from fusion.decoded_history import build_prior_object, replay_decoded_history
from fusion.net_gain import add_net_correct_gain
from fusion.route_a_policy import route_a_dev_wide_best_fixed_beta
from fusion.stats import participant_cluster_bootstrap_rate_diff

PRIORS = ("gpt2", "ngram5_kn")
_KEY = list(KEY_COLUMNS)


def main() -> None:
    selections = load_selections()
    for prior_model in PRIORS:
        attribution = load_attribution(prior_model)
        beta = route_a_dev_wide_best_fixed_beta(attribution)

        print(f"=== {prior_model} (beta={beta}) ===")

        prior_object = build_prior_object(prior_model)
        decoded = replay_decoded_history(selections, prior_object, beta=beta)
        decoded["arm"] = "decoded_history"
        decoded = add_net_correct_gain(decoded)

        ground_truth = attribution[np.isclose(attribution["beta"], beta)][
            _KEY + ["fused_correct", "neural_correct"]
        ].copy()
        if len(ground_truth) != len(selections):
            raise ValueError(
                f"ground-truth-prefix slice has {len(ground_truth)} rows for "
                f"{len(selections)} selections at beta={beta}"
            )
        ground_truth["arm"] = "ground_truth_prefix"
        ground_truth = add_net_correct_gain(ground_truth)

        # KEY_COLUMNS already includes participant_id, and replay_decoded_history/the
        # ground-truth slice both carry it straight through -- no re-attachment merge needed.
        long_frame = pd.concat(
            [
                decoded[["participant_id", "arm", "net_correct_gain"]],
                ground_truth[["participant_id", "arm", "net_correct_gain"]],
            ],
            ignore_index=True,
        )
        point, lower, upper = participant_cluster_bootstrap_rate_diff(
            long_frame, "net_correct_gain", "decoded_history", "ground_truth_prefix"
        )

        print(f"  decoded_history_fused_correct: {decoded['fused_correct'].astype(int).mean()}")
        print(f"  ground_truth_prefix_fused_correct: {ground_truth['fused_correct'].astype(int).mean()}")
        print(f"  decoded_history_net_correct_gain: {decoded['net_correct_gain'].mean()}")
        print(f"  ground_truth_prefix_net_correct_gain: {ground_truth['net_correct_gain'].mean()}")
        print(f"  decoded_minus_ground_truth: point={point}, lower={lower}, upper={upper}")
        print(f"  n_selections: {len(selections)}")


if __name__ == "__main__":
    main()
