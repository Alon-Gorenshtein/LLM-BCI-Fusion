"""Adaptive versus fixed weighting when the language model sees decoded history.

Both arms are tuned inside the decoded-history setting (see fusion.decoded_history_fair). Differences
are adaptive minus fixed net correct gain per selection; intervals resample participants, and
participants crossed with linguistic contexts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.net_gain import add_net_correct_gain
from fusion.route_a_policy import route_a_dev_wide_best_fixed_beta
from fusion.stats import participant_cluster_bootstrap_rate_diff
from fusion.clustering import bootstrap_means, percentile_ci
from fusion.data import build_working_frame, load_attribution, load_selections
from fusion.decoded_history import build_prior_object, replay_decoded_history
from fusion.decoded_history_fair import CachedPrior, decoded_history_fair_comparison

OUT = Path(__file__).resolve().parents[1] / "output" / "revision"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True, choices=["gpt2", "ngram5_kn"])
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    working = build_working_frame(args.prior)
    prior_object = CachedPrior(build_prior_object(args.prior))
    start = time.time()
    result = decoded_history_fair_comparison(working, prior_object, max_folds=args.max_folds)
    elapsed = time.time() - start
    frame = result["selections"]
    diff = (frame["adaptive_net_gain"] - frame["fixed_net_gain"]).to_numpy()
    summary = {
        "prior": args.prior, "n_selections": int(len(frame)), "seconds": round(elapsed, 1),
        "n_language_model_calls": len(prior_object._cache),
        "mean_fixed_net_gain": float(frame["fixed_net_gain"].mean()),
        "mean_adaptive_net_gain": float(frame["adaptive_net_gain"].mean()),
        "diff_adaptive_minus_fixed": float(diff.mean()),
        "n_adaptive_better": int((diff > 0).sum()), "n_fixed_better": int((diff < 0).sum()),
        "folds": result["folds"],
    }
    if args.max_folds is None:
        for scheme in ("participant", "crossed"):
            boot = bootstrap_means(diff, frame["participant_id"].to_numpy(), frame["context_prefix"].to_numpy(),
                                   scheme=scheme, n_boot=args.n_boot)
            lo, hi = percentile_ci(boot)
            summary[f"ci_{scheme}"] = [float(lo[0]), float(hi[0])]
    if args.max_folds is None:
        # Fixed policy at its tuned beta, decoded history versus ground-truth prefix (the earlier check),
        # run with the same cached language model so it costs one extra pass over the contexts.
        key = list(KEY_COLUMNS)
        attribution = load_attribution(args.prior)
        beta = route_a_dev_wide_best_fixed_beta(attribution)
        decoded = add_net_correct_gain(replay_decoded_history(load_selections(), prior_object, beta=beta))
        decoded["arm"] = "decoded_history"
        truth = add_net_correct_gain(attribution[np.isclose(attribution["beta"], beta)][key + ["fused_correct", "neural_correct"]].copy())
        truth["arm"] = "ground_truth_prefix"
        long_frame = pd.concat([decoded[["participant_id", "arm", "net_correct_gain"]],
                                truth[["participant_id", "arm", "net_correct_gain"]]], ignore_index=True)
        point, lower, upper = participant_cluster_bootstrap_rate_diff(long_frame, "net_correct_gain", "decoded_history", "ground_truth_prefix")
        summary["fixed_policy_replay"] = {
            "beta": float(beta),
            "decoded_history_fused_correct": float(decoded["fused_correct"].astype(int).mean()),
            "ground_truth_prefix_fused_correct": float(truth["fused_correct"].astype(int).mean()),
            "decoded_history_net_correct_gain": float(decoded["net_correct_gain"].mean()),
            "ground_truth_prefix_net_correct_gain": float(truth["net_correct_gain"].mean()),
            "decoded_minus_ground_truth": {"point": float(point), "lower": float(lower), "upper": float(upper)},
        }
    out = Path(args.out) if args.out else OUT / f"decoded_fair_{args.prior}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
