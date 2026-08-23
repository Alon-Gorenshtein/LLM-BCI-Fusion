"""Round 3 addendum, design spec S14.2 item 1: how often is the grid oracle's chosen beta unique,
versus tied among several equally-correct beta values? fusion.arms.select_oracle_arm breaks ties by
highest p_fused_target, then row order -- a real rule, but previously undisclosed and unquantified.
Dev-only (Studies B, F, L): Study N is split out and discarded, never read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import load_attribution
from fusion.holdout import split_dev_test
from fusion.arms import KEY_COLUMNS


def oracle_tie_structure(prior_model: str) -> dict:
    attribution = load_attribution(prior_model)
    dev, _test = split_dev_test(attribution)

    n_correct_betas = (
        dev[dev["fused_correct"]]
        .groupby(list(KEY_COLUMNS))["beta"]
        .nunique()
    )
    all_keys = dev.groupby(list(KEY_COLUMNS)).size()
    n_correct_betas = n_correct_betas.reindex(all_keys.index, fill_value=0)

    distribution = n_correct_betas.value_counts().sort_index()
    return {
        "prior_model": prior_model,
        "n_selections": int(len(all_keys)),
        "distribution_of_num_correct_betas": {int(k): int(v) for k, v in distribution.items()},
        "fraction_unique_oracle_beta": float((n_correct_betas == 1).mean()),
        "fraction_any_tie": float((n_correct_betas >= 2).mean()),
        "fraction_no_correct_beta": float((n_correct_betas == 0).mean()),
    }


if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        result = oracle_tie_structure(prior)
        print(f"--- {prior} ---")
        print(f"n_selections: {result['n_selections']}")
        print(f"distribution of num equally-correct betas: {result['distribution_of_num_correct_betas']}")
        print(f"fraction with a unique oracle-optimal beta: {result['fraction_unique_oracle_beta']:.4f}")
        print(f"fraction with >=2 equally-optimal betas (a tie): {result['fraction_any_tie']:.4f}")
        print(f"fraction where no beta produces a correct selection: {result['fraction_no_correct_beta']:.4f}")
