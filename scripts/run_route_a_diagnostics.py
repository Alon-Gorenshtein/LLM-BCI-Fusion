from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import build_working_frame, load_attribution
from fusion.route_a_diagnostics import characterize_route_a_components
from fusion.route_a_holdout import split_dev_test_route_a

PRIORS = ("gpt2", "ngram5_kn")


def main() -> None:
    for prior_model in PRIORS:
        working = build_working_frame(prior_model)
        attribution = load_attribution(prior_model)
        dev_frame, _ = split_dev_test_route_a(working)
        dev_attribution, _ = split_dev_test_route_a(attribution)

        result = characterize_route_a_components(dev_frame, dev_attribution)
        print(f"=== {prior_model} ===")
        print(f"  Stage 1 AUROC: {result['stage1_auroc']:.4f}, AUPRC: {result['stage1_auprc']:.4f}")
        print(f"  n_actionable: {result['n_actionable']} of {result['n_total']}")
        for beta in sorted(result["stage2_auroc_by_beta"]):
            print(
                f"  Stage 2 @ beta={beta}: AUROC={result['stage2_auroc_by_beta'][beta]:.4f}, "
                f"AUPRC={result['stage2_auprc_by_beta'][beta]:.4f}"
            )


if __name__ == "__main__":
    main()
