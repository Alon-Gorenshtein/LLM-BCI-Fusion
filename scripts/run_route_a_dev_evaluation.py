"""Dev-side (out-of-fold) evaluation of the Route A controller, both priors. Run and record this
BEFORE touching StudyB -- this is the last checkpoint to catch a bug or a design problem while it is
still cheap to fix, exactly as the joint gate's dev-side checkpoint was used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.actionability import actionability_by_key, beta_correctness_long
from fusion.data import build_working_frame, load_attribution
from fusion.route_a_holdout import split_dev_test_route_a
from fusion.route_a_policy import run_route_a_policy_selection

PRIORS = ("gpt2", "ngram5_kn")


def main() -> None:
    for prior_model in PRIORS:
        working = build_working_frame(prior_model)
        attribution = load_attribution(prior_model)
        dev_frame, _ = split_dev_test_route_a(working)
        dev_attribution, _ = split_dev_test_route_a(attribution)

        dev_actionability = actionability_by_key(dev_attribution)
        dev_beta_correctness = beta_correctness_long(dev_attribution)
        result = run_route_a_policy_selection(
            dev_frame, dev_attribution, dev_actionability, dev_beta_correctness
        )
        print(f"=== {prior_model} ===")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print(f"  actionable rate: {dev_actionability['actionable'].mean():.4f}")


if __name__ == "__main__":
    main()
