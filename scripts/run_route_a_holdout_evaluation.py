"""THE ONE-SHOT EVALUATION. Run this exactly once per prior_model. If this script is re-run after
its first real invocation for a given prior_model, STOP -- do not act on a second reading, and
disclose the double-read plainly rather than quietly using the second run's number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.route_a_final_evaluation import evaluate_route_a_on_holdout

PRIORS = ("gpt2", "ngram5_kn")


def main() -> None:
    for prior_model in PRIORS:
        result = evaluate_route_a_on_holdout(prior_model)
        print(f"=== {prior_model} ===")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
