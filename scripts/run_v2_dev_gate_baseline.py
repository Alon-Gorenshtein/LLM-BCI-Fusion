"""Recover the dev-side baseline comparison that the one-shot holdout run computed but discarded:
how the frozen gate scored against the best fixed beta on the very data its thresholds were tuned
against.

This is NOT a second read of Study N. It calls `run_frozen_gate_selection` and nothing else -- that
function is the dev-only threshold-selection step, it is handed the dev half of the split alone, and
it re-verifies that for itself via `_require_dev_only`. `evaluate_on_holdout` and
`frozen_v_hat_by_key` -- the only paths that touch the holdout -- are never invoked here. The
selection is deterministic, so the `dev_gated_fused_correct` printed below reproduces the value
already recorded in output/v2_holdout_evaluation.txt exactly; that match is the check that this is
the same computation, only reported more fully.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import build_working_frame, load_attribution
from fusion.holdout import split_dev_test
from fusion.final_evaluation import run_frozen_gate_selection

if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        dev_frame, _test_frame = split_dev_test(build_working_frame(prior))
        dev_attribution, _test_attribution = split_dev_test(load_attribution(prior))

        policy = run_frozen_gate_selection(dev_frame, dev_attribution)
        print(f"--- {prior} ---")
        print(f"policy: {policy}")
        print(f"dev best_fixed (beta={policy['dev_wide_best_fixed_beta']}) fused_correct: "
              f"{policy['dev_best_fixed_fused_correct']:.10f}")
        print(f"dev gated fused_correct:                     "
              f"{policy['dev_gated_fused_correct']:.10f}")
        print(f"dev gated - best_fixed:                      "
              f"{policy['dev_gated_minus_best_fixed']:+.10f}")
        print()
