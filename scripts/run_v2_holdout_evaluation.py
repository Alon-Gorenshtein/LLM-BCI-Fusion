"""The single real evaluation the Version 2 addendum exists to produce. Run this exactly once per
prior_model. Do not re-run with different thresholds or features after seeing the result -- record
whichever direction it points, per the design spec S13.6 honesty requirement.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.final_evaluation import evaluate_on_holdout

if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        result = evaluate_on_holdout(prior)
        policy = result["policy"]
        print(f"--- {prior} ---")
        print(f"frozen policy: {policy}")
        print(f"frozen_gate net_correct_gain: {result['frozen_gate_net_correct_gain']:.4f}")
        print(f"best_fixed net_correct_gain:  {result['best_fixed_net_correct_gain']:.4f}")
        diff = result["frozen_gate_minus_best_fixed"]
        print(f"frozen_gate - best_fixed: {diff['point']:.4f} (95% CI {diff['lower']:.4f} to {diff['upper']:.4f})")
        print(f"frozen_gate fused_correct: {result['frozen_gate_fused_correct']:.4f}")
        print(f"best_fixed fused_correct:  {result['best_fixed_fused_correct']:.4f}")
        print(f"applied beta counts: {result['applied_beta_counts']}")
        print(f"test V(x) mean {result['test_v_hat_mean']:.4f} sd {result['test_v_hat_sd']:.4f}")
        print(f"n holdout selections: {result['n_holdout_selections']} "
              f"({result['n_holdout_participants']} participants)")
        print()
