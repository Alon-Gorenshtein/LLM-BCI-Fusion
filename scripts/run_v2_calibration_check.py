"""Version 2 addendum step 1 (design spec S13.2): does cross-fitted temperature calibration of the
two posteriors change the fixed-versus-adaptive picture, before any new gate is built?

Dev-only. `run_calibrated_four_arm_comparison` splits with `fusion.holdout.split_dev_test` and keeps
the dev half alone, so Study N is never read here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.calibration import run_calibrated_four_arm_comparison

if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        result = run_calibrated_four_arm_comparison(prior)
        gain = result["adaptive_calibrated_vs_fixed_net_correct_gain"]
        print(f"--- {prior} ---")
        print(f"full result: {result}")
        print(f"adaptive_calibrated - fixed net_correct_gain: {gain['point']:.4f} "
              f"(95% CI {gain['lower']:.4f} to {gain['upper']:.4f})")
        print(f"mean neural temperature: {result['mean_neural_temperature']:.4f}")
        print(f"mean LM temperature:     {result['mean_lm_temperature']:.4f}")
        print()
