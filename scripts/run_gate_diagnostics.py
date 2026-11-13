"""Run the dev-only gate-component characterization
against real data, using the already-frozen (high, low, best_fixed_beta) policy values already
recorded by the one-shot holdout run (scripts/run_v2_holdout_evaluation.py) for each prior. Reading those already-recorded values
here is not a new Study N read -- they were selected using dev data only, and this script does not call evaluate_on_holdout or evaluate_frozen_policy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import build_working_frame, load_attribution
from fusion.holdout import split_dev_test
from fusion.arbitration_diagnostics import (
    characterize_gate_components,
    classify_gate_deviation_outcomes,
)

# Already-frozen dev-side policy values for each prior, as recorded by the one-shot holdout run
# (scripts/run_v2_holdout_evaluation.py): gpt2 high=0.02/low=-0.2/best_fixed_beta=1.0;
# ngram5_kn high=0.2/low=-0.02/best_fixed_beta=0.5.
_FROZEN_POLICIES = {
    "gpt2": {"high": 0.02, "low": -0.2, "best_fixed_beta": 1.0},
    "ngram5_kn": {"high": 0.2, "low": -0.02, "best_fixed_beta": 0.5},
}

if __name__ == "__main__":
    for prior, policy in _FROZEN_POLICIES.items():
        working = build_working_frame(prior)
        attribution = load_attribution(prior)
        dev_frame, _test_frame = split_dev_test(working)
        dev_attribution, _test_attribution = split_dev_test(attribution)

        result = characterize_gate_components(
            dev_frame, dev_attribution,
            high=policy["high"], low=policy["low"], best_fixed_beta=policy["best_fixed_beta"],
        )
        print(f"--- {prior} ---")
        print(f"phantom_agreement classifier: AUROC={result['phantom_agreement_auroc']:.4f} "
              f"AUPRC={result['phantom_agreement_auprc']:.4f} n_events={result['phantom_agreement_n_events']}")
        print(f"prior_capture classifier: AUROC={result['prior_capture_auroc']:.4f} "
              f"AUPRC={result['prior_capture_auprc']:.4f} n_events={result['prior_capture_n_events']}")
        print(f"V(x) quartiles [min, Q1, median, Q3, max]: {[round(q, 4) for q in result['v_hat_quartiles']]}")
        print(f"fraction of dev selections where the gate deviates from best-fixed beta: "
              f"{result['fraction_deviating_from_best_fixed']:.4f}")
        print(f"  of which, favors more LM influence: {result['fraction_favoring_lm']:.4f}")
        print(f"  of which, favors more neural influence: {result['fraction_favoring_neural']:.4f}")
        print(f"n_selections: {result['n_selections']}")

        outcomes = classify_gate_deviation_outcomes(
            result["gated_beta_by_selection"], dev_attribution,
            best_fixed_beta=policy["best_fixed_beta"],
        )
        n_dev = outcomes["n_deviating"]
        print(f"among the {n_dev} deviating selections, the deviation:")
        for label in ["helped", "hurt", "no_effect"]:
            share = outcomes[label] / n_dev if n_dev else float("nan")
            print(f"  {label}: {outcomes[label]} ({share:.4f} of deviating)")
