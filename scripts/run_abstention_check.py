# scripts/run_abstention_check.py
"""Selective-abstention risk-coverage check (design spec section 17.6) -- the one further
repo-only fallback licensed after Task 7's candidate-arbitration controller failed its own bar for
both priors (task-7b-report.md: net_correct_gain negative, more harmful than helpful overrides,
negative oracle-headroom-recovered, for both gpt2 and ngram5_kn).

Fixed post-hoc by the final whole-branch review (final-review.md MF-1/MF-2/MF-3), which found the
original design invalid on the real data:

- MF-1 (tie-ordering artifact): the original confidence proxy, |p_hat_best_alternative -
  p_hat_fixed_winner|, is exactly 0.0 for ~91-93% of real selections (every case where the ranker's
  top pick already IS the fixed winner). With the data overwhelmingly tied at one value, which rows
  land in the "top X%" was decided by pandas' unstable tie-break order, not by any real signal.
- MF-2 (direction inverted): margin > 0 (the ranker disagrees with the fixed winner) marks the LEAST
  reliable selections (~0.29 accuracy), not the most confident ones -- the opposite of what the old
  code assumed when it retained high-margin rows.
- MF-3 (wrong comparator scored): `correct` was the controller's own post-override decision, not the
  fixed comparator's own correctness, despite the manuscript describing this analysis as measuring
  "the fixed comparator's own errors".

Corrected design: `confidence_scores` is `p_hat_fixed_winner` directly (the ranker's estimated
probability that the fixed-weight winner is the true target) -- continuous, near-monotonically
correlated with the fixed comparator's own correctness, with no meaningful tie mass. `correct` is
the fixed comparator's own correctness (`fused_correct`), reconstructed via the same
per-outer-fold best-fixed-beta slice `scripts/run_candidate_arbitration.py` uses, not the
controller's `chosen_candidate`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.abstention import risk_coverage_curve
from fusion.arms import KEY_COLUMNS
from fusion.data import build_working_frame, load_attribution
from fusion.nested_loso import run_nested_loso

PRIORS = ("gpt2", "ngram5_kn")
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")
_KEY = list(KEY_COLUMNS)


def main() -> None:
    for prior_model in PRIORS:
        print(f"=== {prior_model} ===")
        nested_result = run_nested_loso(prior_model)
        attribution = load_attribution(prior_model)
        # attribution.parquet itself only carries target_symbol, not target_code (same gap Task 7
        # found) -- merge target_code in from build_working_frame, which carries both.
        target_code_lookup = build_working_frame(prior_model)[_KEY + ["target_code"]]
        n_before = len(attribution)
        attribution = attribution.merge(target_code_lookup, on=_KEY, how="inner", validate="many_to_one")
        if len(attribution) != n_before:
            raise ValueError(
                f"target_code merge dropped attribution rows: {n_before} -> {len(attribution)}"
            )

        # Reconstruct the same nested-CV best-fixed comparator run_candidate_arbitration.py uses:
        # for each outer fold's held-out study, score fused_correct at THAT fold's own fixed_beta
        # (recorded per-row in nested_result), never a beta chosen with knowledge of the held-out
        # study's own labels.
        pieces = []
        for held_out_study in _STUDIES:
            fold_rows = nested_result[nested_result["outer_held_out_study"] == held_out_study]
            fixed_beta = float(fold_rows["fixed_beta"].iloc[0])
            fixed_slice = attribution[
                (attribution["study"] == held_out_study)
                & (attribution["beta"].round(6) == round(fixed_beta, 6))
            ][_KEY + ["fused_correct"]]
            pieces.append(fixed_slice)
        best_fixed_arm = pd.concat(pieces, ignore_index=True)

        merged = nested_result.merge(best_fixed_arm, on=_KEY, how="inner", validate="one_to_one")
        if len(merged) != len(nested_result):
            raise ValueError(
                f"best-fixed-arm join dropped rows: {len(nested_result)} -> {len(merged)}"
            )

        confidence_scores = merged["p_hat_fixed_winner"]
        correct = merged["fused_correct"]

        table = risk_coverage_curve(confidence_scores, correct)
        print(table.to_string(index=False))
        print()

        # Empirical sanity check (final-review.md m-8): make the confidence signal's monotonicity
        # visible and auditable in the committed output itself, not just asserted in a report.
        decile = pd.qcut(confidence_scores, 10, duplicates="drop")
        decile_table = (
            pd.DataFrame({"correct": correct, "decile": decile})
            .groupby("decile", observed=True)["correct"]
            .agg(["mean", "count"])
        )
        print("decile breakdown (confidence_scores = p_hat_fixed_winner, ascending):")
        print(decile_table.to_string())
        print()


if __name__ == "__main__":
    main()
