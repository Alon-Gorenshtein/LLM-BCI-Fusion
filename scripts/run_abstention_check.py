# scripts/run_abstention_check.py
"""Selective-abstention risk-coverage check: ranks selections by the candidate ranker's confidence signal
and reports the share of errors captured at each coverage level."""
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
        # attribution.parquet itself only carries target_symbol, not target_code, so
        # merge target_code in from build_working_frame, which carries both.
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

        # Empirical sanity check: make the confidence signal's monotonicity visible and auditable in the output.
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
