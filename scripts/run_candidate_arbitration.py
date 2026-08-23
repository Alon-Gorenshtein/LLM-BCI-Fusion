# scripts/run_candidate_arbitration.py
"""Real, one-shot-per-prior run of the nested leave-one-study-out candidate-arbitration analysis
(design spec section 17). Not a holdout in the single-fresh-study sense -- this is the honestly
disclosed exploratory cross-study evaluation section 17.3 specifies. There is no reason to call this
"one-shot" the way Study N/StudyB's evaluations were: rerunning it would not contaminate a reserved
holdout (there is none here), but DOES still cost real compute (four outer folds x inner CV per
prior), so run it deliberately rather than repeatedly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.arms import KEY_COLUMNS
from fusion.candidate_arbitration_diagnostics import summarize_nested_loso
from fusion.data import build_working_frame, load_attribution
from fusion.nested_loso import run_nested_loso

PRIORS = ("gpt2", "ngram5_kn")
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")
_KEY = list(KEY_COLUMNS)


def main() -> None:
    for prior_model in PRIORS:
        print(f"=== {prior_model} ===")
        attribution = load_attribution(prior_model)
        # summarize_nested_loso needs target_code on `attribution`, but the real attribution.parquet
        # only carries target_symbol (the unit test's own fixture fakes target_code directly, which
        # masked this gap) -- merge it in from build_working_frame, which already carries both.
        target_code_lookup = build_working_frame(prior_model)[_KEY + ["target_code"]]
        n_before = len(attribution)
        attribution = attribution.merge(target_code_lookup, on=_KEY, how="inner", validate="many_to_one")
        if len(attribution) != n_before:
            raise ValueError(
                f"target_code merge dropped attribution rows: {n_before} -> {len(attribution)}"
            )
        nested_result = run_nested_loso(prior_model)

        # Reconstruct the nested-CV best-fixed comparator: for each outer fold's held-out study,
        # score fused_correct at THAT fold's own fixed_beta (already recorded per-row in
        # nested_result), never a beta chosen with knowledge of the held-out study's own labels.
        key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
        pieces = []
        for held_out_study in _STUDIES:
            fold_rows = nested_result[nested_result["outer_held_out_study"] == held_out_study]
            fixed_beta = float(fold_rows["fixed_beta"].iloc[0])
            fixed_slice = attribution[
                (attribution["study"] == held_out_study)
                & (attribution["beta"].round(6) == round(fixed_beta, 6))
            ][key_cols + ["fused_correct", "neural_correct"]]
            pieces.append(fixed_slice)
        best_fixed_arm = pd.concat(pieces, ignore_index=True)

        summary = summarize_nested_loso(nested_result, attribution, best_fixed_arm)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        for held_out_study in _STUDIES:
            fold = nested_result[nested_result["outer_held_out_study"] == held_out_study]
            print(
                f"  fold {held_out_study}: n={len(fold)}, fixed_beta={fold['fixed_beta'].iloc[0]}, "
                f"delta={fold['delta'].iloc[0]}, intervention_rate={fold['overridden'].astype(int).mean()}"
            )


if __name__ == "__main__":
    main()
