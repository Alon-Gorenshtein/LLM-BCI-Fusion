# scripts/run_abstention_baseline_comparison.py
"""Baseline-comparison and robust-uncertainty check for the selective-abstention analysis: does the candidate-arbitration ranker's confidence
(candidate_ranker_p_hat) identify the fixed comparator's own errors any better than trivial
baselines computed from the fixed-fused posterior, the neural posterior alone, the original
reliability estimator, or neural-LM disagreement? And does the result survive ranking WITHIN each
outer fold's held-out study (fixing the cross-fold-pooling risk) with participant-cluster
bootstrap CIs?

The bootstrap-CI section below reuses the EXACT within-study-ranked flagged mask
`fusion.abstention.risk_coverage_within_group` computed for the comparison table's own
"pooled_within_study" row (via `return_masks=True`), rather than recomputing a separate global
rank. This keeps the CI describing the same flagged set the comparison table's headline numbers
report on -- a fresh global rank would silently substitute a methodologically inferior (and
different) flagged set, reintroducing a milder version of the cross-fold-pooling problem this
whole plan exists to fix.

CIs are computed for ALL 8 BASELINE_SIGNALS (not just candidate_ranker_p_hat): the manuscript
question this script answers -- does the candidate ranker's confidence beat trivial baselines --
is a comparison, and a comparison needs uncertainty quantification on both sides. Picking a single
"closest" baseline ahead of time to CI would require a judgment call this script should not make;
computing all 8 trivially covers "whichever is closest" and gives the full evidentiary picture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.abstention import risk_coverage_within_group
from fusion.arms import KEY_COLUMNS
from fusion.baseline_confidence import BASELINE_SIGNALS, INVERTED_SIGNALS, build_baseline_confidence_table
from fusion.data import build_working_frame, load_attribution
from fusion.stats import (
    participant_cluster_bootstrap_abstention_metrics,
    participant_cluster_bootstrap_paired_metric_difference,
)

PRIORS = ("gpt2", "ngram5_kn")
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")
_KEY = list(KEY_COLUMNS)
COVERAGE_POINTS = (0.95, 0.90)  # 5% and 10% flagged


def _attach_fused_correct(table: pd.DataFrame, prior_model: str) -> pd.DataFrame:
    attribution = load_attribution(prior_model)
    target_code_lookup = build_working_frame(prior_model)[_KEY + ["target_code"]]
    n_before = len(attribution)
    attribution = attribution.merge(target_code_lookup, on=_KEY, how="inner", validate="many_to_one")
    if len(attribution) != n_before:
        raise ValueError(f"target_code merge dropped attribution rows: {n_before} -> {len(attribution)}")

    pieces = []
    for held_out_study in _STUDIES:
        fold_rows = table[table["outer_held_out_study"] == held_out_study]
        fixed_beta = float(fold_rows["fixed_beta"].iloc[0])
        fixed_slice = attribution[
            (attribution["study"] == held_out_study)
            & (attribution["beta"].round(6) == round(fixed_beta, 6))
        ][_KEY + ["fused_correct"]]
        pieces.append(fixed_slice)
    best_fixed_arm = pd.concat(pieces, ignore_index=True)

    merged = table.merge(best_fixed_arm, on=_KEY, how="inner", validate="one_to_one")
    if len(merged) != len(table):
        raise ValueError(f"best-fixed-arm join dropped rows: {len(table)} -> {len(merged)}")
    return merged


def main() -> None:
    for prior_model in PRIORS:
        print(f"=== {prior_model} ===")
        table = build_baseline_confidence_table(prior_model)
        table = _attach_fused_correct(table, prior_model)

        print("\n--- Comparison table: error_fraction_captured / enrichment / accuracy_retained ---")
        # Keep every signal's own within-study-ranked flagged masks (one per coverage level) so
        # the bootstrap-CI section below can reuse the SAME flagged set this table's
        # "pooled_within_study" row for that signal reports on, instead of recomputing a separate
        # global rank -- for every signal, not just candidate_ranker_p_hat.
        masks_by_signal: dict[str, dict[float, pd.Series]] = {}
        for signal in BASELINE_SIGNALS:
            higher_is_more_confident = signal not in INVERTED_SIGNALS
            curve, masks = risk_coverage_within_group(
                table, signal, "fused_correct", group_column="outer_held_out_study",
                coverage_grid=COVERAGE_POINTS, higher_is_more_confident=higher_is_more_confident,
                return_masks=True,
            )
            masks_by_signal[signal] = masks
            pooled = curve[curve["outer_held_out_study"] == "pooled_within_study"]
            print(f"\nSignal: {signal} (higher_is_more_confident={higher_is_more_confident})")
            print(pooled[["coverage", "error_fraction_captured", "enrichment", "accuracy_retained"]].to_string(index=False))

        print("\n--- Per-study sensitivity (candidate_ranker_p_hat only, 5%/10% flagged) ---")
        curve = risk_coverage_within_group(
            table, "candidate_ranker_p_hat", "fused_correct", group_column="outer_held_out_study",
            coverage_grid=COVERAGE_POINTS, higher_is_more_confident=True,
        )
        per_study = curve[curve["outer_held_out_study"] != "pooled_within_study"]
        print(per_study.to_string(index=False))

        print("\n--- Participant-cluster bootstrap 95% CIs (all 8 signals, 5%/10% flagged) ---")
        for signal in BASELINE_SIGNALS:
            print(f"\nSignal: {signal}")
            for coverage in COVERAGE_POINTS:
                # Reuse the exact within-study-ranked is_retained mask computed above for this
                # signal/coverage level -- the SAME flagged set the comparison table's pooled row
                # for this signal reports on -- rather than recomputing a separate global rank.
                is_retained = masks_by_signal[signal][coverage]
                flagged = ~is_retained
                ci_frame = table.assign(_flagged=flagged)
                metrics = participant_cluster_bootstrap_abstention_metrics(
                    ci_frame, participant_column="participant_id", flagged_column="_flagged",
                    correct_column="fused_correct",
                )
                print(f"  coverage={coverage}:")
                for key, (point, lower, upper) in metrics.items():
                    print(f"    {key}: {point:.4f} (95% CI {lower:.4f} to {upper:.4f})")

        # A CI-overlap comparison between two signals evaluated on the SAME selections is
        # under-powered: both signals' errors are correlated (they are scored on identical trials),
        # so an unpaired bootstrap-CI-overlap check can miss a real, consistent difference. The
        # paired-difference bootstrap below resamples participants ONCE per iteration and applies
        # that SAME resampled set to both the ranker's and the baseline's own flagged sets, directly
        # testing whether candidate_ranker_p_hat's confidence beats each trivial baseline at
        # flagging the fixed comparator's own errors.
        ranker_signal = "candidate_ranker_p_hat"
        other_signals = [signal for signal in BASELINE_SIGNALS if signal != ranker_signal]
        print(f"\n--- Paired-difference bootstrap: {ranker_signal} vs. each baseline ---")
        for signal in other_signals:
            print(f"\n{ranker_signal} - {signal}:")
            for metric in ("error_fraction_captured", "enrichment"):
                for coverage in COVERAGE_POINTS:
                    is_retained_a = masks_by_signal[ranker_signal][coverage]
                    is_retained_b = masks_by_signal[signal][coverage]
                    paired_frame = table.assign(
                        _flagged_a=~is_retained_a, _flagged_b=~is_retained_b,
                    )
                    point, lower, upper = participant_cluster_bootstrap_paired_metric_difference(
                        paired_frame, participant_column="participant_id",
                        flagged_column_a="_flagged_a", flagged_column_b="_flagged_b",
                        correct_column="fused_correct", metric=metric,
                    )
                    sig_flag = " [sig]" if (lower > 0 or upper < 0) else " [ns]"
                    print(
                        f"  {ranker_signal} - {signal} @ coverage={coverage}, {metric}: "
                        f"{point:.4f} ({lower:.4f} to {upper:.4f}){sig_flag}"
                    )

        # Descriptive check (no CI): does the pooled candidate_ranker_p_hat result hold up within
        # each of the 4 held-out studies individually, or is it driven by one study? Slices the
        # SAME frame-wide pooled retained mask already computed for candidate_ranker_p_hat above
        # (rather than re-ranking within each study alone, which risk_coverage_within_group's own
        # per-group rows already do for accuracy_retained but not for error_fraction_captured/
        # enrichment) down to each study's own rows.
        print(f"\n--- Per-study enrichment breakdown ({ranker_signal} only) ---")
        for coverage in COVERAGE_POINTS:
            is_retained = masks_by_signal[ranker_signal][coverage]
            flagged = ~is_retained
            print(f"\ncoverage={coverage}:")
            for study in _STUDIES:
                sub = table[table["outer_held_out_study"] == study]
                sub_correct = sub["fused_correct"].astype(bool)
                sub_flagged = flagged.loc[sub.index]
                n_total_errors = int((~sub_correct).sum())
                n_flagged = int(sub_flagged.sum())
                n_flagged_errors = int((~sub_correct[sub_flagged]).sum()) if n_flagged else 0
                error_fraction_captured = (
                    n_flagged_errors / n_total_errors if n_total_errors else float("nan")
                )
                overall_error_rate = n_total_errors / len(sub_correct) if len(sub_correct) else float("nan")
                flagged_error_rate = n_flagged_errors / n_flagged if n_flagged else float("nan")
                enrichment = (
                    flagged_error_rate / overall_error_rate if overall_error_rate else float("nan")
                )
                print(
                    f"  {study}: error_fraction_captured={error_fraction_captured:.4f}, "
                    f"enrichment={enrichment:.4f} (n={len(sub_correct)}, n_flagged={n_flagged}, "
                    f"n_errors={n_total_errors})"
                )
        print()


if __name__ == "__main__":
    main()
