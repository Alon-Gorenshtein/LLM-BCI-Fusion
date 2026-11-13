Limits of Posterior-Shape-Based Adaptive Neural-Language Fusion Across Language-Model Priors in P300 Brain-Computer Interfaces
Analysis code, results and figures
=========================================================================================================================

This repository holds the analysis code behind the paper. It reanalyses 3,373 online P300-speller selections from 47 people
with amyotrophic lateral sclerosis (BigP3BCI, PhysioNet, CC BY 4.0) and asks whether a trial-level reliability estimate built
from the shape of the neural posterior can set the weight of a language-model prior better than a tuned fixed weight. The
comparison covers 24 language-model priors, from character 5-grams to a 46.7-billion-parameter model.

Main result. Tuned adaptive weighting did not reliably improve on tuned fixed weighting. In 9 of the 24 priors the tuning chose a
constant weight, so the two policies made identical decisions. The mean difference in net correct gain (adaptive minus fixed) was
+0.03 percentage points (95% CI, -0.05 to +0.12).

What is here
------------
fusion/            the analysis package (fusion arms, reliability estimator, fair comparison, intervals, selective prediction)
scripts/           one script per analysis, table or figure
tests/             unit tests (pytest)
output/tables/     supplementary tables as CSV
output/revision/   result files written by the scripts (JSON and text)
output/figures/    figures as PDF
output/*.txt, *.json   results of the earlier controller analyses
requirements.txt   package versions used

Data. The inputs are derived posteriors (neural posteriors, language-model prior posteriors for all 24 priors and per-selection
fusion outcomes; no raw EEG) computed from the open BigP3BCI archive. They are available to reviewers and will be deposited in a
public Zenodo archive with a citable DOI on acceptance. The scoring and decoder code that produced them (authorship/) is part of
that archive.

Reproducing the analyses
========================

1. Data. Unpack the derived-posterior bundle (its README.txt describes it). The folder that holds authorship/ and
   output/intermediate/ is the data root:
       export FUSION_COMPANION_ROOT=<path to the bundle folder>
2. Environment. Python 3.12.9 and the versions in requirements.txt. scikit-learn's GroupKFold assigns folds differently
   in different releases, so use scikit-learn 1.9.1 to reproduce the reported values exactly.
3. Tests:  pytest -q
4. Analyses. Every script is run from the repository root, for example
       python scripts/run_revision_ladder.py            # 24-prior fair comparison, nested anchor selection
       python scripts/run_revision_decoded_fair.py --prior gpt2
   Outputs go to output/revision, output/tables and output/figures; the files shipped here are the ones used in the paper.
   Random seeds are set inside the scripts (20260820 for the analyses, 20260821 for random fold assignments,
   20260822 for the re-scoring sample). The scripts that call a language model (decoded-history replay and
   run_revision_precision_check.py) download the model from its public repository; the other scripts only read the bundle.
5. Table and figure scripts: build_etables.py, build_revision_tables.py, build_figures.py and build_revision_figures.py.

Scripts and what each does (first lines of each script's description)
-------------------------------------------------------------------
analyze_oracle_structure.py: Joint gate, step 2: does the grid oracle's chosen beta have any learnable structure against joint neural+LM features? This is the go/no-go checkpoint -- if the
analyze_oracle_structure_context_grouped.py: Does the oracle-structure AUROC (0.79/0.81, scripts/analyze_oracle_structure.py, participant-grouped) survive grouping by linguistic context
analyze_oracle_ties.py: How often is the grid oracle's chosen beta unique, versus tied among several equally-correct beta values? fusion.arms.select_oracle_arm breaks ties by
build_digest.py: Build output/results_digest.json and output/stats_digest.json -- the manuscript's single source of truth for every reported number. Reuses the verified fusion.* modules exactly as
build_etables.py: Build supplement eTables (CSV + Markdown) from output/results_digest.json.
build_figure3_data.py: Builds output/figure3_data.json: the four already-computed numbers behind Figure 3's mechanistic summary panels. No new controller or statistical analysis -- panel (a) is a
build_figures.py: Build the manuscript's publication figures from output/results_digest.json. Figure 1: 22-evaluable-prior fair, matched-search-space comparison across the model ladder --
build_revision_digest.py: Merge the revision analyses into one JSON digest that the revised text is checked against.
build_revision_figures.py: Figures added or rebuilt for the revision. Run after the revision analyses have written output/revision/*.json. Figure 1 is the study flow diagram; Figure 2 is the 24-prior fair
build_revision_tables.py: Markdown supplementary tables for the revision, built from output/revision/*.json.
figure_kit.py: Matplotlib figure kit. Helvetica, colorblind-safe palette, 600 DPI PDF+PNG, no in-figure titles, panel labels, error bars,
hochberg_kit.py: Figure style helpers: near-monochrome with one saturated accent, red reserved for errors, individual points and shaded 95% CI bands, lowercase bold panel letters, thin quiet chrome, no in-figure title
run_abstention_baseline_comparison.py: Baseline-comparison and robust-uncertainty check for the selective-abstention analysis: does the candidate-arbitration ranker's confidence (candidate_ranker_p_hat) identify the fixed comparator's own 
run_abstention_check.py: Selective-abstention risk-coverage check: ranks selections by the candidate ranker's confidence signal and reports the share of errors captured at each coverage level.
run_candidate_arbitration.py: Real, one-shot-per-prior run of the nested leave-one-study-out candidate-arbitration analysis. Not a holdout in the single-fresh-study sense -- this is the openly disclosed exploratory cross-study eva
run_decoded_history_check.py: Real run of the decoded-history replay robustness check. No holdout, no new tuning -- both betas are already frozen from the original Aims 1-4 rescue analysis, recomputed
run_frozen_single_policy_report.py: Freeze ONE policy per arm, selected using fused_correct (the redesigned primary endpoint's own selection objective) only, then report every
run_gate_diagnostics.py: Run the dev-only gate-component characterization against real data, using the already-frozen (high, low, best_fixed_beta) policy values already
run_revision_decoded_fair.py: Adaptive versus fixed weighting when the language model sees decoded history.
run_revision_envelope.py: Reachable-symbol envelope for every non-null prior and four beta ranges.
run_revision_ladder.py: Fair, matched-search-space comparison for every non-null prior, with nested anchor selection.
run_revision_precision_check.py: Numerical-precision and independent re-scoring evidence for the stored language-model priors.
run_revision_prior_audit.py: Consistency audit of the companion posteriors, one row per non-null prior.
run_revision_reliability.py: Reliability estimator against simpler baselines, and a flash-count summary.
run_revision_selective.py: Selective prediction against standard confidence baselines.
run_route_a_dev_evaluation.py: Dev-side (out-of-fold) evaluation of the Route A controller, both priors. Run and record this BEFORE touching StudyB -- this is the last checkpoint to catch a bug or a design problem while it is
run_route_a_diagnostics.py: (no description)
run_route_a_holdout_evaluation.py: THE ONE-SHOT EVALUATION. Run this exactly once per prior_model. If this script is re-run after its first real invocation for a given prior_model, STOP -- do not act on a second reading, and
run_v2_calibration_check.py: Joint gate, step 1: does cross-fitted temperature calibration of the two posteriors change the fixed-versus-adaptive picture, before any new gate is built?
run_v2_dev_gate_baseline.py: Recover the dev-side baseline comparison that the one-shot holdout run computed but discarded: how the frozen gate scored against the best fixed beta on the very data its thresholds were tuned
run_v2_holdout_evaluation.py: The single real evaluation of the frozen joint gate. Run this exactly once per prior_model. Do not re-run with different thresholds or features after seeing the result -- record
