# Reliability-Gated Fusion of a Language-Model Prior with a P300 Neural Posterior

Analysis code and result digests for a reanalysis testing whether an online, per-trial reliability
estimate can set a language model's fusion weight better than a single fixed weight, in a P300
brain-computer interface speller.

Gorenshtein A, Omar M, Jia E, Adiniaev Y, Daniel O, Kruskal J, Ahmed M, Brook O, Barash Y, Klang E.
*Reliability-Gated Fusion of a Language-Model Prior with a P300 Neural Posterior.* Manuscript in
preparation, target Nature Biomedical Engineering.

## What the study measures

3,373 archived online P300-speller selections from 47 people with amyotrophic lateral sclerosis
(bigP3BCI, source studies B/F/L/N) were re-decoded offline and fused with a language-model prior
under five successive fusion policies of increasing sophistication, each evaluated on genuinely
held-out data (nested or nested-across-study cross-validation, never fit and scored on the same
fold).

| | |
|---|---|
| Eligible online selections | 3,373 |
| Participants | 47 |
| Source studies | 4 (B, F, L, N) |
| Priors tested head-to-head | gpt2 (strong) and ngram5_kn (classical 5-gram, Kneser-Ney) |
| Consistency sweep | 24 language-model priors, 11 architecture families |

## Headline results

Four fusion-weighting mechanisms were tested against a single out-of-fold-tuned fixed weight; none
beat it on genuinely held-out data.

- **An online reliability gate**, weighting by the neural posterior's own shape, was significantly
  worse than fixed weighting on net accuracy added under the strong prior (difference, -0.0068; 95%
  CI, -0.0128 to -0.0012) and produced significantly more prior capture, a correct neural reading
  overturned into an error (odds ratio, 1.30; 95% CI, 1.09 to 1.56; *P* = .0031).
- **A richer joint neural-and-language-model gate**, and **an actionability-restricted controller**
  built to intervene only on decision-sensitive trials, likewise did not exceed fixed weighting when
  evaluated once on genuinely held-out data.
- **A decision-boundary-aware candidate-arbitration controller** (exact per-trial crossing points
  between candidates, nested leave-one-study-out across all four bigP3BCI studies) still did not
  exceed fixed weighting for either prior: gpt2 net correct gain -0.0024 (42 trials helped, 50 hurt,
  74 unaffected); ngram5_kn net correct gain -0.0095 (15 helped, 47 hurt, 22 unaffected).
- **A 24-prior consistency sweep** spanning 11 architecture families found no adaptive advantage
  either. A grid oracle given each trial's true outcome still exceeded fixed weighting by a wide
  margin, showing real headroom for a better adaptive policy remains unclaimed by any mechanism
  tested here.
- **The one positive result: selective abstention.** The candidate-arbitration ranker's own
  probability that the fixed comparator is correct, used to flag rather than auto-decide the
  least-confident trials, removes a disproportionate share of the fixed system's own errors. Flagging
  the lowest-confidence 5% of selections removed 14.3% of the fixed comparator's errors under gpt2
  (2.86-fold enrichment) and 15.3% under ngram5_kn (3.05-fold); flagging 10% removed 26.9% (gpt2,
  2.69-fold) and 29.5% (ngram5_kn, 2.95-fold).
- **A deployment-realism check** found that conditioning the language-model prior on the decoder's
  own past emissions, rather than the archive's ground-truth prefix, materially worsens outcomes:
  gpt2 net correct gain fell from 0.0193 to 0.0065 (95% CI on the difference, -0.0201 to -0.0068);
  ngram5_kn fell from 0.0895 to 0.0092 (95% CI, -0.1048 to -0.0566). Published ground-truth-prefix
  headline numbers, in this study and elsewhere, overstate real deployment performance, more so for
  weaker priors.

## Layout

```
fusion/                        analysis package
  data.py                      loads the sibling attribution study's materialized bigP3BCI artifacts
  arms.py                      the four fusion-policy arms and their shared key columns
  reliability.py, weighting.py the online reliability gate (Aim 1)
  joint_features.py            joint neural + language-model gate features (Aim 2)
  route_a_*.py                 the actionability-restricted controller ("Route A")
  decision_boundary.py         exact per-candidate beta-crossing math (pairwise_crossing_beta,
                                fixed_winner_stability_interval, candidate_boundary_distance)
  candidate_features.py        the 5-feature candidate table consumed by the arbitration ranker
  candidate_arbitration.py     fits the ranker, applies the override rule, selects delta out-of-fold
  candidate_arbitration_diagnostics.py, nested_loso.py
                                nested leave-one-study-out evaluation and net-correct-gain scoring
  abstention.py                risk-coverage curve for the selective-abstention safety check
  decoded_history.py           the decoded-history-replay deployment-realism check
  calibration.py, ladder.py, transport.py, net_gain.py, stats.py
                                cross-fitted calibration, the 24-prior ladder, transportability, and
                                shared statistics helpers
scripts/                      pipeline entry points: run_candidate_arbitration, run_abstention_check,
                                run_decoded_history_check, run_route_a_*, run_v2_*, run_gate_diagnostics,
                                analyze_oracle_*, build_digest, build_etables, build_figures,
                                assemble_manuscript
tests/                        pytest suite; 137 tests, 116 run standalone with no data at all (see
                                "Reproducing")
results/digests/               results_digest.json plus the plain-text run logs behind every
                                manuscript number (candidate_arbitration_nested_loso.txt,
                                abstention_risk_coverage.txt, decoded_history_check.txt, and others)
results/tables/                eTable1-7, the supplementary tables
results/figures/               Figure 1-2 and eFigure 1-2, 600 DPI, PDF + PNG
```

## Reproducing

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib
python -m pytest tests -q
```

116 of 137 tests are pure unit tests on synthetic inputs and run standalone. The remaining 21 (in
`test_data.py`, `test_arms.py`, `test_net_gain.py`, `test_frozen_single_policy_report.py`,
`test_oracle_structure.py`, `test_oracle_structure_context_grouped.py`, `test_oracle_ties.py`) call
`fusion/data.py`, which requires the sibling `study_bci_llm_authorship` repository's already
materialized `output/intermediate/` as a sibling directory (not distributed here; see "Data").

## Data

- **bigP3BCI** is openly available from PhysioNet and is not redistributed here. DOI
  [10.13026/0byy-ry86](https://doi.org/10.13026/0byy-ry86).
- This study does not re-decode the raw recordings itself. `fusion/data.py` walks upward from its own
  location to find a sibling `study_bci_llm_authorship` checkout and reads that study's already
  materialized selections, neural posteriors, and per-prior attribution tables from its
  `output/intermediate/`, read-only. Reproducing `results/` end to end therefore requires that
  sibling repository's own pipeline to have been run first.
- Language-model weights (gpt2, ngram5_kn, and the 24-prior ladder) were obtained from the Hugging
  Face Hub or fit locally; none are redistributed here.

## Two things worth knowing if you reuse this code

1. **Every probability array must be clipped to `[1e-12, 1.0]` before `np.log()`.** A real selection
   in the archive carries a probability as small as 2.7e-15; an unclipped log on it produces `-inf`
   and crashes the decision-boundary math (`fusion/decision_boundary.py`,
   `fusion/candidate_features.py`). This is an established convention across the package
   (`fusion/calibration.py`, `fusion/features.py`, `fusion/joint_features.py` all clip already); any
   new module that logs a probability needs the same guard.
2. **The selective-abstention confidence signal must be the ranker's raw `p_hat_fixed_winner`, not a
   margin or gap to the runner-up candidate.** A gap-to-runner-up construction is exactly 0.0 for
   roughly 91-93% of real selections (the two leading candidates tie), so ranking by it is decided by
   unstable sort order, not signal, and the direction is backwards: a large gap marks the *least*
   reliable selections, not the most. `fusion/abstention.py`'s `risk_coverage_curve` takes whatever
   `confidence_scores` it is given at face value; the correctness guarantee lives entirely in the
   caller's choice of that argument (see `scripts/run_abstention_check.py`).

## License

MIT (see `LICENSE`). The license covers this code. It does not extend to the bigP3BCI dataset, which
carries its own PhysioNet terms.
