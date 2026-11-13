# scripts/run_revision_selective.py
"""Selective prediction against standard confidence baselines.

For each prior, the eight-signal confidence table (fusion.baseline_confidence) is scored against the
errors of the fixed-weight fused selection, ranking within each outer held-out study. Reported per
signal: AURC and excess AURC over the oracle ranking with 95% participant-cluster bootstrap
intervals, the full pooled risk-coverage curve, the paired AURC difference against the fixed-fused
top-1 probability (the maximum-softmax baseline) with Holm-adjusted bootstrap p values, and the
5% and 10% flagged-selection counts with bootstrap intervals. Reference rows give a random ranking
(AURC equals the error rate) and the oracle ranking.

Usage:
    python scripts/run_revision_selective.py [--prior gpt2 ngram5_kn] [--n-boot 2000] [--out PATH]
Writes PATH (.json, default output/revision/selective.json) and a plain-text summary beside it
with the same stem and a .txt suffix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusion.abstention import risk_coverage_within_group
from fusion.arms import KEY_COLUMNS
from fusion.baseline_confidence import (
    BASELINE_SIGNALS, INVERTED_SIGNALS, build_baseline_confidence_table,
)
from fusion.data import build_working_frame, load_attribution
from fusion.selective import (
    aurc, aurc_optimal, excess_aurc, participant_resamples, pooled_risk_coverage,
)
from fusion.stats import participant_cluster_bootstrap_abstention_metrics

DEFAULT_PRIORS = ("gpt2", "ngram5_kn")
REFERENCE_SIGNAL = "fixed_fused_top1_prob"
FLAGGED_LEVELS = {"5%": 0.95, "10%": 0.90}  # retained coverage for each flagged percentage
SEED = 20260820
_KEY = list(KEY_COLUMNS)
_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")
DEFAULT_OUT = REPO_ROOT / "output" / "revision" / "selective.json"


def _attach_fixed_correct(table: pd.DataFrame, prior_model: str) -> pd.DataFrame:
    """Add `fused_correct`: correctness of the fixed-weight fused selection at each outer fold's
    own fixed beta."""
    attribution = load_attribution(prior_model)
    target_code_lookup = build_working_frame(prior_model)[_KEY + ["target_code"]]
    n_before = len(attribution)
    attribution = attribution.merge(target_code_lookup, on=_KEY, how="inner", validate="many_to_one")
    if len(attribution) != n_before:
        raise ValueError(f"target_code merge dropped attribution rows: {n_before} -> {len(attribution)}")
    pieces = []
    for study in _STUDIES:
        fixed_beta = float(table.loc[table["outer_held_out_study"] == study, "fixed_beta"].iloc[0])
        pieces.append(attribution[
            (attribution["study"] == study) & (attribution["beta"].round(6) == round(fixed_beta, 6))
        ][_KEY + ["fused_correct"]])
    merged = table.merge(pd.concat(pieces, ignore_index=True), on=_KEY, how="inner",
                         validate="one_to_one")
    if len(merged) != len(table):
        raise ValueError(f"fixed-arm join dropped rows: {len(table)} -> {len(merged)}")
    return merged


def _interval(draws: np.ndarray) -> list[float]:
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return [float(lower), float(upper)]


def _bootstrap_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p value for a difference, with a +1 correction in numerator and
    denominator so a finite number of draws never returns exactly zero."""
    n = len(draws)
    p_low = (1 + np.sum(draws <= 0)) / (n + 1)
    p_high = (1 + np.sum(draws >= 0)) / (n + 1)
    return float(min(1.0, 2 * min(p_low, p_high)))


def _confidence_scores(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Larger = more confident for every signal; INVERTED_SIGNALS are negated."""
    scores = {}
    for signal in BASELINE_SIGNALS:
        values = table[signal].to_numpy(float)
        scores[signal] = -values if signal in INVERTED_SIGNALS else values
    return scores


def _flagged_block(table: pd.DataFrame, n_boot: int) -> dict[str, dict]:
    """Per signal and flagged level: counts and participant-cluster bootstrap intervals, on the
    within-study-ranked flagged set of fusion.abstention.risk_coverage_within_group."""
    result: dict[str, dict] = {}
    n_total_errors = int((~table["fused_correct"].astype(bool)).sum())
    for signal in BASELINE_SIGNALS:
        _, masks = risk_coverage_within_group(
            table, signal, "fused_correct", group_column="outer_held_out_study",
            coverage_grid=tuple(FLAGGED_LEVELS.values()),
            higher_is_more_confident=signal not in INVERTED_SIGNALS, return_masks=True,
        )
        per_level = {}
        for label, coverage in FLAGGED_LEVELS.items():
            flagged = ~masks[coverage]
            ci_frame = table.assign(_flagged=flagged)
            metrics = participant_cluster_bootstrap_abstention_metrics(
                ci_frame, participant_column="participant_id", flagged_column="_flagged",
                correct_column="fused_correct", n_bootstrap=n_boot,
            )
            n_flagged_errors = int((~table.loc[flagged, "fused_correct"].astype(bool)).sum())
            per_level[label] = {
                "retained_coverage": coverage,
                "selections_flagged": int(flagged.sum()),
                "errors_total": n_total_errors,
                "errors_among_flagged": n_flagged_errors,
                "error_fraction_captured": dict(zip(("point", "lower", "upper"),
                                                    metrics["error_fraction_captured"])),
                "enrichment": dict(zip(("point", "lower", "upper"), metrics["enrichment"])),
            }
        result[signal] = per_level
    return result


def analyse_prior(prior_model: str, n_boot: int) -> dict:
    table = build_baseline_confidence_table(prior_model)
    table = _attach_fixed_correct(table, prior_model).reset_index(drop=True)

    errors = 1 - table["fused_correct"].astype(int).to_numpy()
    groups = table["outer_held_out_study"].to_numpy()
    participants = table["participant_id"].to_numpy()
    scores = _confidence_scores(table)
    error_rate = float(errors.mean())

    # Point estimates on the full data.
    oracle_aurc = aurc_optimal(errors, groups)
    curves = {}
    point = {}
    for signal, score in scores.items():
        coverage, risk = pooled_risk_coverage(score, errors, groups)
        curves[signal] = {"coverage": coverage.tolist(), "risk": risk.tolist()}
        point[signal] = aurc(score, errors, groups)

    # Paired participant-cluster bootstrap: every signal, the oracle ranking and the overall
    # error rate (the expected AURC of a random ranking) share the same resamples.
    names = list(scores)
    draws = np.empty((n_boot, len(names)))
    oracle_draws = np.empty(n_boot)
    random_draws = np.empty(n_boot)
    for b, idx in enumerate(participant_resamples(participants, n_boot, SEED)):
        for j, signal in enumerate(names):
            draws[b, j] = aurc(scores[signal][idx], errors[idx], groups[idx])
        oracle_draws[b] = aurc_optimal(errors[idx], groups[idx])
        random_draws[b] = errors[idx].mean()
    column = {name: j for j, name in enumerate(names)}

    signals_out = {}
    for signal in names:
        d = draws[:, column[signal]]
        signals_out[signal] = {
            "inverted_and_negated": signal in INVERTED_SIGNALS,
            "aurc": {"point": point[signal], "interval": _interval(d)},
            "excess_aurc": {"point": point[signal] - oracle_aurc,
                            "interval": _interval(d - oracle_draws)},
            "risk_coverage": curves[signal],
        }

    contrasts = [s for s in names if s != REFERENCE_SIGNAL]
    raw_p, rows = [], {}
    ref_draws = draws[:, column[REFERENCE_SIGNAL]]
    for signal in contrasts:
        diff = draws[:, column[signal]] - ref_draws
        raw_p.append(_bootstrap_p(diff))
        rows[signal] = {
            "aurc_difference_vs_reference": {
                "point": point[signal] - point[REFERENCE_SIGNAL], "interval": _interval(diff),
                "note": "negative favours the signal (lower AURC)",
            },
            "p_raw": raw_p[-1],
        }
    holm = multipletests(raw_p, method="holm")[1]
    for signal, p_adj in zip(contrasts, holm):
        rows[signal]["p_holm"] = float(p_adj)
        signals_out[signal]["vs_reference"] = rows[signal]

    flagged = _flagged_block(table, n_boot)
    for signal in names:
        signals_out[signal]["flagged"] = flagged[signal]

    return {
        "prior": prior_model,
        "n_selections": int(len(table)),
        "n_participants": int(len(np.unique(participants))),
        "n_errors": int(errors.sum()),
        "error_rate": error_rate,
        "reference_signal": REFERENCE_SIGNAL,
        "n_contrasts_holm": len(contrasts),
        "signals": signals_out,
        "references": {
            "random_ranking": {
                "aurc": {"point": error_rate, "interval": _interval(random_draws)},
                "excess_aurc": {"point": error_rate - oracle_aurc,
                                "interval": _interval(random_draws - oracle_draws)},
            },
            "oracle_ranking": {
                "aurc": {"point": oracle_aurc, "interval": _interval(oracle_draws)},
                "excess_aurc": {"point": 0.0, "interval": [0.0, 0.0]},
            },
        },
    }


def _fmt(entry: dict, digits: int = 4) -> str:
    lo, hi = entry["interval"]
    return f"{entry['point']:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def render_text(results: dict) -> str:
    lines = [
        "Selective prediction against standard confidence baselines",
        f"Participant-cluster bootstrap draws: {results['n_boot']}; coverage grid 0.01 to 1.00 "
        f"(100 points); ranking within each outer held-out study.",
        "Lower AURC is better. Inverted signals (entropy, JS divergence) are negated before ranking.",
    ]
    for prior_model, res in results["priors"].items():
        lines += [
            "",
            f"=== {prior_model} ===",
            f"Selections {res['n_selections']}, participants {res['n_participants']}, "
            f"fixed-weight errors {res['n_errors']} (error rate {res['error_rate']:.4f})",
            "",
            "AURC and excess AURC (95% interval)",
        ]
        for signal, row in res["signals"].items():
            lines.append(f"  {signal:28s} AURC {_fmt(row['aurc'])}  excess {_fmt(row['excess_aurc'])}")
        for name, row in res["references"].items():
            lines.append(f"  {name:28s} AURC {_fmt(row['aurc'])}  excess {_fmt(row['excess_aurc'])}")
        lines += [
            "",
            f"AURC difference against {res['reference_signal']} (negative favours the signal); "
            f"Holm across {res['n_contrasts_holm']} contrasts",
        ]
        for signal, row in res["signals"].items():
            if "vs_reference" not in row:
                continue
            vs = row["vs_reference"]
            lines.append(
                f"  {signal:28s} {_fmt(vs['aurc_difference_vs_reference'])}  "
                f"p {vs['p_raw']:.4f}  Holm p {vs['p_holm']:.4f}"
            )
        for label in FLAGGED_LEVELS:
            lines += ["", f"{label} of selections flagged (lowest confidence, within study)"]
            for signal, row in res["signals"].items():
                f = row["flagged"][label]
                ec, en = f["error_fraction_captured"], f["enrichment"]
                lines.append(
                    f"  {signal:28s} flagged {f['selections_flagged']}, errors total "
                    f"{f['errors_total']}, errors among flagged {f['errors_among_flagged']}; "
                    f"captured {ec['point']:.4f} ({ec['lower']:.4f} to {ec['upper']:.4f}); "
                    f"enrichment {en['point']:.3f} ({en['lower']:.3f} to {en['upper']:.3f})"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prior", nargs="+", default=list(DEFAULT_PRIORS),
                        help="prior model keys to analyse (default: gpt2 ngram5_kn)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="participant-cluster bootstrap draws (default: 2000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output .json path; a .txt summary is written beside it")
    args = parser.parse_args()

    results = {"n_boot": args.n_boot, "seed": SEED, "priors": {}}
    for prior_model in args.prior:
        print(f"=== {prior_model} ===", flush=True)
        results["priors"][prior_model] = analyse_prior(prior_model, args.n_boot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    text_path = args.out.with_suffix(".txt")
    text_path.write_text(render_text(results))
    print(f"wrote {args.out}\nwrote {text_path}")


if __name__ == "__main__":
    main()
