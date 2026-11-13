"""Reachable-symbol envelope for every non-null prior and four beta ranges.

For each selection, the set of symbols that can be the fused argmax for some beta in the range is
computed exactly (fusion.envelope.reachable_symbols), except for the 5-point grid range, where it
is the set of argmax symbols at the five grid betas. Per prior and range the script reports the
share of selections with 1, 2 and 3 or more reachable symbols, the reachable-correct rate (share of
selections whose target is reachable; a tie-free ceiling) and the headroom of that rate over the
best fixed-weight fused correctness (single beta chosen out-of-fold on participant-grouped folds).

Usage:
    python scripts/run_revision_envelope.py [--priors gpt2,ngram5_kn] [--ranges widest_anchor]
                                            [--out output/revision/envelope.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fusion.data import AUTHORSHIP_ROOT, build_working_frame, load_attribution
from fusion.envelope import reachable_symbols
from fusion.rescue import BETA_GRID, select_best_fixed_beta_arm_cv

from authorship.attribution import PROBABILITY_FLOOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "revision" / "envelope.json"

# name -> (lower beta, upper beta); None marks the discrete grid range
RANGES: dict[str, tuple[float, float] | None] = {
    "grid5": None,
    "default_adaptive": (0.25, 4.0),
    "widest_anchor": (0.1, 20.0),
    "full": (1e-9, 1e9),
}


def non_null_priors() -> list[str]:
    frame = pd.read_parquet(
        AUTHORSHIP_ROOT / "output" / "intermediate" / "attribution.parquet", columns=["prior_model"]
    )
    return sorted(m for m in frame["prior_model"].unique() if m != "uniform")


def _log_matrices(working: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    neural = np.stack([np.asarray(p, float) for p in working["p_neural"]])
    prior = np.stack([np.asarray(p, float) for p in working["p_lm"]])
    return (np.log(np.maximum(neural, PROBABILITY_FLOOR)),
            np.log(np.maximum(prior, PROBABILITY_FLOOR)))


def reachable_sets(working: pd.DataFrame, lo_hi: tuple[float, float] | None) -> list[tuple[int, ...]]:
    """Candidate set per selection: symbols that are the fused argmax for some beta in the range,
    plus the neural-only argmax (always available to a policy; the earlier grid audit included it)."""
    a, b = _log_matrices(working)
    neural_only = np.argmax(a, axis=1)
    if lo_hi is None:
        winners = np.stack([np.argmax(beta * a + b, axis=1) for beta in BETA_GRID], axis=1)
        return [tuple(sorted(set(int(j) for j in row) | {int(n)})) for row, n in zip(winners, neural_only)]
    lo, hi = lo_hi
    return [
        tuple(sorted(set(reachable_symbols(row.p_neural, row.p_lm, lo, hi)) | {int(n)}))
        for row, n in zip(working.itertuples(index=False), neural_only)
    ]


def summarize(sets: list[tuple[int, ...]], targets: np.ndarray, best_fixed_correct: float) -> dict:
    sizes = np.array([len(s) for s in sets])
    reachable_correct = np.array([int(t) in s for s, t in zip(sets, targets)])
    rate = float(reachable_correct.mean())
    return {
        "n_selections": int(len(sets)),
        "share_1_reachable": float((sizes == 1).mean()),
        "share_2_reachable": float((sizes == 2).mean()),
        "share_3plus_reachable": float((sizes >= 3).mean()),
        "mean_n_reachable": float(sizes.mean()),
        "reachable_correct_rate": rate,
        "best_fixed_fused_correct_rate": float(best_fixed_correct),
        "headroom": float(rate - best_fixed_correct),
    }


def run_prior(prior_model: str, range_names: list[str]) -> dict:
    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)
    best_fixed = select_best_fixed_beta_arm_cv(attribution, outcome="fused_correct")
    best_fixed_correct = float(best_fixed["fused_correct"].mean())
    targets = working["target_code"].to_numpy() - 1
    return {
        name: summarize(reachable_sets(working, RANGES[name]), targets, best_fixed_correct)
        for name in range_names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", default="all", help="comma-separated prior names, or 'all'")
    parser.add_argument("--ranges", default="all", help=f"comma-separated subset of {list(RANGES)}")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    priors = non_null_priors() if args.priors == "all" else args.priors.split(",")
    range_names = list(RANGES) if args.ranges == "all" else args.ranges.split(",")
    unknown = [r for r in range_names if r not in RANGES]
    if unknown:
        raise SystemExit(f"unknown range(s) {unknown}; choose from {list(RANGES)}")

    result: dict = {
        "ranges": {n: (list(RANGES[n]) if RANGES[n] else list(BETA_GRID)) for n in range_names},
        "priors": {},
    }
    for prior_model in priors:
        start = time.time()
        result["priors"][prior_model] = run_prior(prior_model, range_names)
        first = result["priors"][prior_model][range_names[0]]
        print(f"{prior_model:45s} {range_names[0]}: 1-symbol share {first['share_1_reachable']:.3f}, "
              f"reachable-correct {first['reachable_correct_rate']:.3f}, "
              f"headroom {first['headroom']:+.3f} ({time.time() - start:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
