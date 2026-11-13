"""Fair, matched-search-space comparison for every non-null prior, with nested anchor selection.

Per prior: a best-fixed-weight arm (single grid beta chosen out of fold) is compared with a
tuned-adaptive arm (anchor chosen out of fold) on net correct gain per selection. The adaptive arm is
run twice: with the anchor chosen on in-sample reliability estimates (the earlier procedure) and with
the anchor chosen on out-of-fold estimates from inner participant-grouped folds (nested). All
differences are adaptive minus fixed, so a positive value favours adaptive.

Intervals use three resampling schemes on the same paired per-selection differences: participants,
linguistic contexts, and both crossed. A simultaneous band and a pooled mean across priors come from
the same resamples, so the dependence between priors scored on the same selections is kept.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.clustering import bootstrap_means, equivalence_counts, percentile_ci, simultaneous_band
from fusion.data import AUTHORSHIP_ROOT, build_working_frame, load_attribution
from fusion.net_gain import add_net_correct_gain
from fusion.rescue import _grouped_folds, select_best_fixed_beta_arm_cv, tune_adaptive_anchor_cv

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "revision"
CACHE = OUT_DIR / "fair_ladder_cache"
KEY = list(KEY_COLUMNS)
MARGINS = (0.005, 0.010, 0.020)
SEED = 20260820


def non_null_priors() -> list[str]:
    frame = pd.read_parquet(AUTHORSHIP_ROOT / "output" / "intermediate" / "attribution.parquet",
                            columns=["prior_model"])
    return sorted(m for m in frame["prior_model"].unique() if m != "uniform")


def _fold_choices(working, fixed_arm, adaptive_arm):
    out = []
    for train, test in _grouped_folds(working, 5):
        fx = fixed_arm[fixed_arm["participant_id"].isin(test)]
        ad = adaptive_arm[adaptive_arm["participant_id"].isin(test)]
        out.append({
            "n_test_participants": len(test),
            "fixed_beta": float(fx["chosen_beta"].iloc[0]),
            "anchor": [float(ad["chosen_beta_min"].iloc[0]), float(ad["chosen_beta_max"].iloc[0])],
        })
    return out


def run_prior(prior: str) -> dict:
    path = CACHE / (prior.replace("/", "_") + ".pkl")
    if path.exists():
        return pickle.loads(path.read_bytes())
    working = build_working_frame(prior)
    attribution = load_attribution(prior)
    fixed = add_net_correct_gain(select_best_fixed_beta_arm_cv(attribution, outcome="fused_correct"))
    ad_in = add_net_correct_gain(tune_adaptive_anchor_cv(working, outcome="fused_correct"))
    ad_ne = add_net_correct_gain(tune_adaptive_anchor_cv(working, outcome="fused_correct", inner_folds=5))

    frame = working[KEY + ["context_prefix"]].copy()
    for name, arm in (("fixed", fixed), ("adaptive_insample", ad_in), ("adaptive_nested", ad_ne)):
        piece = arm[KEY + ["net_correct_gain"]].rename(columns={"net_correct_gain": name})
        frame = frame.merge(piece, on=KEY, how="left", validate="one_to_one")
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if frame[["fixed", "adaptive_insample", "adaptive_nested"]].isna().any().any():
        raise ValueError(f"{prior}: arms do not cover the same selections")

    folds_in = _fold_choices(working, fixed, ad_in)
    folds_ne = _fold_choices(working, fixed, ad_ne)
    record = {
        "prior": prior,
        "parameters": float(attribution["prior_parameters"].iloc[0]),
        "frame": frame,
        "folds_insample": folds_in,
        "folds_nested": folds_ne,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(record))
    return record


def summarize(records: list[dict], n_boot: int) -> dict:
    base = records[0]["frame"]
    for r in records[1:]:
        if not r["frame"][KEY].equals(base[KEY]):
            raise ValueError("selection order differs between priors")
    participants = base["participant_id"].to_numpy()
    contexts = base["context_prefix"].to_numpy()
    names = [r["prior"] for r in records]
    k = len(records)
    d_in = np.column_stack([r["frame"]["adaptive_insample"] - r["frame"]["fixed"] for r in records])
    d_ne = np.column_stack([r["frame"]["adaptive_nested"] - r["frame"]["fixed"] for r in records])
    both = np.hstack([d_in, d_ne])
    point = both.mean(axis=0)

    boots = {s: bootstrap_means(both, participants, contexts, scheme=s, n_boot=n_boot, seed=SEED)
             for s in ("participant", "context", "crossed")}
    ci = {s: percentile_ci(b) for s, b in boots.items()}

    result = {"n_selections": int(len(base)), "n_priors": k, "n_boot": n_boot, "seed": SEED,
              "orientation": "adaptive minus fixed; positive favours adaptive", "priors": {}, "summary": {}}
    for j, rec in enumerate(records):
        entry = {"parameters": rec["parameters"],
                 "mean_fixed": float(rec["frame"]["fixed"].mean()),
                 "folds_insample": rec["folds_insample"], "folds_nested": rec["folds_nested"]}
        for label, off in (("insample", 0), ("nested", k)):
            col = rec["frame"]["adaptive_" + label]
            diff = (col - rec["frame"]["fixed"]).to_numpy()
            idx = off + j
            entry[label] = {
                "mean_adaptive": float(col.mean()),
                "diff": float(point[idx]),
                "ci_participant": [float(ci["participant"][0][idx]), float(ci["participant"][1][idx])],
                "ci_context": [float(ci["context"][0][idx]), float(ci["context"][1][idx])],
                "ci_crossed": [float(ci["crossed"][0][idx]), float(ci["crossed"][1][idx])],
                "n_adaptive_better": int((diff > 0).sum()),
                "n_fixed_better": int((diff < 0).sum()),
                "identical_on_every_selection": bool((diff == 0).all()),
                "share_folds_constant_weight": float(np.mean(
                    [f["anchor"][0] == f["anchor"][1] for f in rec["folds_" + label]])),
            }
        entry["share_folds_same_anchor_insample_vs_nested"] = float(np.mean(
            [a["anchor"] == b["anchor"] for a, b in zip(rec["folds_insample"], rec["folds_nested"])]))
        result["priors"][rec["prior"]] = entry

    for label, off in (("insample", 0), ("nested", k)):
        sl = slice(off, off + k)
        lo, hi, c = simultaneous_band(boots["crossed"][:, sl], point[sl])
        pooled = boots["crossed"][:, sl].mean(axis=1)
        summary = {
            "simultaneous_multiplier": c,
            "simultaneous_band": {n: [float(lo[i]), float(hi[i])] for i, n in enumerate(names)},
            "simultaneous_band_excludes_zero": [n for i, n in enumerate(names) if lo[i] > 0 or hi[i] < 0],
            "pooled_mean_diff": float(point[sl].mean()),
            "pooled_mean_diff_ci_crossed": [float(x) for x in np.percentile(pooled, [2.5, 97.5])],
            "counts": {},
        }
        for scheme in ("participant", "crossed"):
            lower, upper = ci[scheme][0][sl], ci[scheme][1][sl]
            ties = [bool(result["priors"][n][label]["identical_on_every_selection"]) for n in names]
            summary["counts"][scheme] = {
                "identical_on_every_selection": int(sum(ties)),
                "ci_favors_adaptive": int(sum(1 for i in range(k) if not ties[i] and lower[i] > 0)),
                "ci_favors_fixed": int(sum(1 for i in range(k) if not ties[i] and upper[i] < 0)),
                "ci_crosses_zero": int(sum(1 for i in range(k) if not ties[i] and lower[i] <= 0 <= upper[i])),
                "equivalence": {f"{m:.3f}": equivalence_counts(lower, upper, m) for m in MARGINS},
            }
        result["summary"][label] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", default="all")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default=str(OUT_DIR / "fair_ladder.json"))
    args = parser.parse_args()
    priors = non_null_priors() if args.priors == "all" else args.priors.split(",")
    records = []
    for prior in priors:
        start = time.time()
        records.append(run_prior(prior))
        print(f"{prior:45s} done in {time.time() - start:5.0f}s", flush=True)
    result = summarize(records, args.n_boot)
    Path(args.out).write_text(json.dumps(result, indent=2))
    arrays = {r["prior"].replace("/", "_") + "__" + c: r["frame"][c].to_numpy()
              for r in records for c in ("fixed", "adaptive_insample", "adaptive_nested")}
    np.savez_compressed(Path(args.out).with_name("fair_ladder_arrays.npz"), **arrays,
                        participants=records[0]["frame"]["participant_id"].to_numpy(),
                        contexts=records[0]["frame"]["context_prefix"].to_numpy())
    lines = [f"{'prior':45s} {'fixed':>7s} {'adaptive(nested)':>17s} {'diff':>8s} {'crossed 95% CI':>20s}  better/worse"]
    for name, e in result["priors"].items():
        n = e["nested"]
        lines.append(f"{name:45s} {e['mean_fixed']:7.4f} {n['mean_adaptive']:17.4f} {n['diff']:+8.4f} "
                     f"[{n['ci_crossed'][0]:+.4f}, {n['ci_crossed'][1]:+.4f}]  {n['n_adaptive_better']}/{n['n_fixed_better']}")
    Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(result["summary"]["nested"]["counts"], indent=1))


if __name__ == "__main__":
    main()
