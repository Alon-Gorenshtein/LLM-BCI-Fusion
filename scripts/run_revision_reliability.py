"""Reliability estimator against simpler baselines, and a flash-count summary.

The reliability estimator (three posterior-shape features through a logistic model) predicts whether
the neural argmax is correct. This script compares its out-of-fold discrimination with each single
feature used on its own, as a raw score and through its own one-feature logistic fit, using the same
participant-grouped folds. It also tabulates flashes per selection by source study, which bounds what
the net-correct-gain endpoint can say about stopping rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from fusion.data import build_working_frame, load_selections
from fusion.features import reliability_features_frame
from fusion.reliability import RANDOM_STATE, fit_reliability_cv, reliability_calibration_error

OUT = Path(__file__).resolve().parents[1] / "output" / "revision"
FEATURES = ("top1_prob", "margin", "entropy_bits")
N_RANDOM_FOLD_ASSIGNMENTS = 30
SEED_FOLDS = 20260821


def _oof_single_feature(values: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold probability from a one-feature logistic model, same grouped folds as the
    three-feature estimator."""
    x = values.reshape(-1, 1)
    out = np.full(len(values), np.nan)
    for train, test in GroupKFold(n_splits=n_folds).split(x, labels, groups):
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(x[train], labels[train])
        out[test] = model.predict_proba(x[test])[:, 1]
    return out


def _metrics(score: np.ndarray, labels: np.ndarray, probabilistic: bool) -> dict:
    out = {"auroc": float(roc_auc_score(labels, score))}
    if probabilistic:
        out["brier"] = float(brier_score_loss(labels, score))
        out["ece"] = reliability_calibration_error(
            pd.DataFrame({"neural_correct": labels.astype(bool), "g_hat": score}))
    return out


def run(n_boot: int) -> dict:
    frame = build_working_frame("gpt2").reset_index(drop=True)   # neural-only quantities, identical across priors
    fitted = fit_reliability_cv(frame, n_folds=5)
    feats = reliability_features_frame(frame["p_neural"])
    labels = frame["neural_correct"].astype(int).to_numpy()
    groups = frame["participant_id"].to_numpy()

    scores = {"three_feature_logistic": fitted["g_hat"].to_numpy()}
    probabilistic = {"three_feature_logistic": True}
    for name in FEATURES:
        sign = -1.0 if name == "entropy_bits" else 1.0
        scores[f"{name}_raw"] = sign * feats[name].to_numpy()
        probabilistic[f"{name}_raw"] = False
        scores[f"{name}_logistic"] = _oof_single_feature(feats[name].to_numpy(), labels, groups)
        probabilistic[f"{name}_logistic"] = True

    point = {k: _metrics(v, labels, probabilistic[k]) for k, v in scores.items()}

    rng = np.random.default_rng(20260820)
    uniq = np.unique(groups)
    rows = {p: np.flatnonzero(groups == p) for p in uniq}
    compare = [k for k in scores if k != "three_feature_logistic"]
    diffs = {k: np.empty(n_boot) for k in compare}
    base = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        a = roc_auc_score(labels[idx], scores["three_feature_logistic"][idx])
        base[b] = a
        for k in compare:
            diffs[k][b] = a - roc_auc_score(labels[idx], scores[k][idx])
    contrasts = {}
    for k in compare:
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        contrasts[k] = {
            "auroc_three_feature_minus_this": point["three_feature_logistic"]["auroc"] - point[k]["auroc"],
            "lower": float(lo), "upper": float(hi),
        }
    lo, hi = np.percentile(base, [2.5, 97.5])
    point["three_feature_logistic"]["auroc_lower"], point["three_feature_logistic"]["auroc_upper"] = float(lo), float(hi)

    # Sensitivity of the three-feature out-of-fold AUROC to which participants share a fold.
    X = feats.to_numpy()
    rng_folds = np.random.default_rng(SEED_FOLDS)
    random_fold_auroc = []
    for _ in range(N_RANDOM_FOLD_ASSIGNMENTS):
        permuted = rng_folds.permutation(uniq)
        fold_of = {p: k % 5 for k, p in enumerate(permuted)}
        assignment = np.array([fold_of[p] for p in groups])
        oof = np.full(len(labels), np.nan)
        for k in range(5):
            tr, te = assignment != k, assignment == k
            model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X[tr], labels[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        random_fold_auroc.append(float(roc_auc_score(labels, oof)))
    fold_sensitivity = {
        "n_assignments": N_RANDOM_FOLD_ASSIGNMENTS,
        "mean": float(np.mean(random_fold_auroc)), "min": float(np.min(random_fold_auroc)),
        "max": float(np.max(random_fold_auroc)),
    }

    selections = load_selections()
    flash = {}
    for study, sub in selections.groupby("study"):
        counts = sub["n_flashes"]
        flash[study] = {
            "n_selections": int(len(sub)),
            "min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max()),
            "n_distinct_values": int(counts.nunique()),
            "share_at_modal_value": float((counts == counts.mode().iloc[0]).mean()),
            "modal_value": int(counts.mode().iloc[0]),
        }
    # Total flashes per selection (artifact-free plus rejected) and how the offline neural posterior
    # compares with the archive's own online selections.
    sel = selections.assign(total_flashes=selections["n_flashes"] + selections["n_flashes_dropped"])
    pn = np.stack(sel["p_neural"].to_numpy())
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "] + list("123456789")
    neural_symbol = np.array([symbols[i] for i in pn.argmax(1)])
    online = {
        "neural_argmax_correct": float((pn.argmax(1) == sel["target_code"].to_numpy() - 1).mean()),
        "archive_online_correct": float(sel["archive_correct"].mean()),
        "neural_argmax_equals_archive_selection": float((neural_symbol == sel["archive_selected_symbol"].to_numpy()).mean()),
    }
    flash_total = {}
    for study, sub in sel.groupby("study"):
        t = sub["total_flashes"]
        flash_total[study] = {"min": int(t.min()), "median": float(t.median()), "max": int(t.max()),
                              "n_distinct_values": int(t.nunique()),
                              "conditions": {k: int(v) for k, v in sub["condition"].value_counts().items()}}
    return {"n_selections": int(len(frame)), "n_boot": n_boot, "metrics": point,
            "offline_versus_online_accuracy": online, "total_flashes_by_study": flash_total,
            "auroc_contrasts": contrasts, "three_feature_auroc_across_random_fold_assignments": fold_sensitivity,
            "flashes_by_study": flash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default=str(OUT / "reliability_baselines.json"))
    args = parser.parse_args()
    result = run(args.n_boot)
    Path(args.out).write_text(json.dumps(result, indent=2))
    lines = ["Out-of-fold discrimination of the neural argmax being correct:"]
    for name, m in result["metrics"].items():
        lines.append(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    lines.append("Three-feature AUROC minus each baseline (95% participant bootstrap interval):")
    for name, c in result["auroc_contrasts"].items():
        lines.append(f"  vs {name}: {c['auroc_three_feature_minus_this']:+.4f} ({c['lower']:+.4f} to {c['upper']:+.4f})")
    fs = result["three_feature_auroc_across_random_fold_assignments"]
    lines.append(f"Three-feature AUROC over {fs['n_assignments']} random participant-fold assignments: "
                 f"mean {fs['mean']:.4f}, min {fs['min']:.4f}, max {fs['max']:.4f}")
    lines.append("Flashes per selection by study:")
    for study, f in result["flashes_by_study"].items():
        lines.append(f"  {study}: {f}")
    Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
