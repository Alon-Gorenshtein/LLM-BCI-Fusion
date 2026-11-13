"""Does the oracle-structure AUROC (0.79/0.81,
scripts/analyze_oracle_structure.py, participant-grouped) survive grouping by linguistic context
instead of participant? Repeated contexts crossing participant folds could inflate the
participant-grouped estimate if the same context's near-identical LM posterior appears in both
train and test. This check groups cross-validation folds by context_prefix instead, so no context
appears in both sides of a fold. Dev-only (Studies B, F, L): Study N is split out and discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import build_working_frame, load_attribution
from fusion.holdout import split_dev_test
from fusion.arms import select_oracle_arm, KEY_COLUMNS
from fusion.joint_features import joint_features_frame
from fusion.reliability import RANDOM_STATE


def oracle_structure_auroc_context_grouped(prior_model: str, n_folds: int = 5) -> dict:
    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)

    dev_working, _test_working = split_dev_test(working)
    dev_attribution, _test_attribution = split_dev_test(attribution)

    oracle = select_oracle_arm(dev_attribution)
    features = joint_features_frame(dev_working["p_neural"], dev_working["p_lm"])
    features[list(KEY_COLUMNS)] = dev_working[list(KEY_COLUMNS)].reset_index(drop=True)
    features["context_prefix"] = dev_working["context_prefix"].reset_index(drop=True)

    merged = oracle[list(KEY_COLUMNS) + ["beta"]].merge(features, on=list(KEY_COLUMNS), how="inner")
    feature_cols = [c for c in features.columns if c not in KEY_COLUMNS and c != "context_prefix"]
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = merged["beta"].to_numpy()
    groups = merged["context_prefix"].to_numpy()

    # See fusion analyze_oracle_structure.py: sklearn's type_of_target treats a float array with
    # non-integer values (the 0.25/0.5 entries in the beta grid) as "continuous" rather than
    # "multiclass", which makes LogisticRegression.fit and label_binarize both raise on the real
    # beta grid. Encode the discrete beta values as integer class codes for the classification
    # machinery; `classes` (the original float beta values) is kept only for reporting.
    classes = np.unique(y)
    class_index = {v: i for i, v in enumerate(classes)}
    y_codes = np.array([class_index[v] for v in y])

    n_groups = len(np.unique(groups))
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds {n_groups} unique contexts")

    splitter = GroupKFold(n_splits=n_folds)
    oof_scores = np.full((len(merged), len(classes)), np.nan)
    for train_idx, test_idx in splitter.split(X, y_codes, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("group leakage across a context-grouped CV fold")
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
        model.fit(X[train_idx], y_codes[train_idx])
        fold_classes = model.classes_
        proba = model.predict_proba(X[test_idx])
        for j, cls in enumerate(fold_classes):
            oof_scores[test_idx, cls] = proba[:, j]

    y_bin = label_binarize(y_codes, classes=np.arange(len(classes)))
    per_class_auroc = []
    for j in range(len(classes)):
        if y_bin[:, j].sum() == 0 or y_bin[:, j].sum() == len(y_bin):
            continue
        per_class_auroc.append(roc_auc_score(y_bin[:, j], oof_scores[:, j]))

    return {
        "prior_model": prior_model,
        "grouping": "context_prefix",
        "macro_ovr_auroc": float(np.mean(per_class_auroc)),
        "n_selections": len(merged),
        "n_unique_contexts": int(n_groups),
    }


if __name__ == "__main__":
    for prior in ["gpt2", "ngram5_kn"]:
        result = oracle_structure_auroc_context_grouped(prior)
        print(
            f"{prior}: context-grouped macro OvR AUROC = {result['macro_ovr_auroc']:.3f} "
            f"({result['n_selections']} selections, {result['n_unique_contexts']} unique contexts)"
        )
