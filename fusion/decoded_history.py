"""Decoded-history replay robustness check: re-score an ALREADY-FROZEN
best-fixed-beta policy using the fusion arm's own decoded prefix within each phrase, instead of the
archive's ground-truth prefix. No new tuning, no new holdout -- the beta is already established
(gpt2=1.0, ngram5_kn=0.5, both from `fusion.route_a_policy.route_a_dev_wide_best_fixed_beta` on the
full four-study population).
"""
from __future__ import annotations

import sys

import pandas as pd

from fusion.arms import KEY_COLUMNS
from fusion.data import AUTHORSHIP_ROOT

if str(AUTHORSHIP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORSHIP_ROOT))
from authorship.attribution import attribution_row, fuse  # noqa: E402

_KEY = list(KEY_COLUMNS)
_REQUIRED = set(_KEY) | {"p_neural", "target_code"}


def build_prior_object(prior_model: str):
    """The live, re-queryable prior object for one of this study's two primary priors."""
    from authorship.priors import CharNgramKNPrior, TransformerPrior

    if prior_model == "gpt2":
        return TransformerPrior("gpt2")
    if prior_model == "ngram5_kn":
        return CharNgramKNPrior(order=5)
    raise ValueError(f"decoded-history replay only supports gpt2 or ngram5_kn, got {prior_model!r}")


def replay_decoded_history(selections: pd.DataFrame, prior_object, beta) -> pd.DataFrame:
    """Replay every phrase (grouped by relative_path, ordered by trial_number) using the DECODED
    prefix (the fused arm's own prior emissions within that phrase) instead of the ground-truth
    prefix. `prior_object` must expose `.prior(context: str) -> np.ndarray` (both TransformerPrior
    and CharNgramKNPrior already do). Returns KEY_COLUMNS + ["fused_correct", "neural_correct"],
    one row per input selection.
    """
    missing = _REQUIRED - set(selections.columns)
    if missing:
        raise ValueError(f"selections is missing required columns: {sorted(missing)}")

    if isinstance(beta, pd.Series):
        per_row_beta = beta.reindex(selections.index)
        if per_row_beta.isna().any():
            raise ValueError("per-row beta must cover every selection")
        selections = selections.assign(beta_row=per_row_beta.to_numpy())
    else:
        selections = selections.assign(beta_row=float(beta))

    ordered = selections.sort_values(["relative_path", "trial_number"]).reset_index(drop=True)
    rows = []
    for _, phrase in ordered.groupby("relative_path", sort=False):
        decoded_context = ""
        for row in phrase.itertuples(index=False):
            p_lm = prior_object.prior(decoded_context)
            fused = fuse(row.p_neural, p_lm, beta=row.beta_row)
            result = attribution_row(row.p_neural, p_lm, fused, row.target_code - 1)
            rows.append({
                **{column: getattr(row, column) for column in _KEY},
                "fused_correct": result["fused_correct"],
                "neural_correct": result["neural_correct"],
            })
            decoded_context += result["emitted_symbol"]
    out = pd.DataFrame(rows)
    if len(out) != len(selections):
        raise ValueError(
            f"decoded-history replay produced {len(out)} rows for {len(selections)} selections"
        )
    return out
