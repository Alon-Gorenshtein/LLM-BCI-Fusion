"""Fair adaptive-versus-fixed comparison inside the decoded-history setting.

Every earlier comparison conditioned the language model on the true copy-spelling prefix. Here the
language model sees only the symbols the fused decoder itself emitted, so an error enters the next
context, as it would in closed-loop use. Both arms are tuned inside this setting: for each
participant-grouped fold, the best single beta and the best adaptive anchor are chosen from the
training participants' decoded-history replays and then applied to the held-out participants, who are
replayed once per arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fusion.arms import KEY_COLUMNS
from fusion.decoded_history import replay_decoded_history
from fusion.features import reliability_features_frame
from fusion.reliability import RANDOM_STATE
from fusion.rescue import ANCHOR_GRID, BETA_GRID, _grouped_folds, anchor_training_g_hat

_KEY = list(KEY_COLUMNS)


class CachedPrior:
    """Memoises prior(context) so repeated decoded contexts cost one language-model call."""

    def __init__(self, prior):
        self._prior = prior
        self._cache: dict[str, np.ndarray] = {}

    def prior(self, context: str):
        if context not in self._cache:
            self._cache[context] = self._prior.prior(context)
        return self._cache[context]


def _net_gain(replayed: pd.DataFrame) -> pd.Series:
    return replayed["fused_correct"].astype(int) - replayed["neural_correct"].astype(int)


def _anchor_betas(g_hat: pd.Series, beta_min: float, beta_max: float) -> pd.Series:
    g = np.clip(g_hat.to_numpy(), 1e-6, 1 - 1e-6)
    return pd.Series(beta_min * (beta_max / beta_min) ** g, index=g_hat.index)


def decoded_history_fair_comparison(
    working: pd.DataFrame,
    prior_object,
    n_folds: int = 5,
    inner_folds: int = 5,
    beta_grid: tuple = BETA_GRID,
    anchor_grid: tuple = ANCHOR_GRID,
    max_folds: int | None = None,
) -> dict:
    """`working` needs the key columns, p_neural, target_code, neural_correct; context_prefix is
    carried through when present. Returns {"selections": frame with fixed_net_gain and
    adaptive_net_gain per selection, "folds": chosen weight and anchor per fold}. `max_folds` stops
    after that many folds (timing runs only; the returned frame is then partial)."""
    working = working.reset_index(drop=True)
    cached = prior_object if isinstance(prior_object, CachedPrior) else CachedPrior(prior_object)
    pieces, folds = [], []
    for fold_number, (train_participants, test_participants) in enumerate(_grouped_folds(working, n_folds)):
        if max_folds is not None and fold_number >= max_folds:
            break
        train = working[working["participant_id"].isin(train_participants)]
        test = working[working["participant_id"].isin(test_participants)]

        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(reliability_features_frame(train["p_neural"]), train["neural_correct"].astype(int))
        g_test = pd.Series(
            model.predict_proba(reliability_features_frame(test["p_neural"]))[:, 1], index=test.index)
        g_train = pd.Series(anchor_training_g_hat(train, inner_folds), index=train.index)

        fixed_scores = {b: float(_net_gain(replay_decoded_history(train, cached, b)).mean()) for b in beta_grid}
        best_beta = max(fixed_scores, key=fixed_scores.get)
        anchor_scores = {
            a: float(_net_gain(replay_decoded_history(train, cached, _anchor_betas(g_train, *a))).mean())
            for a in anchor_grid
        }
        best_anchor = max(anchor_scores, key=anchor_scores.get)

        fixed_replay = replay_decoded_history(test, cached, best_beta)
        adaptive_replay = replay_decoded_history(test, cached, _anchor_betas(g_test, *best_anchor))
        piece = fixed_replay[_KEY].copy()
        piece["fixed_net_gain"] = _net_gain(fixed_replay).to_numpy()
        adaptive_piece = adaptive_replay[_KEY].copy()
        adaptive_piece["adaptive_net_gain"] = _net_gain(adaptive_replay).to_numpy()
        pieces.append(piece.merge(adaptive_piece, on=_KEY, how="inner", validate="one_to_one"))
        folds.append({
            "n_test_participants": len(test_participants),
            "fixed_beta": float(best_beta), "anchor": [float(best_anchor[0]), float(best_anchor[1])],
            "fixed_train_net_gain": fixed_scores[best_beta], "adaptive_train_net_gain": anchor_scores[best_anchor],
        })

    selections = pd.concat(pieces, ignore_index=True)
    if "context_prefix" in working.columns:
        selections = selections.merge(working[_KEY + ["context_prefix"]], on=_KEY, how="left", validate="one_to_one")
    if max_folds is None and len(selections) != len(working):
        raise ValueError(f"fair comparison produced {len(selections)} rows for {len(working)} selections")
    return {"selections": selections, "folds": folds}
