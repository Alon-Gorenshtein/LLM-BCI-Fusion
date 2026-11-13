import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from fusion.features import reliability_features_frame
from fusion.reliability import RANDOM_STATE
from fusion.rescue import anchor_training_g_hat


def _frame(n_participants=12, rows=20, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for p in range(n_participants):
        for _ in range(rows):
            v = rng.dirichlet(np.full(36, 0.3))
            recs.append({"participant_id": f"P{p}", "p_neural": v,
                         "neural_correct": bool(rng.random() < v.max())})
    return pd.DataFrame(recs)


def test_in_sample_default_matches_a_single_fit():
    frame = _frame()
    feats = reliability_features_frame(frame["p_neural"])
    labels = frame["neural_correct"].astype(int).to_numpy()
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(feats, labels)
    assert np.allclose(anchor_training_g_hat(frame, None), model.predict_proba(feats)[:, 1])


def test_inner_cv_scores_every_row_with_a_model_that_never_saw_it():
    frame = _frame()
    g = anchor_training_g_hat(frame, 4)
    feats = reliability_features_frame(frame["p_neural"])
    labels = frame["neural_correct"].astype(int).to_numpy()
    groups = frame["participant_id"].to_numpy()
    expected = np.full(len(frame), np.nan)
    for tr, te in GroupKFold(n_splits=4).split(feats, labels, groups):
        assert not set(groups[tr]) & set(groups[te])
        m = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(feats.iloc[tr], labels[tr])
        expected[te] = m.predict_proba(feats.iloc[te])[:, 1]
    assert np.allclose(g, expected)
    assert not np.allclose(g, anchor_training_g_hat(frame, None))
