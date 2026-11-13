import numpy as np
import pandas as pd
import pytest

from fusion.candidate_arbitration import (
    apply_combining_rule,
    fit_candidate_ranker,
    fit_candidate_ranker_frozen,
    select_delta,
)


def _toy_features(n_participants=8, per_participant=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    trial = 0
    for p in range(n_participants):
        for _ in range(per_participant):
            for candidate_index, is_target in [(1, True), (7, False)]:
                rows.append({
                    "study": "StudyF", "participant_id": f"p{p}", "session_id": "s1",
                    "trial_number": trial, "relative_path": f"p{p}_{trial}.csv",
                    "candidate_index": candidate_index, "is_target": is_target,
                    "is_fixed_winner": candidate_index == 1,
                    "neural_p_candidate": rng.uniform(0, 1), "lm_p_candidate": rng.uniform(0, 1),
                    "boundary_distance": rng.normal(), "stability_interval_width": rng.uniform(0, 2),
                    "neural_lm_log_odds_candidate": rng.normal(),
                })
            trial += 1
    return pd.DataFrame(rows)


_FEATURE_COLUMNS = [
    "neural_p_candidate", "lm_p_candidate", "boundary_distance",
    "stability_interval_width", "neural_lm_log_odds_candidate",
]


def test_fit_candidate_ranker_returns_one_probability_per_row():
    features = _toy_features()
    p_hat = fit_candidate_ranker(features, n_folds=3)
    assert len(p_hat) == len(features)
    assert ((p_hat >= 0) & (p_hat <= 1)).all()


def test_fit_candidate_ranker_frozen_returns_one_probability_per_test_row():
    train = _toy_features(seed=0)
    test = _toy_features(n_participants=2, seed=1)
    p_hat = fit_candidate_ranker_frozen(train, test)
    assert len(p_hat) == len(test)


def test_apply_combining_rule_defaults_to_fixed_winner_below_delta():
    features = pd.DataFrame([
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 1, "is_fixed_winner": True},
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 7, "is_fixed_winner": False},
    ])
    p_hat = pd.Series([0.5, 0.55], index=features.index)  # alternative only slightly higher
    result = apply_combining_rule(features, p_hat, delta=0.2)
    assert result.iloc[0]["chosen_candidate"] == 1  # fixed winner kept, margin too small
    assert result.iloc[0]["overridden"] == False
    assert result.iloc[0]["p_hat_fixed_winner"] == 0.5
    assert result.iloc[0]["p_hat_best_alternative"] == 0.55


def test_apply_combining_rule_overrides_when_margin_clears_delta():
    features = pd.DataFrame([
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 1, "is_fixed_winner": True},
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 7, "is_fixed_winner": False},
    ])
    p_hat = pd.Series([0.3, 0.8], index=features.index)  # large margin
    result = apply_combining_rule(features, p_hat, delta=0.2)
    assert result.iloc[0]["chosen_candidate"] == 7
    assert result.iloc[0]["overridden"] == True
    assert result.iloc[0]["p_hat_fixed_winner"] == 0.3
    assert result.iloc[0]["p_hat_best_alternative"] == 0.8


def test_select_delta_picks_the_delta_that_maximizes_chosen_candidate_correctness():
    features = pd.DataFrame([
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 1, "is_target": False,
         "is_fixed_winner": True},
        {"study": "StudyF", "participant_id": "p1", "session_id": "s1", "trial_number": 1,
         "relative_path": "t1.csv", "candidate_index": 7, "is_target": True,
         "is_fixed_winner": False},
    ])
    # Margin is 0.9 - 0.1 = 0.8. delta=0.0 and delta=0.5 both clear it (override to the correct
    # candidate 7, score 1.0); delta=0.9 does not (stays on the wrong fixed winner, score 0.0).
    # The loop keeps the FIRST delta achieving the max score, so 0.0 must win deterministically.
    p_hat = pd.Series([0.1, 0.9], index=features.index)
    best_delta = select_delta(features, p_hat, delta_grid=(0.0, 0.5, 0.9))
    assert best_delta == 0.0
