import pandas as pd

from fusion.candidates import achievable_candidates_by_key
from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID


def _key(trial):
    return dict(study="StudyF", participant_id="p1", session_id="s1", trial_number=trial,
                relative_path=f"t{trial}.csv")


def _toy_attribution():
    rows = []
    # Selection 1: neural_argmax=5, and every beta in the grid also fuses to argmax=5 -> 1 candidate.
    for beta in BETA_GRID:
        rows.append({**_key(1), "beta": beta, "fused_argmax": 5, "neural_argmax": 5})
    # Selection 2: neural_argmax=7, fused_argmax=7 at low betas, fused_argmax=9 at beta=4.0 only
    # -> 2 candidates {7, 9}.
    for beta in BETA_GRID:
        argmax = 9 if beta == 4.0 else 7
        rows.append({**_key(2), "beta": beta, "fused_argmax": argmax, "neural_argmax": 7})
    return pd.DataFrame(rows)


def test_achievable_candidates_counts_neural_plus_grid_fused_argmax():
    result = achievable_candidates_by_key(_toy_attribution())
    assert set(result.columns) == set(KEY_COLUMNS) | {"candidates", "n_distinct_candidates"}
    by_trial = result.set_index("trial_number")
    assert by_trial.loc[1, "candidates"] == (5,)
    assert by_trial.loc[1, "n_distinct_candidates"] == 1
    assert by_trial.loc[2, "candidates"] == (7, 9)
    assert by_trial.loc[2, "n_distinct_candidates"] == 2


def test_achievable_candidates_includes_neural_argmax_even_if_never_the_fused_winner():
    rows = []
    for beta in BETA_GRID:
        # fused_argmax is always 3, but neural_argmax (2) never wins at any grid beta -- it must
        # still appear as its own always-available candidate.
        rows.append({**_key(1), "beta": beta, "fused_argmax": 3, "neural_argmax": 2})
    result = achievable_candidates_by_key(pd.DataFrame(rows))
    assert result.iloc[0]["candidates"] == (2, 3)
    assert result.iloc[0]["n_distinct_candidates"] == 2
