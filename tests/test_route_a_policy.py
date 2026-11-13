import numpy as np
import pandas as pd

from fusion.rescue import BETA_GRID
from fusion.route_a_policy import combine_route_a_policy, route_a_dev_wide_best_fixed_beta


def _toy_attribution_for_best_fixed():
    # beta=2.0 has the highest mean fused_correct (3/3), so best_fixed_beta must be 2.0.
    rows = []
    for beta in BETA_GRID:
        correct = beta == 2.0
        for i in range(3):
            rows.append({"beta": beta, "fused_correct": correct})
    return pd.DataFrame(rows)


def test_route_a_dev_wide_best_fixed_beta_picks_the_best_grid_value():
    assert route_a_dev_wide_best_fixed_beta(_toy_attribution_for_best_fixed()) == 2.0


def _key_row(trial):
    return dict(study="StudyF", participant_id="p1", session_id="s1", trial_number=trial,
                relative_path=f"t{trial}.csv")


def test_combine_route_a_policy_defers_to_fixed_when_not_actionable():
    dev_frame = pd.DataFrame([_key_row(1)])
    p_actionable = pd.Series([0.1], index=dev_frame.index)
    q_hat_long = pd.DataFrame([
        {**_key_row(1), "beta": beta, "q_hat": 0.9 if beta == 4.0 else 0.1} for beta in BETA_GRID
    ])
    result = combine_route_a_policy(dev_frame, p_actionable, q_hat_long, best_fixed_beta=1.0, tau=0.5)
    assert result["beta_hat"].iloc[0] == 1.0  # p_actionable=0.1 < tau=0.5 -> defer to fixed


def test_combine_route_a_policy_uses_argmax_q_hat_when_actionable():
    dev_frame = pd.DataFrame([_key_row(1)])
    p_actionable = pd.Series([0.9], index=dev_frame.index)
    q_hat_long = pd.DataFrame([
        {**_key_row(1), "beta": beta, "q_hat": 0.9 if beta == 4.0 else 0.1} for beta in BETA_GRID
    ])
    result = combine_route_a_policy(dev_frame, p_actionable, q_hat_long, best_fixed_beta=1.0, tau=0.5)
    assert result["beta_hat"].iloc[0] == 4.0  # p_actionable=0.9 >= tau -> argmax q_hat is beta=4.0


def test_run_route_a_policy_selection_returns_internally_consistent_result():
    import numpy as np
    import pandas as pd
    from fusion.route_a_policy import TAU_GRID, run_route_a_policy_selection

    rng = np.random.default_rng(42)
    rows = []
    for p in range(8):
        for t in range(15):
            rows.append({
                "study": "StudyF", "participant_id": f"p{p}", "session_id": "s1",
                "trial_number": t, "relative_path": f"p{p}_{t}.csv",
                "p_neural": rng.dirichlet(np.ones(36)), "p_lm": rng.dirichlet(np.ones(36)),
            })
    dev_frame = pd.DataFrame(rows)

    from fusion.rescue import BETA_GRID
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    attribution_rows = []
    actionability_rows = []
    correctness_rows = []
    for _, row in dev_frame[key_cols].iterrows():
        # NOTE: differs from the simplest construction, which drew argmaxes ~ Uniform(0, 36) over 5 betas
        # and set actionable = (argmaxes not all equal) -- with 36 categories that condition is
        # true with probability ~= 1 - 36**-4, so every row in the fixture came out actionable=True
        # and fit_actionability_model's per-fold logistic regression raised sklearn's "needs samples
        # of at least 2 classes" ValueError regardless of this task's implementation. Alternating on
        # trial parity gives an exact, GroupKFold-robust 50/50 label split instead.
        actionable = row["trial_number"] % 2 == 0
        actionability_rows.append({**row.to_dict(), "actionable": bool(actionable)})
        for beta in BETA_GRID:
            correct = bool(rng.integers(0, 2))
            attribution_rows.append({**row.to_dict(), "beta": beta, "fused_correct": correct})
            correctness_rows.append({**row.to_dict(), "beta": beta, "fused_correct": correct})
    dev_attribution = pd.DataFrame(attribution_rows)
    dev_actionability = pd.DataFrame(actionability_rows)
    dev_beta_correctness = pd.DataFrame(correctness_rows)

    result = run_route_a_policy_selection(
        dev_frame, dev_attribution, dev_actionability, dev_beta_correctness, n_folds=2
    )
    assert result["tau"] in TAU_GRID
    assert result["best_fixed_beta"] in BETA_GRID
    assert result["n_dev_selections"] == len(dev_frame)
    assert abs(
        result["dev_gated_minus_best_fixed"]
        - (result["dev_gated_fused_correct"] - result["dev_best_fixed_fused_correct"])
    ) < 1e-9
