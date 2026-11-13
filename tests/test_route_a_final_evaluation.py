import numpy as np
import pandas as pd
import pytest

from fusion.rescue import BETA_GRID
from fusion.route_a_final_evaluation import evaluate_route_a_frozen_policy


def _toy_prior_frame(studies_participants, per_participant=8, seed=0):
    """studies_participants: list of (study, [participant_ids])."""
    rng = np.random.default_rng(seed)
    rows = []
    trial = 0
    for study, participants in studies_participants:
        for pid in participants:
            for t in range(per_participant):
                rows.append({
                    "study": study, "participant_id": pid, "session_id": "s1",
                    "trial_number": trial, "relative_path": f"{pid}_{trial}.csv",
                    "p_neural": rng.dirichlet(np.ones(36)), "p_lm": rng.dirichlet(np.ones(36)),
                })
                trial += 1
    return pd.DataFrame(rows)


def _toy_attribution(frame, seed=1):
    # NOTE: differs from the simplest construction, which drew all five per-selection argmaxes ~
    # Uniform(0, 36) -- with 36 categories that makes fused_argmax vary across the beta grid (i.e.
    # actionable=True) with probability ~= 1 - 36**-4 for every selection, so dev_actionability
    # comes out single-class and fit_actionability_model's per-fold logistic regression raises
    # sklearn's "needs samples of at least 2 classes" ValueError regardless of the
    # implementation. tests/test_route_a_stage1.py hit the identical issue and fixed it
    # the same way: alternate on trial parity for an exact, GroupKFold-robust 50/50 label split
    # (even trials get one fixed argmax repeated across all betas -> actionable=False; odd trials
    # keep the original random-per-beta draw -> actionable=True with the same ~1 probability).
    rng = np.random.default_rng(seed)
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    rows = []
    for _, row in frame[key_cols].iterrows():
        if row["trial_number"] % 2 == 0:
            argmaxes = np.full(len(BETA_GRID), rng.integers(0, 36))
        else:
            argmaxes = rng.integers(0, 36, size=len(BETA_GRID))
        for beta, argmax in zip(BETA_GRID, argmaxes):
            rows.append({
                **row.to_dict(), "beta": beta, "fused_argmax": int(argmax),
                "fused_correct": bool(rng.integers(0, 2)), "neural_correct": bool(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def test_evaluate_route_a_frozen_policy_returns_expected_shape():
    dev_frame = _toy_prior_frame([("StudyF", ["f1", "f2", "f3"]), ("StudyL", ["l1", "l2", "l3"])])
    test_frame = _toy_prior_frame([("StudyB", ["b1", "b2"])], seed=2)
    dev_attribution = _toy_attribution(dev_frame)
    test_attribution = _toy_attribution(test_frame, seed=3)

    result = evaluate_route_a_frozen_policy(
        dev_frame, dev_attribution, test_frame, test_attribution, n_folds=2
    )
    assert "policy" in result
    assert "route_a_net_correct_gain" in result
    assert "best_fixed_net_correct_gain" in result
    assert "route_a_minus_best_fixed" in result
    point, lower, upper = (
        result["route_a_minus_best_fixed"]["point"],
        result["route_a_minus_best_fixed"]["lower"],
        result["route_a_minus_best_fixed"]["upper"],
    )
    assert lower <= point <= upper
    assert result["n_holdout_selections"] == len(test_frame)


def test_evaluate_route_a_frozen_policy_raises_when_dev_contains_holdout_study():
    dev_frame = _toy_prior_frame([
        ("StudyF", ["f1", "f2", "f3"]), ("StudyL", ["l1", "l2", "l3"]), ("StudyB", ["b1"]),
    ])
    test_frame = _toy_prior_frame([("StudyB", ["b1", "b2"])], seed=2)
    dev_attribution = _toy_attribution(dev_frame)
    test_attribution = _toy_attribution(test_frame, seed=3)

    with pytest.raises(ValueError, match="StudyB"):
        evaluate_route_a_frozen_policy(
            dev_frame, dev_attribution, test_frame, test_attribution, n_folds=2
        )


def test_evaluate_route_a_frozen_policy_raises_when_dev_frame_has_no_study_column():
    dev_frame = _toy_prior_frame([("StudyF", ["f1", "f2", "f3"]), ("StudyL", ["l1", "l2", "l3"])])
    test_frame = _toy_prior_frame([("StudyB", ["b1", "b2"])], seed=2)
    dev_attribution = _toy_attribution(dev_frame)
    test_attribution = _toy_attribution(test_frame, seed=3)
    dev_frame_no_study = dev_frame.drop(columns=["study"])

    with pytest.raises(ValueError, match="study"):
        evaluate_route_a_frozen_policy(
            dev_frame_no_study, dev_attribution, test_frame, test_attribution, n_folds=2
        )
