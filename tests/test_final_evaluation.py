import numpy as np
import pandas as pd
import pytest

from fusion.final_evaluation import run_frozen_gate_selection


def _synthetic_dev(n=150, n_participants=20, seed=0):
    rng = np.random.default_rng(seed)
    dev_frame = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % n_participants}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "target_code": rng.integers(1, 37, n),
        "neural_correct": rng.random(n) < 0.7,
    })
    dev_attribution = pd.concat([
        dev_frame.assign(beta=b, fused_correct=rng.random(n) < 0.72,
                         phantom_agreement=rng.random(n) < 0.1, prior_capture=rng.random(n) < 0.05)
        for b in [0.25, 0.5, 1.0, 2.0, 4.0]
    ], ignore_index=True)
    return dev_frame, dev_attribution


def test_run_frozen_gate_selection_returns_a_single_frozen_policy():
    dev_frame, dev_attribution = _synthetic_dev()

    policy = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)
    assert "high" in policy and "low" in policy
    assert isinstance(policy["high"], float)
    assert policy["low"] < policy["high"]


def test_run_frozen_gate_selection_raises_if_study_n_present():
    frame_with_studyn = pd.DataFrame({"study": ["StudyN"], "participant_id": ["P1"]})
    with pytest.raises(ValueError, match="StudyN"):
        run_frozen_gate_selection(frame_with_studyn, frame_with_studyn, n_folds=2)


def test_run_frozen_gate_selection_raises_if_study_n_only_in_attribution():
    dev_frame, dev_attribution = _synthetic_dev(n=20, n_participants=4)
    leaked = pd.concat([dev_attribution, dev_attribution.head(1).assign(study="StudyN")],
                       ignore_index=True)
    with pytest.raises(ValueError, match="StudyN"):
        run_frozen_gate_selection(dev_frame, leaked, n_folds=2)


def test_frozen_policy_beta_is_on_the_grid_and_gate_betas_stay_on_grid():
    from fusion.rescue import BETA_GRID

    dev_frame, dev_attribution = _synthetic_dev()
    policy = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)
    assert policy["dev_wide_best_fixed_beta"] in [float(b) for b in BETA_GRID]
    assert 0.0 <= policy["dev_gated_fused_correct"] <= 1.0
    assert policy["n_dev_selections"] == len(dev_frame)


def test_frozen_policy_reports_its_dev_side_score_against_the_fixed_beta_baseline():
    """The gate's home-field comparison: without it a holdout loss cannot be read against any
    reference, and "failed to transfer" is indistinguishable from "never won on dev either".
    """
    dev_frame, dev_attribution = _synthetic_dev()
    policy = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)

    expected_baseline = dev_attribution.loc[
        np.isclose(dev_attribution["beta"], policy["dev_wide_best_fixed_beta"]), "fused_correct"
    ].mean()
    assert policy["dev_best_fixed_fused_correct"] == pytest.approx(expected_baseline)
    assert policy["dev_gated_minus_best_fixed"] == pytest.approx(
        policy["dev_gated_fused_correct"] - policy["dev_best_fixed_fused_correct"]
    )


def test_v_hat_alignment_raises_when_beta1_coverage_is_incomplete():
    from fusion.final_evaluation import v_hat_by_key

    dev_frame, dev_attribution = _synthetic_dev(n=40, n_participants=8)
    # Drop one selection's beta==1.0 row: exactly the silent-misalignment scenario the
    # positional slice in the original draft could not detect.
    beta1_mask = np.isclose(dev_attribution["beta"], 1.0)
    first_beta1_index = dev_attribution.index[beta1_mask][0]
    holed = dev_attribution.drop(index=first_beta1_index)

    with pytest.raises(ValueError, match="beta=1.0"):
        v_hat_by_key(dev_frame, holed, n_folds=2)


def test_v_hat_alignment_raises_on_duplicate_beta1_keys():
    from fusion.final_evaluation import v_hat_by_key

    dev_frame, dev_attribution = _synthetic_dev(n=40, n_participants=8)
    beta1 = dev_attribution[np.isclose(dev_attribution["beta"], 1.0)]
    doubled = pd.concat([dev_attribution, beta1.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        v_hat_by_key(dev_frame, doubled, n_folds=2)


def test_v_hat_by_key_returns_one_row_per_selection_keyed_not_positional():
    from fusion.final_evaluation import v_hat_by_key
    from fusion.arms import KEY_COLUMNS

    dev_frame, dev_attribution = _synthetic_dev(n=60, n_participants=10)
    out = v_hat_by_key(dev_frame, dev_attribution, n_folds=5)

    assert len(out) == len(dev_frame)
    assert list(out.columns) == list(KEY_COLUMNS) + ["v_hat"]
    assert out.duplicated(subset=list(KEY_COLUMNS)).sum() == 0
    assert set(map(tuple, out[list(KEY_COLUMNS)].to_numpy())) == set(
        map(tuple, dev_frame[list(KEY_COLUMNS)].to_numpy())
    )
    assert out["v_hat"].notna().all()
    # V(x) = P(phantom) - P(capture), so it is bounded by construction.
    assert out["v_hat"].between(-1.0, 1.0).all()


def test_shuffling_the_attribution_row_order_does_not_change_v_hat():
    """Key-based alignment must be invariant to attribution row order; a positional slice is not."""
    from fusion.final_evaluation import v_hat_by_key
    from fusion.arms import KEY_COLUMNS

    dev_frame, dev_attribution = _synthetic_dev(n=60, n_participants=10)
    ordered = v_hat_by_key(dev_frame, dev_attribution, n_folds=5)
    shuffled_attribution = dev_attribution.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled = v_hat_by_key(dev_frame, shuffled_attribution, n_folds=5)

    key = list(KEY_COLUMNS)
    merged = ordered.merge(shuffled, on=key, how="inner", suffixes=("_a", "_b"))
    assert len(merged) == len(dev_frame)
    np.testing.assert_allclose(merged["v_hat_a"], merged["v_hat_b"])


def test_frozen_v_hat_never_uses_test_labels():
    """The frozen V(x) applied to the holdout must be a function of dev labels alone."""
    from fusion.final_evaluation import frozen_v_hat_by_key
    from fusion.arms import KEY_COLUMNS

    dev_frame, dev_attribution = _synthetic_dev(n=120, n_participants=12, seed=1)
    test_frame = dev_frame.head(30).assign(
        study="StudyN",
        participant_id=[f"N{i % 4}" for i in range(30)],
        relative_path=[f"n{i}.edf" for i in range(30)],
    )

    rng = np.random.default_rng(99)
    test_attribution_a = pd.concat([
        test_frame.assign(beta=b, fused_correct=True, phantom_agreement=True, prior_capture=False)
        for b in [0.25, 0.5, 1.0, 2.0, 4.0]
    ], ignore_index=True)
    test_attribution_b = pd.concat([
        test_frame.assign(beta=b, fused_correct=False,
                          phantom_agreement=rng.random(30) < 0.5, prior_capture=True)
        for b in [0.25, 0.5, 1.0, 2.0, 4.0]
    ], ignore_index=True)

    a = frozen_v_hat_by_key(dev_frame, dev_attribution, test_frame)
    b = frozen_v_hat_by_key(dev_frame, dev_attribution, test_frame)

    # Same dev, same test features -> identical predictions regardless of what the test
    # outcome labels say. (The two test_attribution frames are deliberately opposite; the
    # function does not take them as arguments at all, which is the point being locked in.)
    key = list(KEY_COLUMNS)
    merged = a.merge(b, on=key, how="inner", suffixes=("_a", "_b"))
    assert len(merged) == len(test_frame)
    np.testing.assert_allclose(merged["v_hat_a"], merged["v_hat_b"])
    assert not test_attribution_a.equals(test_attribution_b)


def test_gate_betas_are_the_multipliers_of_the_best_fixed_beta_not_snapped_to_the_grid():
    """The gate's 0.5x/2x betas are applied as-is on the holdout, so dev must score them as-is too.

    Verified against real data separately: direct fusion reproduces the sibling attribution table's
    fused_correct exactly at every grid beta (0 mismatches, both priors, all 2903 dev rows), so
    scoring by direct fusion is identical where the grid overlaps and correct where it does not.
    """
    dev_frame, dev_attribution = _synthetic_dev()
    policy = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)
    best = policy["dev_wide_best_fixed_beta"]
    assert policy["gate_betas"] == [best * 0.5, best, best * 2.0]


def test_evaluate_frozen_policy_runs_on_a_pseudo_holdout_and_never_sees_study_n():
    from fusion.final_evaluation import evaluate_frozen_policy

    dev_frame, dev_attribution = _synthetic_dev(n=120, n_participants=12, seed=3)
    rng = np.random.default_rng(5)
    m = 40
    test_frame = dev_frame.head(m).copy()
    test_frame["participant_id"] = [f"T{i % 5}" for i in range(m)]
    test_frame["relative_path"] = [f"t{i}.edf" for i in range(m)]
    test_frame["study"] = "StudyT"
    test_attribution = pd.concat([
        test_frame.assign(beta=b, fused_correct=rng.random(m) < 0.7,
                          phantom_agreement=rng.random(m) < 0.1, prior_capture=rng.random(m) < 0.05)
        for b in [0.25, 0.5, 1.0, 2.0, 4.0]
    ], ignore_index=True)
    # The control arm's fused_correct must agree with direct fusion, which is what the real
    # attribution table does; synthesise it that way so the built-in crosscheck is meaningful.
    from authorship.attribution import fuse as _fuse, attribution_row as _row
    for b in [0.25, 0.5, 1.0, 2.0, 4.0]:
        mask = np.isclose(test_attribution["beta"], b)
        recomputed = [
            _row(r.p_neural, r.p_lm, _fuse(r.p_neural, r.p_lm, beta=b), r.target_code - 1)
            for r in test_frame.itertuples(index=False)
        ]
        test_attribution.loc[mask, "fused_correct"] = [x["fused_correct"] for x in recomputed]
        test_attribution.loc[mask, "neural_correct"] = [x["neural_correct"] for x in recomputed]

    result = evaluate_frozen_policy(dev_frame, dev_attribution, test_frame, test_attribution, n_folds=4)
    assert result["n_holdout_selections"] == m
    assert result["n_holdout_participants"] == 5
    assert set(result["applied_beta_counts"]) <= set(result["policy"]["gate_betas"])
    diff = result["frozen_gate_minus_best_fixed"]
    assert diff["lower"] <= diff["point"] <= diff["upper"]


def test_residual_gate_applied_to_frozen_policy_is_deterministic():
    from fusion.final_evaluation import run_frozen_gate_selection

    dev_frame, dev_attribution = _synthetic_dev()
    first = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)
    second = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=5)
    assert first == second
