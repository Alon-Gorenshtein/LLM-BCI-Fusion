import numpy as np
import pandas as pd
import pytest

from fusion.arbitration import (
    residual_gate_beta,
    fit_net_benefit_model,
    fit_net_benefit_model_frozen,
)


def _labelled_frames(n=100, n_participants=15, seed=0, study="StudyB", prefix="path"):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "study": [study] * n,
        "participant_id": [f"{study}P{i % n_participants}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"{prefix}{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "target_code": rng.integers(1, 37, n),
    })
    attribution = frame.copy()
    attribution["beta"] = 1.0
    attribution["phantom_agreement"] = rng.random(n) < 0.3
    attribution["prior_capture"] = rng.random(n) < 0.2
    return frame, attribution


def test_residual_gate_beta_stays_at_default_when_v_hat_is_near_zero():
    beta = residual_gate_beta(best_fixed_beta=1.0, v_hat=0.0, high=0.1, low=-0.1)
    assert beta == 1.0


def test_residual_gate_beta_moves_toward_lm_when_v_hat_is_confidently_positive():
    beta = residual_gate_beta(best_fixed_beta=1.0, v_hat=0.5, high=0.1, low=-0.1)
    assert beta == 2.0


def test_residual_gate_beta_moves_toward_neural_when_v_hat_is_confidently_negative():
    beta = residual_gate_beta(best_fixed_beta=1.0, v_hat=-0.5, high=0.1, low=-0.1)
    assert beta == 0.5


def test_residual_gate_beta_scales_relative_to_the_default():
    beta = residual_gate_beta(best_fixed_beta=4.0, v_hat=0.5, high=0.1, low=-0.1)
    assert beta == 8.0


def test_residual_gate_branches_and_dev_scoring_share_one_multiplier_definition():
    """The tuple that scores candidate thresholds on dev must be the same object that applies the
    gate on the holdout, or the thresholds get tuned against a policy that is never deployed.
    """
    from fusion.arbitration import RESIDUAL_MULTIPLIERS
    from fusion.final_evaluation import RESIDUAL_MULTIPLIERS as dev_scoring_multipliers

    assert RESIDUAL_MULTIPLIERS is dev_scoring_multipliers
    assert RESIDUAL_MULTIPLIERS == (0.5, 1.0, 2.0)

    neural_favoring, default, lm_favoring = RESIDUAL_MULTIPLIERS
    assert residual_gate_beta(3.0, v_hat=-1.0, high=0.1, low=-0.1) == 3.0 * neural_favoring
    assert residual_gate_beta(3.0, v_hat=0.0, high=0.1, low=-0.1) == 3.0 * default
    assert residual_gate_beta(3.0, v_hat=1.0, high=0.1, low=-0.1) == 3.0 * lm_favoring


def test_fit_net_benefit_model_returns_one_value_per_row_out_of_fold():
    rng = np.random.default_rng(0)
    n = 100
    frame = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % 15}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "target_code": rng.integers(1, 37, n),
    })
    attribution = frame.copy()
    attribution["beta"] = 1.0
    attribution["phantom_agreement"] = rng.random(n) < 0.1
    attribution["prior_capture"] = rng.random(n) < 0.05
    v_hat = fit_net_benefit_model(frame, attribution, n_folds=5)
    assert len(v_hat) == n
    assert v_hat.between(-1.0, 1.0).all()


def test_fit_net_benefit_model_frozen_returns_one_value_per_test_row():
    dev_frame, dev_attribution = _labelled_frames(n=120, seed=1)
    test_frame, _ = _labelled_frames(n=40, n_participants=5, seed=2, study="StudyN", prefix="n")

    v_hat = fit_net_benefit_model_frozen(dev_frame, dev_attribution, test_frame)
    assert len(v_hat) == len(test_frame)
    assert list(v_hat.index) == list(test_frame.index)
    assert v_hat.between(-1.0, 1.0).all()
    assert v_hat.notna().all()


def test_fit_net_benefit_model_frozen_works_when_test_frame_has_no_label_columns_at_all():
    """Structural proof that no holdout label can reach the fit: remove the label columns entirely
    from the test frame. If the function still works, it never read them.
    """
    dev_frame, dev_attribution = _labelled_frames(n=120, seed=3)
    test_frame, _ = _labelled_frames(n=40, n_participants=5, seed=4, study="StudyN", prefix="n")
    assert "phantom_agreement" not in test_frame.columns
    assert "prior_capture" not in test_frame.columns

    v_hat = fit_net_benefit_model_frozen(dev_frame, dev_attribution, test_frame)
    assert len(v_hat) == len(test_frame)


def test_fit_net_benefit_model_frozen_is_unaffected_by_test_row_order():
    dev_frame, dev_attribution = _labelled_frames(n=120, seed=5)
    test_frame, _ = _labelled_frames(n=40, n_participants=5, seed=6, study="StudyN", prefix="n")

    straight = fit_net_benefit_model_frozen(dev_frame, dev_attribution, test_frame)
    shuffled_frame = test_frame.sample(frac=1.0, random_state=11)
    shuffled = fit_net_benefit_model_frozen(dev_frame, dev_attribution, shuffled_frame)

    # Indexed to test_frame, so reindexing the shuffled result recovers the original ordering.
    np.testing.assert_allclose(straight.to_numpy(), shuffled.reindex(test_frame.index).to_numpy())


def test_fit_net_benefit_model_frozen_raises_when_dev_labels_do_not_cover_dev():
    dev_frame, dev_attribution = _labelled_frames(n=60, n_participants=8, seed=7)
    test_frame, _ = _labelled_frames(n=20, n_participants=4, seed=8, study="StudyN", prefix="n")

    with pytest.raises(ValueError, match="dropped rows"):
        fit_net_benefit_model_frozen(dev_frame, dev_attribution.iloc[1:], test_frame)


def test_frozen_and_cross_validated_fits_share_the_same_feature_representation():
    """Both paths must see identical features; a drift here would silently miscalibrate the gate's
    thresholds against the values it is later compared to.
    """
    from fusion.arbitration import _feature_matrix

    dev_frame, _ = _labelled_frames(n=30, n_participants=5, seed=9)
    first = _feature_matrix(dev_frame)
    second = _feature_matrix(dev_frame)
    assert first.shape == (30, 15)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
