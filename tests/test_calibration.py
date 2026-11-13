import numpy as np
import pandas as pd
import pytest

from fusion.calibration import calibrate_posterior, fit_temperature_cv


def test_calibrate_posterior_is_identity_at_temperature_one():
    p = np.array([0.7, 0.2, 0.1])
    calibrated = calibrate_posterior(p, temperature=1.0)
    assert np.allclose(calibrated, p, atol=1e-9)


def test_calibrate_posterior_sums_to_one_and_stays_nonnegative():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(36))
    for t in [0.2, 0.5, 1.0, 2.0, 5.0]:
        calibrated = calibrate_posterior(p, temperature=t)
        assert np.all(calibrated >= 0)
        assert np.isclose(calibrated.sum(), 1.0, atol=1e-8)


def test_calibrate_posterior_high_temperature_flattens_distribution():
    p = np.array([0.9, 0.05, 0.05])
    flattened = calibrate_posterior(p, temperature=10.0)
    # entropy must increase toward uniform as temperature grows
    assert flattened.max() < p.max()


def test_fit_temperature_cv_returns_one_value_per_row_out_of_fold():
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame({
        "participant_id": [f"P{i % 12}" for i in range(n)],
        "target_code": rng.integers(1, 37, n),
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
    })
    temps = fit_temperature_cv(frame, column="p_neural", n_folds=5)
    assert len(temps) == n
    assert np.all(temps > 0)


def test_fit_temperature_cv_raises_on_group_leakage_across_a_fold(monkeypatch):
    rng = np.random.default_rng(2)
    n = 12
    frame = pd.DataFrame({
        "participant_id": [f"P{i % 4}" for i in range(n)],
        "target_code": rng.integers(1, 37, n),
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
    })

    class LeakySplitter:
        """Stand-in for GroupKFold whose folds deliberately overlap on participant_id, so the
        test exercises fit_temperature_cv's own leakage guard rather than GroupKFold's contract.
        """

        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, X, groups=None):
            idx = np.arange(len(X))
            yield idx, idx

    monkeypatch.setattr("fusion.calibration.GroupKFold", LeakySplitter)
    with pytest.raises(RuntimeError, match="leakage"):
        fit_temperature_cv(frame, column="p_neural", n_folds=3)


def test_fit_temperature_cv_raises_on_too_many_folds():
    frame = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2"],
        "target_code": [1, 2, 3],
        "p_neural": [np.full(36, 1 / 36)] * 3,
    })
    with pytest.raises(ValueError, match="n_folds"):
        fit_temperature_cv(frame, column="p_neural", n_folds=5)
