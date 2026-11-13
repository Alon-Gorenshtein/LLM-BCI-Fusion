import numpy as np
import pytest

from fusion.weighting import reliability_to_beta, BETA_MIN, BETA_MAX


def test_midpoint_reliability_gives_the_fixed_weight_default():
    assert reliability_to_beta(0.5) == pytest.approx(1.0, rel=1e-9)


def test_extremes_map_to_the_grid_bounds():
    assert reliability_to_beta(0.0) == pytest.approx(BETA_MIN, rel=1e-3)
    assert reliability_to_beta(1.0) == pytest.approx(BETA_MAX, rel=1e-3)


def test_mapping_is_monotone_increasing():
    g = np.linspace(0.01, 0.99, 50)
    beta = reliability_to_beta(g)
    assert np.all(np.diff(beta) > 0)
