import numpy as np
import pytest

from fusion.features import (
    posterior_entropy_bits, top1_probability, top1_top2_margin, reliability_features,
)


def test_uniform_distribution_has_max_entropy_and_zero_margin():
    uniform = np.full(36, 1 / 36)
    assert posterior_entropy_bits(uniform) == pytest.approx(np.log2(36), abs=1e-9)
    assert top1_top2_margin(uniform) == pytest.approx(0.0, abs=1e-9)
    assert top1_probability(uniform) == pytest.approx(1 / 36, abs=1e-9)


def test_one_hot_distribution_has_zero_entropy_and_full_margin():
    one_hot = np.zeros(36)
    one_hot[5] = 1.0
    assert posterior_entropy_bits(one_hot) == pytest.approx(0.0, abs=1e-6)
    assert top1_top2_margin(one_hot) == pytest.approx(1.0, abs=1e-9)
    assert top1_probability(one_hot) == pytest.approx(1.0, abs=1e-9)


def test_reliability_features_rejects_wrong_shape():
    with pytest.raises(ValueError):
        reliability_features(np.ones(10))
