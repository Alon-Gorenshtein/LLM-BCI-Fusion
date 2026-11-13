import numpy as np
import pytest

from fusion.fused_posterior_features import (
    fused_posterior, fused_posterior_features, fused_posterior_features_frame,
)


def test_fused_posterior_at_beta_zero_reduces_to_lm_posterior():
    rng = np.random.default_rng(0)
    p_lm = rng.dirichlet(np.ones(36))
    p_neural = rng.dirichlet(np.ones(36))
    fused = fused_posterior(p_neural, p_lm, beta=0.0)
    assert np.allclose(fused, p_lm, atol=1e-9)


def test_fused_posterior_features_match_hand_derivation_for_a_peaked_case():
    # p_neural puts all mass on symbol 0; p_lm uniform. At beta=1, log p_neural for symbol 0 is 0
    # while every other symbol is clipped to log(1e-12), so the fused posterior collapses almost
    # entirely onto symbol 0 regardless of p_lm's uniform shape.
    p_neural = np.zeros(36)
    p_neural[0] = 1.0
    p_lm = np.full(36, 1 / 36)
    features = fused_posterior_features(p_neural, p_lm, beta=1.0)
    assert features["fused_top1_prob"] > 0.99
    assert features["fused_entropy_bits"] < 0.1
    assert features["fused_margin"] > 0.99


def test_fused_posterior_rejects_wrong_shape():
    with pytest.raises(ValueError):
        fused_posterior(np.ones(10), np.ones(36) / 36, beta=1.0)


def test_fused_posterior_features_frame_has_one_row_per_input():
    rng = np.random.default_rng(1)
    p_neural_col = [rng.dirichlet(np.ones(36)) for _ in range(5)]
    p_lm_col = [rng.dirichlet(np.ones(36)) for _ in range(5)]
    frame = fused_posterior_features_frame(p_neural_col, p_lm_col, beta=1.0)
    assert len(frame) == 5
    assert set(frame.columns) == {"fused_entropy_bits", "fused_top1_prob", "fused_margin"}
