import numpy as np

from fusion.joint_features import joint_features, joint_features_frame

N_SYMBOLS = 36


def _one_hot(index):
    p = np.full(N_SYMBOLS, 1e-9)
    p[index] = 1.0 - 1e-9 * (N_SYMBOLS - 1)
    return p / p.sum()


def test_joint_features_returns_expected_keys():
    p_neural = np.full(N_SYMBOLS, 1 / N_SYMBOLS)
    p_lm = np.full(N_SYMBOLS, 1 / N_SYMBOLS)
    feats = joint_features(p_neural, p_lm)
    expected_keys = {
        "neural_entropy_bits", "neural_top1_prob", "neural_margin",
        "lm_entropy_bits", "lm_top1_prob", "lm_margin",
        "argmax_agree", "lm_prob_of_neural_winner", "neural_prob_of_lm_winner",
        "neural_rank_of_lm_winner", "lm_rank_of_neural_winner",
        "kl_neural_from_lm", "js_divergence", "entropy_diff", "entropy_ratio",
    }
    assert set(feats.keys()) == expected_keys


def test_argmax_agree_true_when_winners_match():
    p_neural = _one_hot(3)
    p_lm = _one_hot(3)
    feats = joint_features(p_neural, p_lm)
    assert feats["argmax_agree"] == 1.0
    assert feats["neural_rank_of_lm_winner"] == 0
    assert feats["lm_rank_of_neural_winner"] == 0


def test_argmax_agree_false_when_winners_differ():
    p_neural = _one_hot(3)
    p_lm = _one_hot(10)
    feats = joint_features(p_neural, p_lm)
    assert feats["argmax_agree"] == 0.0


def test_kl_and_js_zero_for_identical_distributions():
    p = np.full(N_SYMBOLS, 1 / N_SYMBOLS)
    feats = joint_features(p, p.copy())
    assert np.isclose(feats["kl_neural_from_lm"], 0.0, atol=1e-9)
    assert np.isclose(feats["js_divergence"], 0.0, atol=1e-9)


def test_entropy_diff_and_ratio_relate_the_two_entropies():
    p_neural = _one_hot(0)  # near-zero entropy
    p_lm = np.full(N_SYMBOLS, 1 / N_SYMBOLS)  # maximum entropy
    feats = joint_features(p_neural, p_lm)
    assert feats["entropy_diff"] < 0  # neural entropy well below lm entropy
    assert feats["entropy_ratio"] < 1.0


def test_joint_features_frame_has_one_row_per_input_pair():
    rng = np.random.default_rng(0)
    p_neural_col = [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(5)]
    p_lm_col = [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(5)]
    frame = joint_features_frame(p_neural_col, p_lm_col)
    assert len(frame) == 5
    assert "kl_neural_from_lm" in frame.columns
