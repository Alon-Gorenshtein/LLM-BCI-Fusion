import numpy as np

from fusion.selective import aurc, aurc_optimal, excess_aurc, pooled_risk_coverage


def test_exact_toy_value():
    scores = np.arange(10, 0, -1, dtype=float)           # item 0 most confident
    errors = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])      # the two least confident are wrong
    groups = np.zeros(10, int)
    cov, risk = pooled_risk_coverage(scores, errors, groups, n_points=10)
    assert np.allclose(risk[:8], 0.0) and np.isclose(risk[8], 1 / 9) and np.isclose(risk[9], 0.2)
    assert np.isclose(aurc(scores, errors, groups, n_points=10), (1 / 9 + 0.2) / 10)
    assert np.isclose(excess_aurc(scores, errors, groups, n_points=10), 0.0)


def test_reversed_ranking_is_worse_and_random_is_near_base_rate():
    rng = np.random.default_rng(1)
    errors = (rng.random(3000) < 0.25).astype(int)
    groups = rng.integers(0, 4, 3000)
    assert excess_aurc(-errors.astype(float), errors, groups) == 0.0     # minus the error flag is the oracle ranking
    assert excess_aurc(errors.astype(float), errors, groups) > 0.15       # errors ranked first
    assert abs(aurc(rng.random(3000), errors, groups) - errors.mean()) < 0.02
    assert aurc_optimal(errors, groups) < errors.mean()


def test_groups_are_ranked_separately():
    scores = np.array([9.0, 8.0, 1.0, 0.5])       # group B scores are all lower than group A
    errors = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])
    cov, risk = pooled_risk_coverage(scores, errors, groups, n_points=2)
    assert np.isclose(risk[0], 0.0)               # top item of each group is correct
