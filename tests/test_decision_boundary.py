import numpy as np

from fusion.decision_boundary import (
    candidate_boundary_distance,
    fixed_winner_stability_interval,
    pairwise_crossing_beta,
)


def test_pairwise_crossing_beta_matches_hand_derivation():
    # S_j(beta) = beta*log(p_neural[j]) + log(p_lm[j]). Choose two VALID probabilities (each in
    # (0, 1], since pairwise_crossing_beta clips to that domain before logging) with known logs so
    # the crossing point is exactly computable by hand: log(p_neural) = [0.0, -1.0],
    # log(p_lm) = [-2.0, 0.0].
    p_neural = np.array([1.0, np.exp(-1)])  # log = [0, -1]
    p_lm = np.array([np.exp(-2), 1.0])      # log = [-2, 0]
    # S_0(beta) = 0*beta + (-2) = -2; S_1(beta) = -1*beta + 0 = -beta. Equal when beta = 2.
    crossing = pairwise_crossing_beta(p_neural, p_lm, 0, 1)
    assert abs(crossing - 2.0) < 1e-9


def test_pairwise_crossing_beta_returns_none_for_parallel_lines():
    p_neural = np.array([1.0, 1.0])  # equal log(p_neural) -> parallel lines
    p_lm = np.array([2.0, 1.0])
    assert pairwise_crossing_beta(p_neural, p_lm, 0, 1) is None


def test_fixed_winner_stability_interval_matches_brute_force_scan():
    rng = np.random.default_rng(0)
    p_neural = rng.dirichlet(np.ones(36))
    p_lm = rng.dirichlet(np.ones(36))
    fixed_beta = 1.0

    lower, upper = fixed_winner_stability_interval(p_neural, p_lm, fixed_beta)

    def score(beta):
        return beta * np.log(p_neural) + np.log(p_lm)

    winner = int(np.argmax(score(fixed_beta)))
    betas = np.linspace(max(lower - 0.05, 1e-4), min(upper + 0.05, 5.0), 20000)
    winners = np.array([np.argmax(score(b)) for b in betas])
    same = winners == winner
    idx_fixed = int(np.searchsorted(betas, fixed_beta))
    lo = idx_fixed
    while lo > 0 and same[lo - 1]:
        lo -= 1
    hi = idx_fixed
    while hi < len(same) - 1 and same[hi + 1]:
        hi += 1
    # The analytic interval must match the brute-force scan to the scan's own resolution.
    assert abs(lower - betas[lo]) < 0.01
    assert abs(upper - betas[hi]) < 0.01 or upper > 4.9  # scan range caps at 5.0


def test_fixed_winner_stability_interval_lower_bound_is_floored_at_zero():
    rng = np.random.default_rng(1)
    p_neural = rng.dirichlet(np.ones(36))
    p_lm = rng.dirichlet(np.ones(36))
    lower, upper = fixed_winner_stability_interval(p_neural, p_lm, fixed_beta=1.0)
    assert lower >= 0.0


def test_candidate_boundary_distance_is_zero_for_the_winner_itself():
    rng = np.random.default_rng(1)
    p_neural = rng.dirichlet(np.ones(36))
    p_lm = rng.dirichlet(np.ones(36))
    fixed_beta = 1.0
    winner = int(np.argmax(fixed_beta * np.log(p_neural) + np.log(p_lm)))
    assert candidate_boundary_distance(p_neural, p_lm, fixed_beta, winner) == 0.0
