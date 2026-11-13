import numpy as np

from fusion.clustering import (bootstrap_means, percentile_ci, simultaneous_band,
                               equivalence_counts)


def _clustered(K=40, m=5, seed=3, sd_cluster=1.0):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0, sd_cluster, K)
    d = np.repeat(effects, m) + rng.normal(0, 0.3, K * m)
    return d, np.repeat(np.arange(K), m)


def test_participant_bootstrap_sd_matches_analytic_cluster_sd():
    d, part = _clustered()
    K = 40
    means = d.reshape(K, 5).mean(1)
    analytic = np.sqrt(np.mean((means - means.mean()) ** 2) / K)
    boot = bootstrap_means(d, part, np.arange(len(d)), scheme="participant", n_boot=4000, seed=1)
    # merging repeated draws (the relabelling bug) would make this ratio about 0.79
    assert 0.92 < boot.std() / analytic < 1.08


def test_crossed_scheme_widens_when_contexts_matter():
    P = C = 30
    rng = np.random.default_rng(5)
    u, v = rng.normal(0, 1, P), rng.normal(0, 1, C)
    part = np.repeat(np.arange(P), C); ctx = np.tile(np.arange(C), P)
    d = u[part] + v[ctx] + rng.normal(0, 0.3, P * C)
    sd_p = bootstrap_means(d, part, ctx, "participant", 2000, 1).std()
    sd_x = bootstrap_means(d, part, ctx, "crossed", 2000, 1).std()
    assert sd_x > 1.25 * sd_p


def test_simultaneous_band_properties():
    rng = np.random.default_rng(0)
    boot = np.column_stack([rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), np.zeros(5000)])
    point = np.zeros(3)
    lo, hi, c = simultaneous_band(boot, point)
    assert c > 2.1                      # wider than a single 95% interval (1.96)
    assert lo[2] == hi[2] == 0.0        # an exact tie keeps zero width and does not enter the max


def test_equivalence_counts_at_margin():
    lower = np.array([-0.004, -0.02, 0.0, 0.001]); upper = np.array([0.004, 0.02, 0.0, 0.03])
    out = equivalence_counts(lower, upper, margin=0.01)
    # rows 1 and 3 lie entirely inside (-0.01, 0.01); rows 2 and 4 straddle a bound; none lies beyond a bound
    assert out == {"inside_margin": 2, "outside_margin": 0, "overlapping_margin": 2}
