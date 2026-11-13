import numpy as np
import pandas as pd

from fusion.stats import participant_cluster_bootstrap_rate_diff


def _two_arm_clustered_frame(K=40, m=5, seed=3):
    """Two arms scored on the same K participants with m selections each. The arm contrast has a
    participant-level component, so the interval width depends on resampling whole participants."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, K)
    contrast = rng.normal(0, 1, K)
    part = np.repeat(np.arange(K), m)
    arm_a = shared[part] + rng.normal(0, 0.3, K * m)
    arm_b = shared[part] + contrast[part] + rng.normal(0, 0.3, K * m)
    frame = pd.concat([
        pd.DataFrame({"participant_id": part, "arm": "a", "y": arm_a}),
        pd.DataFrame({"participant_id": part, "arm": "b", "y": arm_b}),
    ], ignore_index=True)
    return frame, arm_a.reshape(K, m).mean(1) - arm_b.reshape(K, m).mean(1)


def test_participant_bootstrap_interval_width_matches_analytic_cluster_sd():
    frame, cluster_diff = _two_arm_clustered_frame()
    K = len(cluster_diff)
    analytic = np.sqrt(np.mean((cluster_diff - cluster_diff.mean()) ** 2) / K)
    point, lower, upper = participant_cluster_bootstrap_rate_diff(frame, "y", "a", "b", n_bootstrap=1000)
    assert abs(point - cluster_diff.mean()) < 1e-12
    # the 95% percentile interval spans about 2 * 1.96 standard errors; merging repeated draws of
    # a participant would shrink the width to about 0.79 of this
    assert 0.92 < (upper - lower) / (2 * 1.96 * analytic) < 1.08
