import pandas as pd

from fusion.net_gain import orient_adaptive_minus_fixed
from fusion.stats import participant_cluster_bootstrap_rate_diff


def test_orient_flips_sign_and_swaps_bounds():
    legacy = {"net_gain_diff_point": -0.01, "net_gain_diff_lower": -0.02, "net_gain_diff_upper": -0.002}
    assert orient_adaptive_minus_fixed(legacy) == {
        "adaptive_minus_fixed_point": 0.01,
        "adaptive_minus_fixed_lower": 0.002,
        "adaptive_minus_fixed_upper": 0.02,
    }


def test_better_adaptive_arm_gives_positive_adaptive_minus_fixed():
    rows = []
    for p in range(10):
        for _ in range(20):
            rows.append({"participant_id": f"P{p}", "arm": "best_fixed", "net_correct_gain": 0})
            rows.append({"participant_id": f"P{p}", "arm": "tuned_adaptive", "net_correct_gain": 1})
    long_frame = pd.DataFrame(rows)
    legacy_point, lo, hi = participant_cluster_bootstrap_rate_diff(
        long_frame, "net_correct_gain", "best_fixed", "tuned_adaptive", n_bootstrap=200)
    out = orient_adaptive_minus_fixed(
        {"net_gain_diff_point": legacy_point, "net_gain_diff_lower": lo, "net_gain_diff_upper": hi})
    assert legacy_point == -1.0 and out["adaptive_minus_fixed_point"] == 1.0
