import numpy as np
import pandas as pd
import pytest

from fusion.stats import (
    oracle_gap_closure, benjamini_hochberg, participant_cluster_bootstrap_rate_diff,
    cluster_robust_contrast,
)


def _long_frame():
    rows = []
    prior_capture_rates = {"floor": 0.40, "fixed": 0.30, "adaptive": 0.15, "oracle": 0.05}
    # fused_correct is an accuracy outcome (higher is better), unlike prior_capture (lower is
    # better) -- oracle is highest, floor is lowest, adaptive lands between fixed and oracle.
    fused_correct_rates = {"floor": 0.60, "fixed": 0.70, "adaptive": 0.85, "oracle": 0.95}
    rng = np.random.default_rng(0)
    rng_fused = np.random.default_rng(1)
    for arm in prior_capture_rates:
        rate = prior_capture_rates[arm]
        fused_rate = fused_correct_rates[arm]
        for participant in range(20):
            for _ in range(30):
                rows.append({
                    "participant_id": f"P{participant}",
                    "arm": arm,
                    "prior_capture": bool(rng.random() < rate),
                    "fused_correct": bool(rng_fused.random() < fused_rate),
                })
    return pd.DataFrame(rows)


def test_oracle_gap_closure_is_between_zero_and_one_when_adaptive_is_between_fixed_and_oracle():
    # oracle_gap_closure is only meaningful for outcome="fused_correct" (the oracle arm's own
    # selection objective) -- see the ValueError test below for the rejected case.
    closure = oracle_gap_closure(_long_frame(), "fused_correct")
    assert 0.0 < closure < 1.0


def test_oracle_gap_closure_rejects_an_outcome_other_than_fused_correct():
    """The oracle arm is selected by maximizing fused_correct per selection; calling
    oracle_gap_closure with a different outcome (e.g. prior_capture) would compare a numerator
    built on one objective against a denominator built on another. Must raise, not compute a
    nonsensical closure fraction.
    """
    with pytest.raises(ValueError):
        oracle_gap_closure(_long_frame(), "prior_capture")


def test_benjamini_hochberg_returns_one_value_per_input_and_never_shrinks_a_pvalue():
    adjusted = benjamini_hochberg([0.001, 0.04, 0.20, 0.50])
    assert len(adjusted) == 4
    assert np.all(adjusted >= np.array([0.001, 0.04, 0.20, 0.50]) - 1e-9)


def test_bootstrap_ci_contains_the_point_estimate():
    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        _long_frame(), "prior_capture", "fixed", "adaptive", n_bootstrap=200
    )
    assert lower <= point <= upper


def test_cluster_robust_contrast_recovers_a_known_effect_and_excludes_reference_arm():
    frame = _long_frame()  # floor .40, fixed .30, adaptive .15, oracle .05
    table = cluster_robust_contrast(frame, "prior_capture", reference_arm="fixed")
    row_labels = list(table.index)

    # the reference arm itself must not appear as its own contrast row
    assert not any("T.fixed" in label or label.strip().lower() == "fixed" for label in row_labels)

    # adaptive arm must appear as a contrast row
    adaptive_rows = [label for label in row_labels if "adaptive" in label]
    assert len(adaptive_rows) == 1, f"Expected 1 adaptive row, got {len(adaptive_rows)}: {adaptive_rows}"

    # coefficient column is the first column
    coefficient_column = table.columns[0]
    adaptive_coef = table.loc[adaptive_rows[0], coefficient_column]

    # adaptive has LOWER prior_capture rate than fixed (0.163 vs 0.305),
    # so its log-odds coefficient vs the fixed reference should be negative.
    # The outcome column must be cast to int before passing to GEE to avoid sign inversion.
    assert adaptive_coef < 0, f"Expected negative coefficient for adaptive vs fixed, got {adaptive_coef}"


def test_cluster_robust_contrast_raises_for_a_degenerate_event_rate():
    """The floor arm has no LM and is structurally 0% for prior_capture by construction (there is
    no LM to capture anything) -- fitting GEE on such a frame produces a fabricated, degenerate
    coefficient with no warning. cluster_robust_contrast must raise instead.
    """
    rows = []
    for participant in range(20):
        for _ in range(10):
            rows.append({"participant_id": f"P{participant}", "arm": "floor", "prior_capture": False})
            rows.append({
                "participant_id": f"P{participant}", "arm": "fixed",
                "prior_capture": participant % 2 == 0,
            })
    frame = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        cluster_robust_contrast(frame, "prior_capture", reference_arm="fixed")


def test_bootstrap_reflects_participant_level_variance_not_row_level_variance():
    """A row-level bootstrap ignoring participant boundaries would be dominated by the large
    participant's many rows and converge to a near-zero-width CI. A correct participant-level
    bootstrap must reflect that the two participants completely disagree, and produce a wide CI
    spanning both directions.
    """
    rows = (
        [{"participant_id": "P_big", "arm": "fixed", "prior_capture": True}] * 500
        + [{"participant_id": "P_big", "arm": "adaptive", "prior_capture": False}] * 500
        + [{"participant_id": "P_small", "arm": "fixed", "prior_capture": False}] * 2
        + [{"participant_id": "P_small", "arm": "adaptive", "prior_capture": True}] * 2
    )
    frame = pd.DataFrame(rows)
    _, lower, upper = participant_cluster_bootstrap_rate_diff(
        frame, "prior_capture", "fixed", "adaptive", n_bootstrap=1000
    )
    # A row-level bootstrap would give a near-zero CI dominated by P_big's 500:500 split
    # A correct participant-level bootstrap reflects both participants' disagreement
    assert (upper - lower) > 0.3, f"CI width {upper - lower} too narrow; suggests row-level resampling"
    assert lower < 0.3, f"Lower bound {lower} should be negative or near zero"


def test_cluster_robust_contrast_intercept_has_the_correct_sign_for_a_sub_50pct_reference_rate():
    """Regression test for boolean-dtype-to-int casting: without this cast, statsmodels GEE
    silently inverts every coefficient's sign (including the intercept). The reference arm fixed
    has an empirical rate of 0.30 (below 50%), so its log-odds intercept must be negative.
    """
    frame = _long_frame()  # fixed rate is 0.30, well under 50%
    table = cluster_robust_contrast(frame, "prior_capture", reference_arm="fixed")
    intercept = table.loc["Intercept", table.columns[0]]
    assert intercept < 0, f"reference arm's rate is under 50%, so its log-odds intercept must be negative; got {intercept}"
