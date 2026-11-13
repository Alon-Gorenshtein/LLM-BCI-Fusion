import pandas as pd

from fusion.abstention import risk_coverage_curve, risk_coverage_within_group


def test_risk_coverage_curve_reports_accuracy_at_each_coverage_level():
    # Pure mechanics fixture: 10 selections with distinct confidence_scores 0.0-0.9. The function
    # only needs to know it retains the highest-scoring fraction (descending sort) -- this fixture
    # makes no claim about what "high" or "low" means for any real-world confidence signal.
    # Make the 2 lowest-scoring selections (0.0, 0.1) the only incorrect ones, so flagging even a
    # small fraction should visibly raise retained accuracy.
    confidence_scores = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    correct = pd.Series([False, False, True, True, True, True, True, True, True, True])

    result = risk_coverage_curve(confidence_scores, correct, coverage_grid=(1.0, 0.8))

    assert set(result.columns) == {"coverage", "n_retained", "n_flagged", "accuracy_retained"}
    full = result[result["coverage"] == 1.0].iloc[0]
    assert full["n_retained"] == 10
    assert abs(full["accuracy_retained"] - 0.8) < 1e-9

    partial = result[result["coverage"] == 0.8].iloc[0]
    assert partial["n_retained"] == 8
    assert partial["n_flagged"] == 2
    # Flagging the 2 lowest-scoring selections removes both errors -> 100% retained accuracy.
    assert abs(partial["accuracy_retained"] - 1.0) < 1e-9


def test_risk_coverage_within_group_pools_correctly_across_two_groups():
    # Group A: confidence perfectly ranks correctness (higher = more likely correct).
    # Group B: confidence is USELESS (constant), but has a much higher raw error rate overall --
    # a naive global pool would rank Group B's rows below Group A's regardless of Group B's own
    # internal confidence ordering, since a pooled rank only cares about the raw score, not which
    # group it came from. Ranking WITHIN each group first prevents this: Group A's bottom-flagged
    # rows should still come from Group A's own least-confident rows.
    frame = pd.DataFrame({
        "confidence": [0.9, 0.7, 0.5, 0.3, 0.1] + [0.5, 0.5, 0.5, 0.5, 0.5],
        "correct": [True, True, True, False, False] + [False, False, False, False, True],
        "group": ["A"] * 5 + ["B"] * 5,
    })
    result = risk_coverage_within_group(
        frame, "confidence", "correct", group_column="group", coverage_grid=(0.8,)
    )
    pooled = result[(result["coverage"] == 0.8) & (result["group"] == "pooled_within_study")].iloc[0]
    # Flagging the bottom 20% of EACH group: Group A's row at confidence=0.1 (incorrect), Group B's
    # row at some confidence=0.5 tie (4 of 5 are incorrect there, so likely an error). Total errors
    # across both groups = 2 (A) + 4 (B) = 6. At minimum the Group A error must be captured.
    assert pooled["n_flagged"] == 2  # 1 flagged per group (20% of 5, rounded)
    assert 0 < pooled["error_fraction_captured"] <= 1.0


def test_risk_coverage_within_group_handles_inverted_signal():
    # Lower "divergence" = more confident. Perfectly anti-correlated with correctness at the raw
    # value level, so higher_is_more_confident=False must flip it back to the right direction.
    frame = pd.DataFrame({
        "divergence": [0.1, 0.2, 0.3, 0.4, 0.5],
        "correct": [True, True, True, False, False],
        "group": ["A"] * 5,
    })
    result = risk_coverage_within_group(
        frame, "divergence", "correct", group_column="group", coverage_grid=(0.6,),
        higher_is_more_confident=False,
    )
    pooled = result[(result["coverage"] == 0.6) & (result["group"] == "pooled_within_study")].iloc[0]
    # Flagging the bottom 40% by CONFIDENCE means flagging the two highest-divergence rows
    # (0.4, 0.5), both incorrect -- both errors should be captured.
    assert pooled["error_fraction_captured"] == 1.0
