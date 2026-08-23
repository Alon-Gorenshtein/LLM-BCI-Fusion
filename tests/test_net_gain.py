import pandas as pd

from fusion.net_gain import add_net_correct_gain
from fusion.arms import assemble_all_arms


def test_add_net_correct_gain_computes_the_signed_difference():
    frame = pd.DataFrame({
        "fused_correct": [True, False, True, False],
        "neural_correct": [False, True, True, False],
    })
    result = add_net_correct_gain(frame)
    assert result["net_correct_gain"].tolist() == [1, -1, 0, 0]


def test_add_net_correct_gain_does_not_mutate_the_input_frame():
    frame = pd.DataFrame({"fused_correct": [True], "neural_correct": [False]})
    add_net_correct_gain(frame)
    assert "net_correct_gain" not in frame.columns


def test_run_net_gain_comparison_on_real_data_has_a_bounded_ci():
    from fusion.net_gain import run_net_gain_comparison
    result = run_net_gain_comparison("gpt2")
    assert result["floor_net_gain"] == 0.0
    assert result["net_gain_diff_lower"] <= result["net_gain_diff_point"] <= result["net_gain_diff_upper"]


def test_add_net_correct_gain_on_the_canonical_four_arm_frame_is_zero_for_every_floor_row():
    """The redesigned primary endpoint's core anchor claim (design spec 3.1): the no-LM floor
    arm's net_correct_gain is exactly 0.0 by construction (fused_correct == neural_correct when
    no LM is used), on the pipeline's own canonical four-arm assembly function
    (fusion.arms.assemble_all_arms), not just on the separate rescue-arm path already covered by
    test_run_net_gain_comparison_on_real_data_has_a_bounded_ci above. Before Fix 3,
    OUTCOME_COLUMNS lacked neural_correct entirely, so
    add_net_correct_gain(assemble_all_arms(...)) raised KeyError.
    """
    from fusion.data import build_working_frame, load_attribution
    from fusion.reliability import fit_reliability_cv
    from fusion.weighting import reliability_to_beta

    prior_model = "gpt2"
    working = build_working_frame(prior_model)
    fitted = fit_reliability_cv(working, n_folds=5)
    working_frame_with_beta = fitted.assign(beta_adaptive=reliability_to_beta(fitted["g_hat"]))
    attribution = load_attribution(prior_model)

    combined = assemble_all_arms(working_frame_with_beta, attribution)
    with_gain = add_net_correct_gain(combined)

    floor_rows = with_gain[with_gain["arm"] == "floor"]
    assert len(floor_rows) > 0
    assert (floor_rows["net_correct_gain"] == 0.0).all()
