from fusion.ladder import run_one_prior, run_ladder_sweep


def test_run_one_prior_returns_a_bounded_ci_on_real_data():
    result = run_one_prior("gpt2")
    assert result["prior_model"] == "gpt2"
    assert result["lower"] <= result["point"] <= result["upper"]


def test_run_ladder_sweep_returns_one_row_per_model():
    result = run_ladder_sweep(["gpt2", "uniform"])
    assert len(result) == 2
    assert set(result["prior_model"]) == {"gpt2", "uniform"}
