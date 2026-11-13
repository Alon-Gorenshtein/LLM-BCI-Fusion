import pandas as pd

from fusion.candidate_arbitration_diagnostics import summarize_nested_loso


def _key(trial, study="StudyF"):
    return dict(study=study, participant_id="p1", session_id="s1", trial_number=trial,
                relative_path=f"t{trial}.csv")


def test_summarize_nested_loso_returns_every_required_metric():
    # Two selections. Selection 1: chosen_candidate == fixed candidate (no override), correct.
    # Selection 2: overridden, chosen_candidate is correct (fixed would have been wrong) -> "helped".
    nested_result = pd.DataFrame([
        {**_key(1), "chosen_candidate": 5, "overridden": False, "fixed_beta": 1.0},
        {**_key(2), "chosen_candidate": 9, "overridden": True, "fixed_beta": 1.0},
    ])
    attribution = pd.DataFrame([
        {**_key(1), "beta": 1.0, "fused_argmax": 5, "fused_correct": True, "neural_correct": True,
         "target_code": 6, "phantom_agreement": False, "prior_capture": False},
        {**_key(2), "beta": 1.0, "fused_argmax": 3, "fused_correct": False, "neural_correct": False,
         "target_code": 10, "phantom_agreement": False, "prior_capture": False},
    ])
    best_fixed_arm = attribution[["study", "participant_id", "session_id", "trial_number",
                                   "relative_path", "fused_correct", "neural_correct"]].copy()

    result = summarize_nested_loso(nested_result, attribution, best_fixed_arm)

    for key in [
        "overall_correctness", "net_correct_gain", "net_correct_gain_ci_lower",
        "net_correct_gain_ci_upper", "prior_capture_rate", "phantom_agreement_rate",
        "intervention_rate", "n_helped", "n_hurt", "n_no_effect",
        "fraction_of_fixed_errors_correctable", "oracle_headroom_recovered_fraction",
    ]:
        assert key in result, f"missing required metric: {key}"
    assert result["intervention_rate"] == 0.5  # 1 of 2 selections overridden
    assert result["n_helped"] == 1  # selection 2: overridden, fixed was wrong, chosen is correct
    assert result["n_hurt"] == 0
    assert result["n_no_effect"] == 0
    # sel1: neural_correct=True, chosen_correct=True -> neither capture nor phantom agreement.
    # sel2: neural_correct=False, chosen_correct=True -> a phantom agreement (LM-side rescue),
    # not a lookup of some fixed beta's own precomputed flags (which would silently be wrong here,
    # since attribution's beta=1.0 rows both have phantom_agreement=False by fixture construction).
    assert result["prior_capture_rate"] == 0.0
    assert result["phantom_agreement_rate"] == 0.5
