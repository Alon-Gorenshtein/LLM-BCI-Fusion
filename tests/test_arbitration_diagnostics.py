import numpy as np
import pandas as pd
import pytest

from fusion.arbitration_diagnostics import (
    characterize_gate_components,
    classify_gate_deviation_outcomes,
)

N_SYMBOLS = 36


def test_characterize_gate_components_returns_expected_shape():
    rng = np.random.default_rng(0)
    n = 100
    dev_frame = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % 15}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(n)],
        "target_code": rng.integers(1, N_SYMBOLS + 1, n),
    })
    dev_attribution_beta1 = dev_frame.copy()
    dev_attribution_beta1["beta"] = 1.0
    dev_attribution_beta1["phantom_agreement"] = rng.random(n) < 0.1
    dev_attribution_beta1["prior_capture"] = rng.random(n) < 0.05
    dev_attribution_beta1["fused_correct"] = rng.random(n) < 0.72

    result = characterize_gate_components(
        dev_frame, dev_attribution_beta1, high=0.05, low=-0.05, best_fixed_beta=1.0, n_folds=5
    )
    for key in [
        "phantom_agreement_auroc", "phantom_agreement_auprc", "phantom_agreement_n_events",
        "prior_capture_auroc", "prior_capture_auprc", "prior_capture_n_events",
        "v_hat_quartiles", "fraction_deviating_from_best_fixed",
        "fraction_favoring_lm", "fraction_favoring_neural",
    ]:
        assert key in result, f"missing {key}"
    assert 0.0 <= result["fraction_deviating_from_best_fixed"] <= 1.0
    assert len(result["v_hat_quartiles"]) == 5  # min, Q1, median, Q3, max
    assert len(result["gated_beta_by_selection"]) == n


def test_characterize_gate_components_rejects_a_lossy_label_join():
    rng = np.random.default_rng(0)
    n = 40
    dev_frame = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % 10}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(N_SYMBOLS)) for _ in range(n)],
        "target_code": rng.integers(1, N_SYMBOLS + 1, n),
    })
    dev_attribution_beta1 = dev_frame.copy()
    dev_attribution_beta1["beta"] = 1.0
    dev_attribution_beta1["phantom_agreement"] = rng.random(n) < 0.1
    dev_attribution_beta1["prior_capture"] = rng.random(n) < 0.05
    dev_attribution_beta1 = dev_attribution_beta1.iloc[:-1]  # one selection has no beta=1.0 label

    with pytest.raises(ValueError, match="dropped rows"):
        characterize_gate_components(
            dev_frame, dev_attribution_beta1, high=0.05, low=-0.05, best_fixed_beta=1.0, n_folds=5
        )


def _deviation_fixture():
    """Six selections at best_fixed_beta=1.0. Selection 0 stays at the fixed beta (not deviating);
    the other five deviate, with fused_correct laid out to give one helped, one hurt, and three
    with no effect (two correct at both betas, one incorrect at both).
    """
    keys = pd.DataFrame({
        "study": ["StudyB"] * 6,
        "participant_id": [f"P{i}" for i in range(6)],
        "session_id": ["S1"] * 6,
        "trial_number": range(6),
        "relative_path": [f"path{i}.edf" for i in range(6)],
    })
    gated = keys.copy()
    gated["gated_beta"] = [1.0, 2.0, 2.0, 2.0, 0.5, 0.5]

    #                       sel: 0      1      2      3      4      5
    fused_at = {
        0.5: [False, False, False, False, True,  False],
        1.0: [True,  False, True,  True,  True,  False],
        2.0: [False, True,  False, True,  False, False],
    }
    rows = []
    for beta, correct in fused_at.items():
        frame = keys.copy()
        frame["beta"] = beta
        frame["fused_correct"] = correct
        rows.append(frame)
    return gated, pd.concat(rows, ignore_index=True)


def test_classify_gate_deviation_outcomes_splits_helped_hurt_and_no_effect():
    gated, attribution = _deviation_fixture()

    result = classify_gate_deviation_outcomes(gated, attribution, best_fixed_beta=1.0)

    assert result["n_deviating"] == 5
    assert result["helped"] == 1  # selection 1: incorrect at 1.0, correct at 2.0
    assert result["hurt"] == 1  # selection 2: correct at 1.0, incorrect at 2.0
    assert result["no_effect"] == 3  # selections 3 and 4 correct at both, selection 5 at neither
    assert result["helped"] + result["hurt"] + result["no_effect"] == result["n_deviating"]


def test_classify_gate_deviation_outcomes_requires_every_beta_to_be_materialized():
    gated, attribution = _deviation_fixture()
    attribution = attribution[~np.isclose(attribution["beta"], 2.0)]

    with pytest.raises(ValueError, match="not materialized"):
        classify_gate_deviation_outcomes(gated, attribution, best_fixed_beta=1.0)
