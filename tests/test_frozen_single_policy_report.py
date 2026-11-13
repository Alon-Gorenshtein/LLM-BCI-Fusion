import numpy as np
import pandas as pd

from scripts.run_frozen_single_policy_report import run_frozen_single_policy_report


def test_reports_all_four_outcomes_under_one_frozen_policy(monkeypatch):
    rng = np.random.default_rng(0)
    n = 100
    keys = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % 15}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
    })
    rows = []
    for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
        block = keys.copy()
        block["beta"] = beta
        block["fused_correct"] = rng.random(n) < 0.7
        block["neural_correct"] = rng.random(n) < 0.65
        block["phantom_agreement"] = rng.random(n) < 0.1
        block["prior_capture"] = rng.random(n) < 0.05
        block["p_fused_target"] = rng.random(n)
        rows.append(block)
    attribution = pd.concat(rows, ignore_index=True)

    working = keys.copy()
    working["p_neural"] = [rng.dirichlet(np.ones(36)) for _ in range(n)]
    working["p_lm"] = [rng.dirichlet(np.ones(36)) for _ in range(n)]
    working["target_code"] = rng.integers(1, 37, n)
    working["neural_correct"] = rng.random(n) < 0.65

    def fake_load_attribution(prior_model):
        return attribution

    def fake_build_working_frame(prior_model):
        return working

    monkeypatch.setattr(
        "scripts.run_frozen_single_policy_report.load_attribution", fake_load_attribution
    )
    monkeypatch.setattr(
        "scripts.run_frozen_single_policy_report.build_working_frame", fake_build_working_frame
    )

    result = run_frozen_single_policy_report("fake_prior", n_folds=5)
    assert result["policy_selected_on"] == "fused_correct"
    for outcome in ["prior_capture", "phantom_agreement", "fused_correct", "net_correct_gain"]:
        assert outcome in result["outcomes"]
        entry = result["outcomes"][outcome]
        assert "fixed_rate" in entry and "adaptive_rate" in entry
        assert "difference" in entry and "lower_ci" in entry and "upper_ci" in entry
