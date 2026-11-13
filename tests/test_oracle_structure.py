import numpy as np
import pandas as pd

from scripts.analyze_oracle_structure import oracle_structure_auroc


def test_oracle_structure_auroc_returns_expected_shape(monkeypatch):
    rng = np.random.default_rng(0)
    n = 200
    participants = [f"P{i % 20}" for i in range(n)]
    studies = ["StudyB" if i % 4 else "StudyN" for i in range(n)]
    betas = rng.choice([0.25, 0.5, 1.0, 2.0, 4.0], size=n)

    working = pd.DataFrame({
        "study": studies,
        "participant_id": participants,
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "target_code": rng.integers(1, 37, n),
    })
    attribution = working.copy()
    attribution["beta"] = betas
    attribution["fused_correct"] = rng.random(n) < 0.7
    attribution["p_fused_target"] = rng.random(n)

    def fake_build_working_frame(prior_model):
        return working

    def fake_load_attribution(prior_model):
        # select_oracle_arm needs the FULL grid per selection key, so replicate each row 5 times
        # across the beta grid with independent random fused_correct/p_fused_target.
        rows = []
        for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
            block = working.copy()
            block["beta"] = beta
            block["fused_correct"] = rng.random(n) < 0.7
            block["p_fused_target"] = rng.random(n)
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    monkeypatch.setattr("scripts.analyze_oracle_structure.build_working_frame", fake_build_working_frame)
    monkeypatch.setattr("scripts.analyze_oracle_structure.load_attribution", fake_load_attribution)

    result = oracle_structure_auroc("fake_prior", n_folds=3)
    assert "macro_ovr_auroc" in result
    assert 0.0 <= result["macro_ovr_auroc"] <= 1.0
    assert "n_selections" in result
    assert "beta_class_counts" in result
