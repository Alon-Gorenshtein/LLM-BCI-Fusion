import numpy as np
import pandas as pd

from scripts.analyze_oracle_structure_context_grouped import oracle_structure_auroc_context_grouped


def test_context_grouped_auroc_returns_expected_shape(monkeypatch):
    rng = np.random.default_rng(0)
    n = 200
    contexts = [f"ctx{i % 30}" for i in range(n)]
    studies = ["StudyB" if i % 4 else "StudyN" for i in range(n)]
    betas = rng.choice([0.25, 0.5, 1.0, 2.0, 4.0], size=n)

    working = pd.DataFrame({
        "study": studies,
        "participant_id": [f"P{i % 20}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
        "context_prefix": contexts,
        "p_neural": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "p_lm": [rng.dirichlet(np.ones(36)) for _ in range(n)],
        "target_code": rng.integers(1, 37, n),
    })

    def fake_build_working_frame(prior_model):
        return working

    def fake_load_attribution(prior_model):
        rows = []
        for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
            block = working.copy()
            block["beta"] = beta
            block["fused_correct"] = rng.random(n) < 0.7
            block["p_fused_target"] = rng.random(n)
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    monkeypatch.setattr(
        "scripts.analyze_oracle_structure_context_grouped.build_working_frame",
        fake_build_working_frame,
    )
    monkeypatch.setattr(
        "scripts.analyze_oracle_structure_context_grouped.load_attribution", fake_load_attribution
    )

    result = oracle_structure_auroc_context_grouped("fake_prior", n_folds=3)
    assert "macro_ovr_auroc" in result
    assert 0.0 <= result["macro_ovr_auroc"] <= 1.0
    assert result["grouping"] == "context_prefix"
    assert "n_selections" in result
