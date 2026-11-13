import numpy as np
import pandas as pd

from scripts.analyze_oracle_ties import oracle_tie_structure


def test_oracle_tie_structure_counts_correct_betas_per_selection(monkeypatch):
    rng = np.random.default_rng(0)
    n = 40
    keys = pd.DataFrame({
        "study": ["StudyB"] * n,
        "participant_id": [f"P{i % 8}" for i in range(n)],
        "session_id": ["S1"] * n,
        "trial_number": range(n),
        "relative_path": [f"path{i}.edf" for i in range(n)],
    })
    rows = []
    for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
        block = keys.copy()
        block["beta"] = beta
        # First 10 selections: correct at every beta (5-way tie). Next 30: correct at exactly one.
        correct = np.zeros(n, dtype=bool)
        correct[:10] = True
        if beta == 1.0:
            correct[10:30] = True
        block["fused_correct"] = correct
        block["p_fused_target"] = rng.random(n)
        rows.append(block)
    attribution = pd.concat(rows, ignore_index=True)

    def fake_load_attribution(prior_model):
        return attribution

    monkeypatch.setattr("scripts.analyze_oracle_ties.load_attribution", fake_load_attribution)
    monkeypatch.setattr(
        "scripts.analyze_oracle_ties.split_dev_test",
        lambda frame: (frame, frame.iloc[0:0]),
    )

    result = oracle_tie_structure("fake_prior")
    assert result["n_selections"] == 40
    dist = result["distribution_of_num_correct_betas"]
    assert dist[5] == 10
    assert dist[1] == 20
    assert dist[0] == 10
