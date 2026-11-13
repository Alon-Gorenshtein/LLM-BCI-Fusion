import numpy as np
import pandas as pd
import pytest

from fusion.nested_loso import run_nested_loso


def _toy_prior_frame(seed=0):
    rng = np.random.default_rng(seed)
    studies = [("StudyB", 3), ("StudyF", 3), ("StudyL", 3), ("StudyN", 3)]
    rows = []
    trial = 0
    for study, n_participants in studies:
        for p in range(n_participants):
            for t in range(10):
                rows.append({
                    "study": study, "participant_id": f"{study}_{p}", "session_id": "s1",
                    "trial_number": trial, "relative_path": f"{study}_{p}_{trial}.csv",
                    "p_neural": rng.dirichlet(np.ones(36)), "p_lm": rng.dirichlet(np.ones(36)),
                    "target_code": int(rng.integers(1, 37)),
                })
                trial += 1
    return pd.DataFrame(rows)


def _toy_attribution_for(frame, betas=(0.25, 0.5, 1.0, 2.0, 4.0), seed=1):
    # fused_argmax/neural_argmax are derived from each row's own p_neural/p_lm arrays (the same
    # arithmetic candidate_feature_table uses to compute its fixed-winner candidate) rather than
    # drawn independently at random, so the achievable-candidate set built from this attribution
    # table is guaranteed to contain the true fixed winner -- an invariant apply_combining_rule
    # (fusion/candidate_arbitration.py, not owned by this task) relies on.
    rng = np.random.default_rng(seed)
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    rows = []
    for _, row in frame.iterrows():
        p_neural = np.asarray(row["p_neural"])
        p_lm = np.asarray(row["p_lm"])
        neural_argmax = int(np.argmax(p_neural))
        key_vals = {col: row[col] for col in key_cols}
        for beta in betas:
            fused_argmax = int(np.argmax(beta * np.log(p_neural) + np.log(p_lm)))
            rows.append({
                **key_vals, "beta": beta, "fused_argmax": fused_argmax,
                "neural_argmax": neural_argmax, "fused_correct": bool(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def test_run_nested_loso_returns_exactly_one_row_per_selection_across_all_four_studies(monkeypatch):
    frame = _toy_prior_frame()
    attribution = _toy_attribution_for(frame)

    import fusion.nested_loso as module
    monkeypatch.setattr(module, "_load_frame_and_attribution", lambda prior_model: (frame, attribution))

    result = run_nested_loso("gpt2", n_inner_folds=2)
    assert len(result) == len(frame)
    assert set(result["outer_held_out_study"].unique()) == {"StudyB", "StudyF", "StudyL", "StudyN"}
    # p_hat_fixed_winner/p_hat_best_alternative (added for the abstention analysis) must
    # propagate through from apply_combining_rule's output, unchanged by the outer-fold concat.
    assert {"p_hat_fixed_winner", "p_hat_best_alternative"} <= set(result.columns)
    assert result["p_hat_fixed_winner"].between(0, 1).all()
    assert result["p_hat_best_alternative"].between(0, 1).all()
    # every selection's outer_held_out_study must equal its own study (each selection is predicted
    # exactly once, from the fold where its own study was the outer test set).
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    merged = result.merge(frame[key_cols], on=key_cols, how="inner")
    assert (merged["outer_held_out_study"] == merged["study"]).all()
