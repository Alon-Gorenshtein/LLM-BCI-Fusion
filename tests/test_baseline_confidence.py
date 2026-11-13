import numpy as np
import pandas as pd

from fusion.baseline_confidence import BASELINE_SIGNALS, build_baseline_confidence_table


def _toy_frame(seed=0):
    rng = np.random.default_rng(seed)
    studies = [("StudyB", 3), ("StudyF", 3), ("StudyL", 3), ("StudyN", 3)]
    rows = []
    trial = 0
    for study, n_participants in studies:
        for p in range(n_participants):
            for t in range(10):
                p_neural = rng.dirichlet(np.ones(36))
                neural_argmax = int(np.argmax(p_neural))
                # target_code is set to the neural argmax (a "correct" trial) on a fair coin flip,
                # and to an independent random symbol otherwise. An independently-drawn target
                # (uniform over 36 symbols, uncorrelated with p_neural) gives neural_correct a
                # ~1/36 positive rate; with only ~90 rows per outer-fold training split that is
                # small enough that some seeds (e.g. the original seed=0) produce a training split
                # with zero positive neural_correct rows, which makes
                # fusion.reliability.fit_reliability_frozen's LogisticRegression.fit raise
                # "needs samples of at least 2 classes". Coin-flipping target against the neural
                # argmax keeps the positive rate near 50% regardless of seed, matching how
                # test_nested_loso.py's own attribution fixture avoids a similar degenerate-fixture
                # failure by deriving values from the same arithmetic instead of an independent
                # random draw.
                if rng.integers(0, 2):
                    target = neural_argmax + 1
                else:
                    target = int(rng.integers(1, 37))
                rows.append({
                    "study": study, "participant_id": f"{study}_{p}", "session_id": "s1",
                    "trial_number": trial, "relative_path": f"{study}_{p}_{trial}.csv",
                    "p_neural": p_neural, "p_lm": rng.dirichlet(np.ones(36)),
                    "target_code": target,
                    "neural_correct": bool(neural_argmax == target - 1),
                })
                trial += 1
    return pd.DataFrame(rows)


def _toy_attribution_for(frame, betas=(0.25, 0.5, 1.0, 2.0, 4.0), seed=1):
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


def test_build_baseline_confidence_table_has_one_row_per_selection_with_all_signals(monkeypatch):
    frame = _toy_frame()
    attribution = _toy_attribution_for(frame)

    import fusion.baseline_confidence as module
    monkeypatch.setattr(module, "_load_frame_and_attribution", lambda prior_model: (frame, attribution))

    result = build_baseline_confidence_table("gpt2", n_inner_folds=2)
    assert len(result) == len(frame)
    assert set(result["outer_held_out_study"].unique()) == {"StudyB", "StudyF", "StudyL", "StudyN"}
    for signal in BASELINE_SIGNALS:
        assert signal in result.columns
        assert result[signal].notna().all()
