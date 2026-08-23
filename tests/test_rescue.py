import numpy as np
import pandas as pd

from fusion.rescue import select_best_fixed_beta_arm_cv, tune_adaptive_anchor_cv, ANCHOR_GRID


def test_select_best_fixed_beta_arm_cv_picks_the_training_majority_preference():
    """20 participants: 15 have prior_capture=False ONLY at beta=4.0 (True at every other beta),
    5 have the opposite pattern. No fold's test set (4 of 20 participants, GroupKFold n_folds=5)
    can ever remove enough majority-type participants to flip the training-fold majority, so the
    correct out-of-fold selection lands on beta=4.0 for every single fold -- a clean, deterministic
    check that the selection reflects each fold's own training data rather than a fixed default.
    """
    rows = []
    for i in range(20):
        preferred = 4.0 if i < 15 else 0.25
        for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
            capture = not np.isclose(beta, preferred)
            for _ in range(10):
                rows.append({"participant_id": f"P{i}", "beta": beta, "prior_capture": capture})
    frame = pd.DataFrame(rows)
    result = select_best_fixed_beta_arm_cv(frame, n_folds=5)
    assert len(result) == 20 * 10
    assert set(result["chosen_beta"].unique()) == {4.0}


def test_select_best_fixed_beta_arm_cv_direction_differs_by_outcome():
    """One shared fixture where prior_capture (a harm rate -- lower is better) and fused_correct
    (an accuracy rate -- higher is better) peak at OPPOSITE betas: prior_capture is lowest at
    beta=4.0 (highest at beta=0.25), while fused_correct is highest at beta=1.0 (lowest at
    beta=0.25). Every participant shares the exact same per-beta true/false counts, so the rate
    for each beta is invariant to which participants land in a given fold's training split --
    the correct out-of-fold choice is therefore beta=4.0 for prior_capture and beta=1.0 for
    fused_correct, deterministically, for every fold. An implementation that always minimizes
    `outcome` (the pre-fix bug) would instead pick beta=0.25 for fused_correct -- the single
    worst-accuracy beta in the grid -- so this test fails under that bug and passes only once
    the selector is direction-aware.
    """
    prior_capture_true_count = {0.25: 9, 0.5: 7, 1.0: 5, 2.0: 3, 4.0: 1}
    fused_correct_true_count = {0.25: 1, 0.5: 3, 1.0: 9, 2.0: 6, 4.0: 4}
    rows = []
    for i in range(20):
        for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
            pc_true = prior_capture_true_count[beta]
            fc_true = fused_correct_true_count[beta]
            for rep in range(10):
                rows.append({
                    "participant_id": f"P{i}",
                    "beta": beta,
                    "prior_capture": rep < pc_true,
                    "fused_correct": rep < fc_true,
                })
    frame = pd.DataFrame(rows)

    by_harm = select_best_fixed_beta_arm_cv(frame, n_folds=5, outcome="prior_capture")
    assert set(by_harm["chosen_beta"].unique()) == {4.0}

    by_accuracy = select_best_fixed_beta_arm_cv(frame, n_folds=5, outcome="fused_correct")
    assert set(by_accuracy["chosen_beta"].unique()) == {1.0}


def test_select_best_fixed_beta_arm_cv_rejects_more_folds_than_participants():
    frame = pd.DataFrame([
        {"participant_id": "P0", "beta": 1.0, "prior_capture": False},
        {"participant_id": "P1", "beta": 1.0, "prior_capture": True},
    ])
    import pytest
    with pytest.raises(Exception):
        select_best_fixed_beta_arm_cv(frame, n_folds=5)


def test_tune_adaptive_anchor_cv_produces_one_row_per_selection_with_a_valid_chosen_anchor():
    """tune_adaptive_anchor_cv now fits its own reliability estimator internally (per fold, on
    that fold's training participants only), so the fixture supplies `neural_correct` (the
    estimator's fitting target) rather than a hand-supplied `g_hat`. Each participant's 5 rows
    alternate True/False so that every participant individually carries both classes -- this
    guarantees the internal LogisticRegression always sees both classes in every fold's training
    split, regardless of which participants GroupKFold happens to hold out.
    """
    rows = []
    for i in range(10):
        for j in range(5):
            p_neural = np.full(36, 1 / 36)
            p_neural[0] = 0.4
            p_neural = p_neural / p_neural.sum()
            p_lm = np.full(36, 1 / 36)
            rows.append({
                "study": "StudyB", "participant_id": f"P{i}", "session_id": "S1",
                "trial_number": len(rows), "relative_path": f"P{i}_{len(rows)}.edf",
                "p_neural": p_neural, "p_lm": p_lm, "target_code": 1,
                "neural_correct": j % 2 == 0,
            })
    frame = pd.DataFrame(rows)
    result = tune_adaptive_anchor_cv(frame, n_folds=5)
    assert len(result) == len(frame)
    assert result["arm"].unique().tolist() == ["tuned_adaptive"]
    valid_mins = {lo for lo, hi in ANCHOR_GRID}
    valid_maxs = {hi for lo, hi in ANCHOR_GRID}
    assert set(result["chosen_beta_min"].unique()) <= valid_mins
    assert set(result["chosen_beta_max"].unique()) <= valid_maxs


def test_tune_adaptive_anchor_cv_direction_differs_by_outcome():
    """One shared fixture where prior_capture (minimized) and fused_correct (maximized) prefer
    DIFFERENT parts of ANCHOR_GRID -- ANCHOR_GRID is now `tuple((b, b) for b in BETA_GRID) +
    (...)` (fully nesting BETA_GRID, see Fix 1 in fusion/rescue.py), so it is not hardcoded here;
    the assertions below are written to be robust to its exact membership/order. Every participant
    carries the identical per-row-type mix (3 "harm" rows + 7 "phantom" rows + 2 "filler" rows,
    each internally identical), so aggregate per-anchor rates are invariant to which participants
    land in a given fold's training split -- the same style as
    test_select_best_fixed_beta_arm_cv_direction_differs_by_outcome above.

    Since tune_adaptive_anchor_cv now fits its OWN reliability estimator per fold (Fix 2), g_hat
    is no longer hand-supplied -- it is the fitted model's out-of-fold prediction. Harm and
    phantom rows share an identical, strongly-peaked p_neural shape and are labeled
    neural_correct=True; filler rows are near-uniform and labeled neural_correct=False. This is a
    trivially-separable 2-point training set, so the freshly-fit LogisticRegression predicts
    g_hat ~= 0.9715 for every harm/phantom row in every fold (train or test, verified empirically
    via the real fusion.reliability-style fit -- stable across folds because every participant
    contributes the identical mix). Filler rows exist solely to give every fold's training labels
    both classes (a hand-supplied-g_hat fixture could skip this, but a neural_correct-driven
    fixture cannot); their own fused_correct is engineered to be exactly beta-invariant (p_lm
    uniform => fused argmax == neural argmax for any beta > 0), so they contribute an equal
    constant to every anchor's rate and never change which anchor wins.

    Harm rows: neural correct (peaked at target), LM prior wrong (peaked at a different symbol),
    with the probability gap engineered (via harm_row()'s closed-form floors, deliberately past
    the 1e-12 floor so the effective gap is the floor's own ceiling rather than a fragile nominal
    value) so the fused decision only flips to "correct" for beta somewhere around 9. Minimizing
    prior_capture -- which harm rows alone drive, since phantom/filler rows have
    neural_correct=False in the true fused sense and so never contribute to prior_capture --
    picks whichever anchor(s) cross that threshold; verified empirically (see assertion below)
    that after Fix 1 this is still a UNIQUE anchor, (0.1, 20.0) (beta~=17.19 at g_hat~=0.9715),
    the only grid member whose beta exceeds the harm threshold -- every other anchor, including
    all five new BETA_GRID-derived degenerate ones ((0.25,0.25) through (4.0,4.0)), stays at or
    below beta=7.39.

    Phantom rows: neural wrong, LM prior correct (the opposite pairing), engineered so the fused
    decision is "correct" only BELOW beta=3. Fix 1 widened the set of anchors below this
    threshold: it is no longer the 4-way tie it was before Fix 1 (back when ANCHOR_GRID had only
    one degenerate anchor, (1.0,1.0)) -- adding degenerate anchors for every BETA_GRID value
    means (0.25,0.25), (0.5,0.5), (1.0,1.0), and (2.0,2.0) all also land below the threshold,
    widening the tie to 7 anchors (verified empirically below). Rather than hardcode which single
    anchor wins that tie (brittle to any future grid reordering or added anchor), this test
    verifies the DIRECTION is still correct: the anchor tune_adaptive_anchor_cv actually selects
    for fused_correct is confirmed, by independently re-running tune_adaptive_anchor_cv restricted
    to just that one anchor, to achieve the same dataset-level fused_correct rate as the best any
    single anchor in the full grid can achieve alone -- i.e. the selector picked a true tied-best
    anchor, not a suboptimal one, without this test caring which member of the tie it was. An
    implementation that always minimizes `outcome` (the pre-fix bug) picks a member of the
    worst-rate group instead, so this still fails under that bug and passes only once the
    selector is direction-aware.
    """
    n_symbols = 36
    target_idx = 0
    wrong_idx = 1

    # Harm/phantom neural peak: a moderate, shared gap that both (a) separates cleanly from
    # filler's near-uniform shape for the reliability fit and (b) keeps the harm threshold
    # (HARM_PRIOR_GAP capped by the 1e-12 probability floor, divided by this gap) inside the
    # (7.39, 17.19) window between the grid's two largest realized beta values.
    NEURAL_GAP = 3.0
    # HARM_PRIOR_GAP is deliberately far past the point where authorship.attribution's 1e-12
    # probability floor caps its effective value -- this makes the resulting harm threshold a
    # fixed, floor-anchored constant (~9.2) rather than one sensitive to this nominal number.
    HARM_PRIOR_GAP = 100.0
    PHANTOM_PRIOR_GAP = 3.0 * NEURAL_GAP

    x_h = 1 / (35 + np.exp(NEURAL_GAP))         # neural floor for harm/phantom rows
    y_h = 1 / (35 + np.exp(HARM_PRIOR_GAP))     # prior floor for harm rows
    x_p = x_h                                   # phantom rows share the same neural gap
    y_p = 1 / (35 + np.exp(PHANTOM_PRIOR_GAP))  # prior floor for phantom rows

    def harm_row():
        p_neural = np.full(n_symbols, x_h)
        p_neural[target_idx] = 1 - 35 * x_h
        p_lm = np.full(n_symbols, y_h)
        p_lm[wrong_idx] = 1 - 35 * y_h
        return p_neural, p_lm

    def phantom_row():
        p_neural = np.full(n_symbols, x_p)
        p_neural[wrong_idx] = 1 - 35 * x_p
        p_lm = np.full(n_symbols, y_p)
        p_lm[target_idx] = 1 - 35 * y_p
        return p_neural, p_lm

    def filler_row():
        # Near-uniform p_neural (distinct low-confidence feature vector for the reliability fit)
        # with a UNIFORM p_lm: log(p_lm) is then constant across symbols, so fused argmax ==
        # neural argmax for every beta > 0 -- this row's fused_correct/prior_capture never
        # depends on which anchor is chosen, so it contributes an anchor-invariant constant.
        p_neural = np.full(n_symbols, 1 / n_symbols)
        p_lm = np.full(n_symbols, 1 / n_symbols)
        return p_neural, p_lm

    rows = []
    for i in range(20):
        for _ in range(3):
            p_neural, p_lm = harm_row()
            rows.append({
                "study": "StudyB", "participant_id": f"P{i}", "session_id": "S1",
                "trial_number": len(rows), "relative_path": f"P{i}_{len(rows)}.edf",
                "p_neural": p_neural, "p_lm": p_lm, "target_code": target_idx + 1,
                "neural_correct": True,
            })
        for _ in range(7):
            p_neural, p_lm = phantom_row()
            rows.append({
                "study": "StudyB", "participant_id": f"P{i}", "session_id": "S1",
                "trial_number": len(rows), "relative_path": f"P{i}_{len(rows)}.edf",
                "p_neural": p_neural, "p_lm": p_lm, "target_code": target_idx + 1,
                "neural_correct": True,
            })
        for _ in range(2):
            p_neural, p_lm = filler_row()
            rows.append({
                "study": "StudyB", "participant_id": f"P{i}", "session_id": "S1",
                "trial_number": len(rows), "relative_path": f"P{i}_{len(rows)}.edf",
                "p_neural": p_neural, "p_lm": p_lm, "target_code": target_idx + 1,
                "neural_correct": False,
            })
    frame = pd.DataFrame(rows)

    # prior_capture: verified to still have a genuine unique minimum after Fix 1 (only
    # (0.1, 20.0) crosses the harm threshold), so this side stays an exact assertion.
    by_harm = tune_adaptive_anchor_cv(frame, n_folds=5, outcome="prior_capture")
    assert set(zip(by_harm["chosen_beta_min"], by_harm["chosen_beta_max"])) == {(0.1, 20.0)}

    # fused_correct: Fix 1 widened this to a multi-way tie (verified: 7 of the 12 anchors in the
    # post-Fix-1 ANCHOR_GRID tie for the best training-fold rate), and which one wins is only a
    # function of ANCHOR_GRID's iteration order -- not a property of the selector's correctness.
    # Assert the DIRECTION instead of one literal anchor: the anchor tune_adaptive_anchor_cv
    # actually chose must achieve the same dataset-level fused_correct rate as the best any
    # single anchor in the grid can achieve alone (computed by re-running tune_adaptive_anchor_cv
    # restricted to each one-anchor grid in turn -- exercising the real fit/select/apply
    # machinery, not a reimplementation of it). This still fails under the pre-fix "always
    # minimize" bug, which picks a member of the worst-rate group instead.
    by_accuracy = tune_adaptive_anchor_cv(frame, n_folds=5, outcome="fused_correct")
    chosen_anchors = set(zip(by_accuracy["chosen_beta_min"], by_accuracy["chosen_beta_max"]))
    assert len(chosen_anchors) == 1, (
        f"expected every fold to agree on one anchor (fixture is participant-invariant), got "
        f"{chosen_anchors}"
    )
    (chosen_anchor,) = chosen_anchors

    per_anchor_rate = {
        anchor: tune_adaptive_anchor_cv(
            frame, n_folds=5, outcome="fused_correct", anchor_grid=(anchor,)
        )["fused_correct"].mean()
        for anchor in ANCHOR_GRID
    }
    best_achievable_rate = max(per_anchor_rate.values())
    assert per_anchor_rate[chosen_anchor] == best_achievable_rate
    assert by_accuracy["fused_correct"].mean() == best_achievable_rate
