import numpy as np
import pandas as pd

from fusion.rescue import BETA_GRID
from fusion.route_a_diagnostics import characterize_route_a_components, suppression_cross_tab


def _toy_dev(n_participants=8, per_participant=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    trial = 0
    for p in range(n_participants):
        for _ in range(per_participant):
            rows.append({
                "study": "StudyF", "participant_id": f"p{p}", "session_id": "s1",
                "trial_number": trial, "relative_path": f"p{p}_{trial}.csv",
                "p_neural": rng.dirichlet(np.ones(36)), "p_lm": rng.dirichlet(np.ones(36)),
            })
            trial += 1
    return pd.DataFrame(rows)


def _toy_attribution(dev_frame, seed=1):
    # fused_argmax must vary across BETA_GRID for SOME selections and stay constant for others, so
    # that fusion.actionability.actionability_by_key produces both classes -- drawing fused_argmax
    # independently and uniformly for every beta (as opposed to gating on a per-selection coin
    # flip) makes P(all 5 draws collide) ~= (1/36)^4, so virtually every selection would be
    # actionable=True and Route A Stage 1's CV folds would see a single-class y.
    rng = np.random.default_rng(seed)
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    rows = []
    for _, row in dev_frame[key_cols].iterrows():
        is_actionable = bool(rng.integers(0, 2))
        base_argmax = int(rng.integers(0, 36))
        for beta in BETA_GRID:
            argmax = int(rng.integers(0, 36)) if is_actionable else base_argmax
            rows.append({
                **row.to_dict(), "beta": beta, "fused_argmax": argmax,
                "fused_correct": bool(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def test_characterize_route_a_components_returns_expected_shape():
    dev_frame = _toy_dev()
    dev_attribution = _toy_attribution(dev_frame)
    result = characterize_route_a_components(dev_frame, dev_attribution, n_folds=3)
    assert 0 <= result["stage1_auroc"] <= 1
    assert set(result["stage2_auroc_by_beta"].keys()) == set(BETA_GRID)
    assert result["n_total"] == len(dev_frame)


def test_suppression_cross_tab_covers_every_deviation_outcome_row():
    dev_frame = _toy_dev()
    dev_attribution = _toy_attribution(dev_frame)
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    keys = dev_frame[key_cols].iloc[:12].copy()
    keys["deviation_outcome"] = (["helped"] * 4) + (["hurt"] * 4) + (["no_effect"] * 4)

    result = suppression_cross_tab(dev_frame, dev_attribution, keys, n_folds=3, tau=0.5)
    assert set(result.keys()) == {"helped", "hurt", "no_effect"}
    for outcome in ("helped", "hurt", "no_effect"):
        assert result[outcome]["n"] == 4
        assert 0 <= result[outcome]["suppressed"] <= 4


def test_suppression_cross_tab_default_tau_uses_route_a_policy_selection():
    dev_frame = _toy_dev()
    dev_attribution = _toy_attribution(dev_frame)
    key_cols = ["study", "participant_id", "session_id", "trial_number", "relative_path"]
    keys = dev_frame[key_cols].iloc[:12].copy()
    keys["deviation_outcome"] = (["helped"] * 4) + (["hurt"] * 4) + (["no_effect"] * 4)

    result = suppression_cross_tab(dev_frame, dev_attribution, keys, n_folds=3)
    assert set(result.keys()) == {"helped", "hurt", "no_effect"}
    for outcome in ("helped", "hurt", "no_effect"):
        assert result[outcome]["n"] == 4
        assert 0 <= result[outcome]["suppressed"] <= 4
