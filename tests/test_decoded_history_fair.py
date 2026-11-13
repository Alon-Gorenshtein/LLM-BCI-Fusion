import numpy as np
import pandas as pd

from fusion.decoded_history import replay_decoded_history
from fusion.decoded_history_fair import CachedPrior, decoded_history_fair_comparison


class StubPrior:
    """Language-model prior that depends on the decoded context, so history matters."""
    calls = 0

    def prior(self, context: str):
        StubPrior.calls += 1
        rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(context)) + len(context))
        return rng.dirichlet(np.full(36, 0.5))


def _selections(n_phrases=6, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ph in range(n_phrases):
        for t in range(steps):
            rows.append({"study": "S", "participant_id": f"P{ph % 3}", "session_id": "s1",
                         "trial_number": t, "relative_path": f"phrase{ph}",
                         "p_neural": rng.dirichlet(np.full(36, 0.3)),
                         "target_code": int(rng.integers(1, 37))})
    return pd.DataFrame(rows)


def test_constant_beta_series_equals_scalar_beta():
    sel = _selections()
    a = replay_decoded_history(sel, StubPrior(), beta=1.5)
    b = replay_decoded_history(sel, StubPrior(), beta=pd.Series(1.5, index=sel.index))
    assert a.equals(b)


def test_per_row_beta_is_used():
    sel = _selections()
    betas = pd.Series(np.where(sel["trial_number"] % 2 == 0, 0.25, 4.0), index=sel.index)
    mixed = replay_decoded_history(sel, StubPrior(), beta=betas)
    low = replay_decoded_history(sel, StubPrior(), beta=0.25)
    assert not mixed["fused_correct"].equals(low["fused_correct"])


def test_cache_returns_identical_values_with_fewer_calls():
    StubPrior.calls = 0
    cached = CachedPrior(StubPrior())
    first = cached.prior("ab")
    second = cached.prior("ab")
    assert np.array_equal(first, second) and StubPrior.calls == 1


def _working(n_participants=10, phrases_per=2, steps=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        for ph in range(phrases_per):
            for t in range(steps):
                v = rng.dirichlet(np.full(36, 0.3))
                rows.append({"study": "S", "participant_id": f"P{p}", "session_id": "s1",
                             "trial_number": t, "relative_path": f"P{p}/phrase{ph}",
                             "p_neural": v, "target_code": int(rng.integers(1, 37)),
                             "neural_correct": bool(rng.random() < v.max())})
    return pd.DataFrame(rows)


def test_fair_comparison_covers_every_selection_once_and_tunes_on_training_only():
    working = _working()
    out = decoded_history_fair_comparison(working, CachedPrior(StubPrior()), n_folds=5, inner_folds=3)
    frame = out["selections"]
    assert len(frame) == len(working)
    assert frame[["participant_id", "relative_path", "trial_number"]].duplicated().sum() == 0
    assert {"fixed_net_gain", "adaptive_net_gain"} <= set(frame.columns)
    assert len(out["folds"]) == 5
    for fold in out["folds"]:
        assert fold["fixed_beta"] in (0.25, 0.5, 1.0, 2.0, 4.0)
        assert len(fold["anchor"]) == 2
