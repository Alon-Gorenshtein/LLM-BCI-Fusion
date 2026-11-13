import numpy as np
import pandas as pd
import pytest

from fusion.arms import KEY_COLUMNS
from fusion.decoded_history import build_prior_object, replay_decoded_history


def _one_hot(index, n=36):
    vector = np.zeros(n)
    vector[index] = 1.0
    return vector


class _StubPrior:
    """Records every context it is queried with, in order, and returns a fixed vector per
    context key from a lookup table."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def prior(self, context):
        self.calls.append(context)
        return self.table[context]


def _row(study, participant, session, trial, path, p_neural, target_code):
    return {
        "study": study, "participant_id": participant, "session_id": session,
        "trial_number": trial, "relative_path": path,
        "p_neural": p_neural, "target_code": target_code,
    }


def test_replay_decoded_history_accumulates_context_within_a_phrase_and_resets_across_phrases():
    # Phrase p1: trial 1 queries "" (phrase start), emits 'B' (index 1); trial 2 then queries
    # "B" (the DECODED prefix, not the ground-truth target of trial 1). Phrase p2 (a different
    # relative_path) must reset back to "" for its own first trial.
    selections = pd.DataFrame([
        _row("StudyF", "p1", "s1", 0, "phrase1.csv", _one_hot(1), target_code=1),  # target 'A'
        _row("StudyF", "p1", "s1", 1, "phrase1.csv", _one_hot(2), target_code=3),  # target 'C'
        _row("StudyF", "p1", "s1", 0, "phrase2.csv", _one_hot(1), target_code=2),  # target 'B'
    ])
    table = {
        "": _one_hot(1),   # both phrase starts query "" -> fused argmax 'B' (index 1)
        "B": _one_hot(2),  # phrase1's second trial must query "B" (decoded), not "A" (target)
    }
    prior = _StubPrior(table)

    result = replay_decoded_history(selections, prior, beta=1.0)

    assert prior.calls == ["", "B", ""]
    assert set(result.columns) == set(KEY_COLUMNS) | {"fused_correct", "neural_correct"}
    assert len(result) == 3


def test_replay_decoded_history_raises_if_a_required_column_is_missing():
    selections = pd.DataFrame([_row("StudyF", "p1", "s1", 0, "phrase1.csv", _one_hot(1), 1)])
    selections = selections.drop(columns=["p_neural"])
    prior = _StubPrior({"": _one_hot(1)})
    with pytest.raises(ValueError, match="missing required columns"):
        replay_decoded_history(selections, prior, beta=1.0)


def test_build_prior_object_rejects_unsupported_prior_model():
    with pytest.raises(ValueError, match="gpt2 or ngram5_kn"):
        build_prior_object("some-other-model")
