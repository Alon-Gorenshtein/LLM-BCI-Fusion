import numpy as np
import pytest

from fusion.data import (
    SELECTION_KEY,
    build_working_frame,
    load_attribution,
    load_neural_correctness,
    load_selections,
)


def test_load_selections_has_36_way_posteriors():
    frame = load_selections()
    assert len(frame) == 3373
    lengths = frame["p_neural"].apply(len)
    assert (lengths == 36).all()
    sums = frame["p_neural"].apply(np.sum)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_build_working_frame_joins_every_selection():
    frame = build_working_frame("gpt2")
    assert len(frame) == 3373
    assert "p_lm" in frame.columns
    assert frame["p_lm"].apply(len).eq(36).all()
    assert "neural_correct" in frame.columns
    assert "neural_argmax" in frame.columns
    assert frame["neural_correct"].dtype == bool
    assert frame["neural_argmax"].dtype in [np.int32, np.int64, int]


def test_build_working_frame_neural_correct_is_not_an_alias_of_archive_correct():
    """Regression test for the neural_correct sourcing fix: neural_correct must come from the
    sibling study's own neural-posterior-derived column (via load_neural_correctness), not be a
    silent stand-in for archive_correct (a different, LM-context-dependent outcome).
    """
    frame = build_working_frame("gpt2")
    assert frame["neural_correct"].mean() != pytest.approx(frame["archive_correct"].mean())
    assert (frame["neural_correct"] != frame["archive_correct"]).sum() == 649


def test_build_working_frame_unknown_prior_raises():
    with pytest.raises(FileNotFoundError):
        build_working_frame("not-a-real-model")


def test_load_neural_correctness_has_one_row_per_selection():
    frame = load_neural_correctness()
    assert len(frame) == 3373
    assert "neural_correct" in frame.columns
    assert "neural_argmax" in frame.columns
    assert frame["neural_correct"].dtype == bool
    assert frame["neural_argmax"].dtype in [np.int32, np.int64, int]
    # Verify no duplicates on the selection key
    assert len(frame.drop_duplicates(subset=list(SELECTION_KEY))) == len(frame)


def test_load_attribution_filters_to_requested_beta():
    frame = load_attribution("gpt2", beta=1.0)
    assert len(frame) == 3373
    assert np.allclose(frame["beta"], 1.0)
