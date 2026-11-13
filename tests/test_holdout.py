import pandas as pd
import pytest

from fusion.holdout import HOLDOUT_STUDY, split_dev_test


def _fake_frame():
    rows = []
    for study, participants in [("StudyB", ["B1", "B2"]), ("StudyN", ["N1", "N2"])]:
        for p in participants:
            for i in range(3):
                rows.append({"study": study, "participant_id": p, "trial_number": i})
    return pd.DataFrame(rows)


def test_split_dev_test_separates_by_study():
    dev, test = split_dev_test(_fake_frame())
    assert set(dev["study"].unique()) == {"StudyB"}
    assert set(test["study"].unique()) == {"StudyN"}
    assert len(dev) + len(test) == 12


def test_split_dev_test_has_no_participant_overlap():
    dev, test = split_dev_test(_fake_frame())
    assert set(dev["participant_id"]).isdisjoint(set(test["participant_id"]))


def test_split_dev_test_raises_if_holdout_study_absent():
    frame = _fake_frame()
    frame = frame[frame["study"] != HOLDOUT_STUDY]
    with pytest.raises(ValueError, match="StudyN"):
        split_dev_test(frame)
