import pandas as pd
import pytest

from fusion.route_a_holdout import HOLDOUT_STUDY_ROUTE_A, split_dev_test_route_a


def _toy_frame():
    return pd.DataFrame({
        "study": ["StudyB", "StudyB", "StudyF", "StudyL", "StudyN"],
        "participant_id": ["b1", "b1", "f1", "l1", "n1"],
        "value": [1, 2, 3, 4, 5],
    })


def test_holdout_study_is_studyb():
    assert HOLDOUT_STUDY_ROUTE_A == "StudyB"


def test_split_dev_test_route_a_partitions_by_study():
    dev, test = split_dev_test_route_a(_toy_frame())
    assert set(test["study"].unique()) == {"StudyB"}
    assert set(dev["study"].unique()) == {"StudyF", "StudyL", "StudyN"}
    assert len(dev) + len(test) == 5
    assert len(test) == 2


def test_split_dev_test_route_a_raises_if_studyb_absent():
    frame = _toy_frame()
    frame = frame[frame["study"] != "StudyB"].reset_index(drop=True)
    with pytest.raises(ValueError, match="StudyB"):
        split_dev_test_route_a(frame)


def test_split_dev_test_route_a_never_excludes_studyn_from_dev():
    dev, _ = split_dev_test_route_a(_toy_frame())
    assert "StudyN" in set(dev["study"].unique())
