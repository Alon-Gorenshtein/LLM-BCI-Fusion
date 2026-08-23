"""The Version 2 addendum's genuine holdout: Study N is touched exactly once, in the final
evaluation task, never during calibration, feature design, the oracle-structure analysis, or gate
training. Every other task in this plan calls split_dev_test and uses only the dev half.
"""
from __future__ import annotations

import pandas as pd

HOLDOUT_STUDY = "StudyN"


def split_dev_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by study: everything except HOLDOUT_STUDY is dev, HOLDOUT_STUDY alone is test."""
    if HOLDOUT_STUDY not in frame["study"].unique():
        raise ValueError(f"{HOLDOUT_STUDY!r} not present in frame -- cannot form the holdout split")
    dev = frame[frame["study"] != HOLDOUT_STUDY].reset_index(drop=True)
    test = frame[frame["study"] == HOLDOUT_STUDY].reset_index(drop=True)
    return dev, test
