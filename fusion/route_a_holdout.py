"""Route A's own genuine holdout, distinct from `fusion.holdout` (which
reserves Study N and must never be touched by this module). Study N is already spent as a holdout
for the joint arbitration gate; Route A designates the SMALLEST of the
three remaining studies (StudyB: 846 selections/18 participants) as its own fresh, zero-touch holdout,
chosen before any Route A code ran. Study N joins StudyF and StudyL as ordinary Route A dev data.
"""
from __future__ import annotations

import pandas as pd

HOLDOUT_STUDY_ROUTE_A = "StudyB"


def split_dev_test_route_a(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by study: everything except HOLDOUT_STUDY_ROUTE_A is dev, HOLDOUT_STUDY_ROUTE_A alone
    is test. Study N (already used as the joint-gate holdout) is ordinary dev data here.
    """
    if HOLDOUT_STUDY_ROUTE_A not in frame["study"].unique():
        raise ValueError(
            f"{HOLDOUT_STUDY_ROUTE_A!r} not present in frame -- cannot form the Route A holdout split"
        )
    dev = frame[frame["study"] != HOLDOUT_STUDY_ROUTE_A].reset_index(drop=True)
    test = frame[frame["study"] == HOLDOUT_STUDY_ROUTE_A].reset_index(drop=True)
    return dev, test
