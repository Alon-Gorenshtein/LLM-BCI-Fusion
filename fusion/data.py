"""Load and join the sibling attribution study's already-materialized bigP3BCI artifacts.

Reused read-only from study_bci_llm_authorship/output/intermediate/ -- never modified,
never entangled with that study's own git history.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_authorship_root(start: Path) -> Path:
    """Walk upward from `start` looking for a directory literally named
    study_bci_llm_authorship next to some ancestor. Works at the canonical
    checkout depth and inside any git worktree, unlike a hardcoded parent count.
    """
    for candidate in start.resolve().parents:
        sibling = candidate / "study_bci_llm_authorship"
        if sibling.is_dir():
            return sibling
    raise RuntimeError(
        "could not locate study_bci_llm_authorship as an ancestor sibling directory"
    )


_ROOT_OVERRIDE = os.environ.get("FUSION_COMPANION_ROOT")
AUTHORSHIP_ROOT = Path(_ROOT_OVERRIDE) if _ROOT_OVERRIDE else _find_authorship_root(Path(__file__))
if str(AUTHORSHIP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORSHIP_ROOT))

_INTERMEDIATE = AUTHORSHIP_ROOT / "output" / "intermediate"

REQUIRED_SELECTION_COLUMNS = (
    "study", "participant_id", "session_id", "trial_number", "target_code",
    "target_symbol", "archive_correct", "p_neural", "context_prefix", "relative_path",
)

SELECTION_KEY = ("study", "participant_id", "session_id", "trial_number", "relative_path")


def load_selections() -> pd.DataFrame:
    """One row per online selection: neural posterior, target, and grouping keys."""
    frame = pd.read_parquet(_INTERMEDIATE / "selections.parquet")
    missing = [c for c in REQUIRED_SELECTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"selections.parquet missing columns: {missing}")
    return frame


def load_prior(prior_model: str) -> pd.DataFrame:
    """Context-prefix -> 36-way LM prior for one model in the sibling ladder."""
    shard_name = prior_model.replace("/", "_") + ".parquet"
    shard_path = _INTERMEDIATE / "prior_shards" / shard_name
    if not shard_path.exists():
        raise FileNotFoundError(f"no prior shard for {prior_model!r} at {shard_path}")
    return pd.read_parquet(shard_path, columns=["context_prefix", "p_lm"])


def load_attribution(prior_model: str, beta: float | None = None) -> pd.DataFrame:
    """Precomputed fixed-weight fusion outcomes for one model (optionally one beta)."""
    frame = pd.read_parquet(_INTERMEDIATE / "attribution.parquet")
    frame = frame[frame["prior_model"] == prior_model]
    if beta is not None:
        frame = frame[np.isclose(frame["beta"], beta)]
    return frame.reset_index(drop=True)


def load_neural_correctness() -> pd.DataFrame:
    """Per-selection neural_correct/neural_argmax -- computed by the sibling study from the
    neural posterior and target alone, independent of any LM prior or fusion beta. Pulled once
    from attribution.parquet and deduplicated onto the true unique selection key.
    """
    frame = pd.read_parquet(
        _INTERMEDIATE / "attribution.parquet",
        columns=list(SELECTION_KEY) + ["neural_correct", "neural_argmax"],
    )
    duplicated_disagreement = (
        frame.groupby(list(SELECTION_KEY))[["neural_correct", "neural_argmax"]].nunique().max().max()
    )
    if duplicated_disagreement > 1:
        raise ValueError(
            "neural_correct/neural_argmax disagree across prior_model/beta for at least one "
            "selection -- the prior/beta-invariance assumption this function relies on is violated"
        )
    return frame.drop_duplicates(subset=list(SELECTION_KEY))


def build_working_frame(prior_model: str) -> pd.DataFrame:
    """Join one prior's p_lm onto every selection, keyed by shared linguistic context, plus the
    correctly-sourced neural_correct/neural_argmax for the reliability estimator's fitting target.
    """
    selections = load_selections()
    prior = load_prior(prior_model)
    frame = selections.merge(prior, on="context_prefix", how="inner", validate="many_to_one")
    if len(frame) != len(selections):
        raise ValueError(
            f"join dropped rows: {len(selections)} selections -> {len(frame)} after merge "
            f"with {prior_model!r} prior shard"
        )
    neural_correctness = load_neural_correctness()
    frame = frame.merge(neural_correctness, on=list(SELECTION_KEY), how="inner", validate="many_to_one")
    if len(frame) != len(selections):
        raise ValueError(
            f"neural_correctness join dropped rows: {len(selections)} selections -> {len(frame)} "
            f"after merging attribution.parquet's neural_correct/neural_argmax"
        )
    return frame
