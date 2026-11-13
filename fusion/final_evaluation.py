"""Joint gate, final step: freeze one arbitration-gate configuration using
only dev (Studies B, F, L), then evaluate it exactly once against the best-fixed-beta control on
the Study N holdout. evaluate_on_holdout must be called at most once per prior_model in any real
run of this study -- repeated calls during development would defeat the point of the holdout.

The frozen policy is three things, and all three are fit on dev alone: the net-benefit model V(x),
the single best fixed beta, and the residual-gate thresholds (high, low). Study N contributes no
label to any of them.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fusion.data import AUTHORSHIP_ROOT, build_working_frame, load_attribution
from fusion.holdout import HOLDOUT_STUDY, split_dev_test
from fusion.arbitration import (
    RESIDUAL_MULTIPLIERS,
    fit_net_benefit_model,
    fit_net_benefit_model_frozen,
    residual_gate_beta,
)
from fusion.arms import KEY_COLUMNS
from fusion.rescue import BETA_GRID
from fusion.net_gain import add_net_correct_gain
from fusion.stats import participant_cluster_bootstrap_rate_diff

if str(AUTHORSHIP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORSHIP_ROOT))
from authorship.attribution import fuse, attribution_row  # noqa: E402

# Note the ceiling: the largest magnitude this grid can express is 0.2, so the widest selectable
# band is (high=0.2, low=-0.2), and even that deviates from best_fixed_beta on every row whose
# |V(x)| exceeds 0.2. There is no (high, low) pair meaning "never deviate", so exact identity to
# best_fixed_beta is outside the search space regardless of which pair scores best on dev -- but
# the near-identity (0.2, -0.2) pair IS in the search space and was rejected by the dev-side search
# in favor of tighter, more deviation-prone pairs for both priors (see the recorded policy in
# output/v2_holdout_evaluation.txt): the search preferred to deviate more, it was not forced to.
# Disclosed rather than fixed: widening the grid now would be retuning after seeing the result.
THRESHOLD_GRID = (0.02, 0.05, 0.1, 0.2)

_KEY = list(KEY_COLUMNS)


def _require_dev_only(*frames: pd.DataFrame) -> None:
    """Every dev-side input must carry a `study` column and must not contain the holdout study.

    Deliberately strict about the column being present: a frame whose study membership cannot be
    checked is exactly the frame that could smuggle Study N past this guard.
    """
    for frame in frames:
        if "study" not in frame.columns:
            raise ValueError(
                "dev inputs must carry a 'study' column so holdout membership can be verified"
            )
        if HOLDOUT_STUDY in set(frame["study"].unique()):
            raise ValueError(
                f"{HOLDOUT_STUDY!r} must not be present in run_frozen_gate_selection's inputs"
            )


def _require_complete_beta_coverage(frame: pd.DataFrame, attribution: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Return the `beta` slice of `attribution`, having proven it is exactly one row per selection
    key in `frame`.

    This is what lets downstream alignment be positional-free: with no duplicate keys and no
    missing keys, an inner join on the key columns is a bijection onto `frame`'s rows.
    """
    rows = attribution[np.isclose(attribution["beta"], beta)]
    n_duplicates = int(rows.duplicated(subset=_KEY).sum())
    if n_duplicates:
        raise ValueError(
            f"attribution has {n_duplicates} duplicate selection keys at beta={beta} -- "
            f"the key columns {_KEY} must identify at most one attribution row per beta"
        )
    frame_keys = set(map(tuple, frame[_KEY].to_numpy()))
    beta_keys = set(map(tuple, rows[_KEY].to_numpy()))
    missing = frame_keys - beta_keys
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(frame_keys)} selections have no beta={beta} attribution row "
            f"(first missing key: {sorted(missing)[0]}) -- the join feeding the arbitration gate "
            f"would silently drop them"
        )
    return rows


def v_hat_by_key(frame: pd.DataFrame, attribution: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold V(x) for every row of `frame`, returned WITH its selection keys.

    `fit_net_benefit_model` returns a bare Series carrying no key information, indexed by the
    position of its own internal inner join. Rather than assume that join is lossless and
    order-preserving, this reproduces the identical join here (same deterministic operation on the
    same inputs, so the same row order by construction) and first proves it is a bijection onto
    `frame`. Callers then align by key, never by position.
    """
    beta1 = _require_complete_beta_coverage(frame, attribution, 1.0)
    mirrored_join = frame.merge(
        beta1[_KEY + ["phantom_agreement", "prior_capture"]], on=_KEY, how="inner"
    )
    if len(mirrored_join) != len(frame):
        raise ValueError(
            f"the beta=1.0 join is not a bijection: {len(frame)} selections -> "
            f"{len(mirrored_join)} joined rows"
        )

    v_hat = fit_net_benefit_model(frame, attribution, n_folds=n_folds)
    if len(v_hat) != len(mirrored_join):
        raise ValueError(
            f"fit_net_benefit_model returned {len(v_hat)} values for {len(mirrored_join)} joined "
            f"rows -- its internal join no longer matches the one reproduced here"
        )

    out = mirrored_join[_KEY].reset_index(drop=True).copy()
    out["v_hat"] = np.asarray(v_hat, dtype=float)
    return out


def frozen_v_hat_by_key(
    dev_frame: pd.DataFrame, dev_attribution: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.DataFrame:
    """The frozen V(x) for the holdout, returned WITH its selection keys.

    Thin keyed wrapper over `fusion.arbitration.fit_net_benefit_model_frozen`, which owns the
    actual fit. Adds the holdout-specific guards: dev must not contain Study N, dev's beta=1.0
    labels must cover dev exactly, and no selection may appear in both frames.
    """
    _require_dev_only(dev_frame, dev_attribution)
    _require_complete_beta_coverage(dev_frame, dev_attribution, 1.0)

    dev_keys = set(map(tuple, dev_frame[_KEY].to_numpy()))
    test_keys = set(map(tuple, test_frame[_KEY].to_numpy()))
    overlap = dev_keys & test_keys
    if overlap:
        raise ValueError(
            f"{len(overlap)} selections appear in BOTH the dev and holdout frames -- the frozen "
            f"gate would be evaluated on rows it was fit on"
        )

    v_hat = fit_net_benefit_model_frozen(dev_frame, dev_attribution, test_frame)
    if len(v_hat) != len(test_frame):
        raise ValueError(
            f"frozen V(x) returned {len(v_hat)} values for {len(test_frame)} holdout selections"
        )

    out = test_frame[_KEY].copy()
    # fit_net_benefit_model_frozen indexes its Series to test_frame, so this aligns on index
    # rather than on position.
    out["v_hat"] = v_hat
    if out["v_hat"].isna().any():
        raise ValueError("frozen V(x) did not align to every holdout selection")
    return out.reset_index(drop=True)


def run_frozen_gate_selection(
    dev_frame: pd.DataFrame, dev_attribution: pd.DataFrame, n_folds: int = 5
) -> dict:
    """Select and freeze (high, low) thresholds and the single dev-wide best fixed beta, using only
    dev data. Raises if Study N is present in either input (this function must never see it).
    """
    _require_dev_only(dev_frame, dev_attribution)

    rates = {
        beta: dev_attribution.loc[np.isclose(dev_attribution["beta"], beta), "fused_correct"].mean()
        for beta in BETA_GRID
    }
    dev_wide_best_beta = max(rates, key=rates.get)

    v_hat = v_hat_by_key(dev_frame, dev_attribution, n_folds=n_folds)
    aligned = dev_frame.merge(v_hat, on=_KEY, how="inner", validate="one_to_one")
    if len(aligned) != len(dev_frame):
        raise ValueError(
            f"dev V(x) alignment lost rows: {len(dev_frame)} selections -> {len(aligned)}"
        )

    # Score each candidate gate branch by fusing directly, exactly as the holdout application will,
    # rather than looking up the nearest precomputed grid beta. The gate's 0.5x/2x betas need not
    # land on BETA_GRID (they do not when the best fixed beta sits at a grid edge), and scoring dev
    # under a snapped beta that the holdout would not actually apply would tune the thresholds
    # against a policy that is never deployed. Direct fusion reproduces the sibling attribution
    # table exactly at every grid beta, so this is identical where the grid overlaps and correct
    # where it does not.
    correct_by_multiplier = {}
    for multiplier in RESIDUAL_MULTIPLIERS:
        beta = dev_wide_best_beta * multiplier
        correct_by_multiplier[multiplier] = np.array([
            attribution_row(
                row.p_neural, row.p_lm, fuse(row.p_neural, row.p_lm, beta=beta), row.target_code - 1
            )["fused_correct"]
            for row in aligned.itertuples(index=False)
        ], dtype=bool)

    v = aligned["v_hat"].to_numpy()
    best_score = -np.inf
    best_pair = (THRESHOLD_GRID[0], -THRESHOLD_GRID[0])
    for high in THRESHOLD_GRID:
        for low in [-t for t in THRESHOLD_GRID]:
            if low >= high:
                continue
            gated_correct = np.where(
                v > high,
                correct_by_multiplier[2.0],
                np.where(v < low, correct_by_multiplier[0.5], correct_by_multiplier[1.0]),
            )
            score = float(gated_correct.mean())
            if score > best_score:
                best_score = score
                best_pair = (high, low)

    neural = aligned["neural_correct"].astype(int).mean()
    return {
        "high": float(best_pair[0]),
        "low": float(best_pair[1]),
        "dev_wide_best_fixed_beta": float(dev_wide_best_beta),
        "dev_gated_fused_correct": float(best_score),
        "dev_gated_net_correct_gain": float(best_score - neural),
        # The gate's own home-field score: how it did against the fixed-beta baseline on the very
        # data its thresholds were tuned against. Without this the holdout comparison has no
        # reference point, and a holdout loss reads as "failed to transfer" when the honest reading
        # may be that there was no dev-side win to transfer.
        "dev_best_fixed_fused_correct": float(rates[dev_wide_best_beta]),
        "dev_gated_minus_best_fixed": float(best_score - rates[dev_wide_best_beta]),
        "gate_betas": [float(dev_wide_best_beta * m) for m in RESIDUAL_MULTIPLIERS],
        # Verified, not assumed: whether every gate beta happens to land on BETA_GRID. It does when
        # the best fixed beta is interior and does not when it sits at a grid edge. Nothing depends
        # on it being True -- dev and holdout both fuse directly at the exact beta -- but a reader
        # checking that the dev scoring and the holdout application agree can confirm it here.
        "gate_betas_on_grid": bool(
            all(
                any(np.isclose(dev_wide_best_beta * m, g) for g in BETA_GRID)
                for m in RESIDUAL_MULTIPLIERS
            )
        ),
        "n_dev_selections": int(len(v_hat)),
        # The thresholds are tuned against out-of-fold V(x) but applied to a full-dev-refit V(x)
        # on the holdout. These summaries let a reader check the two distributions are on the same
        # scale, which is what makes that transfer legitimate.
        "dev_v_hat_mean": float(v.mean()),
        "dev_v_hat_sd": float(v.std(ddof=0)),
    }


def evaluate_frozen_policy(
    dev_frame: pd.DataFrame,
    dev_attribution: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_attribution: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """Freeze the gate on `dev_*`, then score it once against the best-fixed-beta control on
    `test_*`. Split out from `evaluate_on_holdout` so the whole path can be exercised on a
    pseudo-holdout carved out of dev, without spending the real Study N read.
    """
    policy = run_frozen_gate_selection(dev_frame, dev_attribution, n_folds=n_folds)
    best_beta = policy["dev_wide_best_fixed_beta"]

    test_v_hat = frozen_v_hat_by_key(dev_frame, dev_attribution, test_frame)
    aligned = test_frame.merge(test_v_hat, on=_KEY, how="inner", validate="one_to_one")
    if len(aligned) != len(test_frame):
        raise ValueError(
            f"V(x) alignment lost rows: {len(test_frame)} holdout selections -> {len(aligned)}"
        )

    gated_rows = []
    applied_betas = []
    for row in aligned.itertuples(index=False):
        beta = residual_gate_beta(best_beta, row.v_hat, policy["high"], policy["low"])
        applied_betas.append(beta)
        fused = fuse(row.p_neural, row.p_lm, beta=beta)
        gated_rows.append(attribution_row(row.p_neural, row.p_lm, fused, row.target_code - 1))
    # Row i of gated_rows was produced from row i of `aligned`, so re-attaching the keys by
    # position is correct by construction here (same convention as fusion.arms.compute_adaptive_arm).
    gated = pd.concat([aligned[_KEY].reset_index(drop=True), pd.DataFrame(gated_rows)], axis=1)
    gated["applied_beta"] = applied_betas
    gated["arm"] = "frozen_gate"

    fixed_rows = _require_complete_beta_coverage(test_frame, test_attribution, best_beta).copy()
    fixed_rows["arm"] = "best_fixed"
    if len(fixed_rows) != len(test_frame):
        raise ValueError(
            f"best-fixed arm has {len(fixed_rows)} rows for {len(test_frame)} holdout selections"
        )

    # Both arms must describe the same selections, and the neural reading recomputed here from
    # p_neural must match the one the sibling attribution table recorded. A mismatch would mean the
    # p_neural/target_code being fused are not the ones the control arm is scored on.
    crosscheck = gated[_KEY + ["neural_correct"]].merge(
        fixed_rows[_KEY + ["neural_correct"]], on=_KEY, how="inner",
        suffixes=("_gated", "_fixed"), validate="one_to_one",
    )
    if len(crosscheck) != len(test_frame):
        raise ValueError("frozen-gate and best-fixed arms do not cover the same holdout selections")
    disagreements = int(
        (crosscheck["neural_correct_gated"].astype(bool) != crosscheck["neural_correct_fixed"].astype(bool)).sum()
    )
    if disagreements:
        raise ValueError(
            f"{disagreements} holdout selections disagree on neural_correct between the recomputed "
            f"gate arm and the precomputed attribution table"
        )

    gated = add_net_correct_gain(gated)
    fixed_rows = add_net_correct_gain(fixed_rows)

    long_frame = pd.concat(
        [
            gated[["participant_id", "arm", "net_correct_gain"]],
            fixed_rows[["participant_id", "arm", "net_correct_gain"]],
        ],
        ignore_index=True,
    )
    point, lower, upper = participant_cluster_bootstrap_rate_diff(
        long_frame, "net_correct_gain", "frozen_gate", "best_fixed"
    )
    beta_counts = pd.Series(applied_betas).value_counts().sort_index()
    return {
        "policy": policy,
        "frozen_gate_net_correct_gain": float(gated["net_correct_gain"].mean()),
        "best_fixed_net_correct_gain": float(fixed_rows["net_correct_gain"].mean()),
        "frozen_gate_minus_best_fixed": {"point": point, "lower": lower, "upper": upper},
        "frozen_gate_fused_correct": float(gated["fused_correct"].astype(int).mean()),
        "best_fixed_fused_correct": float(fixed_rows["fused_correct"].astype(int).mean()),
        "applied_beta_counts": {float(b): int(c) for b, c in beta_counts.items()},
        "test_v_hat_mean": float(aligned["v_hat"].mean()),
        # ddof=0 to match dev_v_hat_sd's numpy default; these two are computed to be compared
        # against each other, so they must not use different conventions.
        "test_v_hat_sd": float(aligned["v_hat"].std(ddof=0)),
        "n_holdout_selections": int(len(test_frame)),
        "n_holdout_participants": int(test_frame["participant_id"].nunique()),
    }


def evaluate_on_holdout(prior_model: str, n_folds: int = 5) -> dict:
    """The single, one-shot evaluation this analysis builds toward: apply the
    frozen gate (selected using dev alone) to Study N, and compare its net correct gain against the
    dev-selected best fixed beta, both applied to Study N for the first and only time.
    """
    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)
    dev_frame, test_frame = split_dev_test(working)
    dev_attribution, test_attribution = split_dev_test(attribution)

    result = evaluate_frozen_policy(
        dev_frame, dev_attribution, test_frame, test_attribution, n_folds=n_folds
    )
    return dict(prior_model=prior_model, **result)
