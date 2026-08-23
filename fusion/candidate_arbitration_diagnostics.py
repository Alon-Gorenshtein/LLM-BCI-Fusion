"""Diagnostics and the full reporting list for the nested-LOSO candidate-arbitration result
(design spec section 17.4). Oracle-headroom-recovered needs the grid oracle's own accuracy at each
selection's own beta grid, already materialized as fused_correct across BETA_GRID in
attribution.parquet -- the oracle's accuracy per selection is simply whether ANY grid beta produced
the correct answer.
"""
from __future__ import annotations

import pandas as pd

from fusion.arms import KEY_COLUMNS

_KEY = list(KEY_COLUMNS)


def summarize_nested_loso(
    nested_result: pd.DataFrame, attribution: pd.DataFrame, best_fixed_arm: pd.DataFrame
) -> dict:
    # Score each chosen candidate: correct iff it equals that selection's own target_code - 1.
    target_lookup = attribution[_KEY + ["target_code"]].drop_duplicates(subset=_KEY)
    scored = nested_result.merge(target_lookup, on=_KEY, how="inner", validate="one_to_one")
    if len(scored) != len(nested_result):
        raise ValueError(
            f"target lookup join dropped rows: {len(nested_result)} -> {len(scored)}"
        )
    scored["chosen_correct"] = scored["chosen_candidate"] == (scored["target_code"] - 1)

    fixed = best_fixed_arm.merge(nested_result[_KEY], on=_KEY, how="inner", validate="one_to_one")
    # Primary endpoint (spec section 17.4): candidate-arbitration vs nested-CV BEST-FIXED fusion,
    # not vs neural-only -- fixed["fused_correct"] is that best-fixed comparator's own correctness.
    net_correct_gain = scored["chosen_correct"].astype(int).mean() - fixed["fused_correct"].astype(int).mean()

    # prior_capture/phantom_agreement (authorship.attribution.attribution_row) are pure functions
    # of THIS decision's own correctness vs neural_correct -- neural_correct = neural_correct
    # and not chosen_correct = prior_capture; chosen_correct and not neural_correct = phantom
    # agreement -- so they are computed directly for the CHOSEN candidate, never looked up from a
    # fixed beta slice (that would silently report a different arm's mechanistic breakdown).
    neural_lookup = attribution[_KEY + ["neural_correct"]].drop_duplicates(subset=_KEY)
    with_neural = scored.merge(neural_lookup, on=_KEY, how="inner", validate="one_to_one")
    if len(with_neural) != len(scored):
        raise ValueError(
            f"neural_correct lookup join dropped rows: {len(scored)} -> {len(with_neural)}"
        )
    prior_capture = (
        with_neural["neural_correct"].astype(bool) & ~with_neural["chosen_correct"].astype(bool)
    )
    phantom_agreement = (
        with_neural["chosen_correct"].astype(bool) & ~with_neural["neural_correct"].astype(bool)
    )

    joined = scored.merge(fixed[_KEY + ["fused_correct"]], on=_KEY, how="inner", validate="one_to_one")
    joined = joined.rename(columns={"fused_correct": "fixed_correct"})
    # joined already carries "overridden" (inherited from nested_result via scored) -- no join needed.
    intervened = joined[joined["overridden"]]

    n_helped = int(((intervened["chosen_correct"]) & (~intervened["fixed_correct"])).sum())
    n_hurt = int(((~intervened["chosen_correct"]) & (intervened["fixed_correct"])).sum())
    n_no_effect = int(len(intervened) - n_helped - n_hurt)

    fixed_errors = joined[~joined["fixed_correct"]]
    correctable = fixed_errors["chosen_correct"].astype(int).sum() if len(fixed_errors) else 0
    fraction_correctable = float(correctable / len(fixed_errors)) if len(fixed_errors) else float("nan")

    oracle_correct_by_key = attribution.groupby(_KEY)["fused_correct"].any().reset_index()
    oracle_correct_by_key = oracle_correct_by_key.rename(columns={"fused_correct": "oracle_correct"})
    with_oracle = joined.merge(oracle_correct_by_key, on=_KEY, how="inner", validate="one_to_one")
    accuracy_new = with_oracle["chosen_correct"].astype(int).mean()
    accuracy_fixed = with_oracle["fixed_correct"].astype(int).mean()
    accuracy_oracle = with_oracle["oracle_correct"].astype(int).mean()
    denominator = accuracy_oracle - accuracy_fixed
    oracle_headroom_recovered = (
        float((accuracy_new - accuracy_fixed) / denominator) if abs(denominator) > 1e-12 else float("nan")
    )

    return {
        "overall_correctness": float(scored["chosen_correct"].astype(int).mean()),
        "net_correct_gain": float(net_correct_gain),
        "prior_capture_rate": float(prior_capture.mean()),
        "phantom_agreement_rate": float(phantom_agreement.mean()),
        "intervention_rate": float(nested_result["overridden"].astype(int).mean()),
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "n_no_effect": n_no_effect,
        "fraction_of_fixed_errors_correctable": fraction_correctable,
        "oracle_headroom_recovered_fraction": oracle_headroom_recovered,
    }
