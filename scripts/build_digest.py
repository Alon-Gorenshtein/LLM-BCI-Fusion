"""Build output/results_digest.json and output/stats_digest.json -- the manuscript's single
source of truth for every reported number. Reuses the verified fusion.* modules exactly as
built and independently re-verified during implementation; adds no new computation logic.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fusion.data import build_working_frame, load_attribution, AUTHORSHIP_ROOT
from fusion.reliability import fit_reliability_cv, reliability_auroc, reliability_calibration_error
from fusion.weighting import reliability_to_beta
from fusion.arms import assemble_all_arms, OUTCOME_COLUMNS
from fusion.net_gain import add_net_correct_gain, run_net_gain_comparison
from fusion.stats import (
    cluster_robust_contrast,
    participant_cluster_bootstrap_rate_diff,
    oracle_gap_closure,
    benjamini_hochberg,
    N_BOOTSTRAP,
)
from fusion.transport import leave_one_study_out_contrast
from fusion.ladder import run_ladder_sweep
from fusion.rescue import run_rescue_experiment

MAIN_PRIOR = "gpt2"
CLASSICAL_PRIOR = "ngram5_kn"

def _canonical_ladder_priors() -> list[str]:
    """Canonical prior_model names as stored in attribution.parquet's own column -- these can
    differ from prior_shards/ filenames (which escape '/' to '_'), so derive the ladder from the
    attribution table itself rather than the shard directory listing.
    """
    frame = pd.read_parquet(AUTHORSHIP_ROOT / "output" / "intermediate" / "attribution.parquet", columns=["prior_model"])
    return sorted(m for m in frame["prior_model"].unique() if m != "uniform")


LADDER_PRIORS = _canonical_ladder_priors()


def safe(fn, *args, **kwargs):
    try:
        return {"ok": True, "value": fn(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def aim1_four_arm(prior_model: str) -> dict:
    working = build_working_frame(prior_model)
    attribution = load_attribution(prior_model)
    fitted = fit_reliability_cv(working, n_folds=5)
    fitted = fitted.assign(beta_adaptive=reliability_to_beta(fitted["g_hat"]))

    all_arms = assemble_all_arms(fitted, attribution)
    all_arms = add_net_correct_gain(all_arms)

    arm_rates = (
        all_arms.groupby("arm")[["fused_correct", "neural_correct", "prior_capture", "phantom_agreement", "net_correct_gain"]]
        .mean()
        .to_dict(orient="index")
    )
    arm_n = all_arms.groupby("arm").size().to_dict()

    contrasts = {}
    # Net correct gain: primary. Floor is a genuine zero anchor here (not degenerate for bootstrap).
    for a, b in [("adaptive", "fixed"), ("adaptive", "floor"), ("oracle", "fixed")]:
        point, lower, upper = participant_cluster_bootstrap_rate_diff(all_arms, "net_correct_gain", a, b)
        contrasts[f"net_correct_gain__{a}_vs_{b}"] = {"point": point, "lower": lower, "upper": upper}

    # Binary secondary/mechanistic outcomes: GEE, floor excluded for prior_capture/phantom_agreement
    # (structurally 0% there -- cluster_robust_contrast raises on a degenerate arm by design).
    non_floor = all_arms[all_arms["arm"] != "floor"]

    def gee_as_dict(frame, outcome):
        result = safe(cluster_robust_contrast, frame, outcome, reference_arm="fixed")
        if result["ok"]:
            result["value"] = result["value"].to_dict(orient="index")
        return result

    for outcome in ["prior_capture", "phantom_agreement"]:
        contrasts[f"gee__{outcome}"] = gee_as_dict(non_floor, outcome)
    contrasts["gee__fused_correct"] = gee_as_dict(all_arms, "fused_correct")

    for outcome in ["prior_capture", "phantom_agreement", "fused_correct"]:
        point, lower, upper = participant_cluster_bootstrap_rate_diff(non_floor if outcome != "fused_correct" else all_arms, outcome, "adaptive", "fixed")
        contrasts[f"bootstrap__{outcome}__adaptive_vs_fixed"] = {"point": point, "lower": lower, "upper": upper}

    gap_closure = safe(oracle_gap_closure, all_arms, "fused_correct")

    return {
        "prior_model": prior_model,
        "n_selections": int(len(working)),
        "arm_rates": jsonable(arm_rates),
        "arm_n": jsonable(arm_n),
        "contrasts": jsonable(contrasts),
        "oracle_gap_closure_fused_correct": jsonable(gap_closure),
        "reliability_auroc": reliability_auroc(fitted),
        "reliability_ece": reliability_calibration_error(fitted),
    }


def main():
    digest = {"generated_by": "scripts/build_digest.py", "n_bootstrap": N_BOOTSTRAP, "ladder_priors": LADDER_PRIORS, "n_ladder_priors": len(LADDER_PRIORS)}

    digest["aim1"] = {p: aim1_four_arm(p) for p in [MAIN_PRIOR, CLASSICAL_PRIOR]}

    digest["aim3_transportability"] = {}
    for p in [MAIN_PRIOR, CLASSICAL_PRIOR]:
        working = build_working_frame(p)
        attribution = load_attribution(p)
        digest["aim3_transportability"][p] = {
            "prior_capture": jsonable(leave_one_study_out_contrast(working, attribution, outcome="prior_capture")),
            "fused_correct": jsonable(leave_one_study_out_contrast(working, attribution, outcome="fused_correct")),
        }

    digest["aim4_ladder"] = {
        "prior_capture": jsonable(run_ladder_sweep(LADDER_PRIORS, outcome="prior_capture")),
        "fused_correct": jsonable(run_ladder_sweep(LADDER_PRIORS, outcome="fused_correct")),
    }

    digest["rescue"] = {}
    for p in [MAIN_PRIOR, CLASSICAL_PRIOR]:
        digest["rescue"][p] = {
            "prior_capture": run_rescue_experiment(p, outcome="prior_capture"),
            "fused_correct": run_rescue_experiment(p, outcome="fused_correct"),
            "net_gain_comparison": run_net_gain_comparison(p),
        }

    # Multiplicity: BH correction across the primary family's contrast p-values where a GEE
    # p-value exists (net_correct_gain uses bootstrap CIs, not p-values, so BH applies only to
    # the GEE-based secondary co-outcome family below).
    pvals = []
    labels = []
    for p in [MAIN_PRIOR, CLASSICAL_PRIOR]:
        for outcome in ["prior_capture", "phantom_agreement", "fused_correct"]:
            gee = digest["aim1"][p]["contrasts"].get(f"gee__{outcome}")
            if gee and gee.get("ok"):
                table = gee["value"]
                for row_key, row in table.items():
                    if "adaptive" in str(row_key).lower():
                        pvals.append(row["P>|z|"])
                        labels.append(f"{p}__{outcome}__{row_key}")
    if pvals:
        adjusted = benjamini_hochberg(pvals)
        digest["bh_correction"] = {"labels": labels, "raw_p": pvals, "adjusted_p": jsonable(adjusted)}
    else:
        digest["bh_correction"] = {"labels": [], "raw_p": [], "adjusted_p": [], "note": "no GEE p-values available for the adaptive contrast row"}

    digest = jsonable(digest)

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results_digest.json", "w") as f:
        json.dump(digest, f, indent=2)
    print(f"Wrote {out_dir / 'results_digest.json'}")
    print(f"n_ladder_priors={len(LADDER_PRIORS)}: {LADDER_PRIORS}")


if __name__ == "__main__":
    main()
