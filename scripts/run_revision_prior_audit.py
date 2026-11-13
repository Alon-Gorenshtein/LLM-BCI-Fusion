"""Consistency audit of the companion posteriors, one row per non-null prior.

For each prior, the beta = 1 fused argmax is recomputed from the stored neural posterior and the
stored language-model prior shard, and compared with the fused argmax stored in the companion
attribution table. The same audit is run on two copies of the companion data (the September
revision copy and the earlier August copy), and the two copies are compared with each other, so a
mixed-version analysis cannot go unnoticed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {"R1": Path(os.environ["FUSION_COMPANION_ROOT"])}
if os.environ.get("FUSION_COMPANION_ROOT_EARLIER"):  # an earlier copy of the companion data, for the version comparison
    ROOTS["August"] = Path(os.environ["FUSION_COMPANION_ROOT_EARLIER"])
OUT = Path(__file__).resolve().parents[1] / "output" / "revision"
KEY = ["study", "participant_id", "session_id", "trial_number", "relative_path"]


def _floor(root: Path) -> float:
    text = (root / "authorship" / "attribution.py").read_text()
    match = re.search(r"PROBABILITY_FLOOR\s*=\s*([0-9.eE+-]+)", text)
    return float(match.group(1))


def _audit_one(root: Path) -> dict:
    inter = root / "output" / "intermediate"
    floor = _floor(root)
    selections = pd.read_parquet(inter / "selections.parquet")
    attribution = pd.read_parquet(
        inter / "attribution.parquet",
        columns=KEY + ["prior_model", "beta", "fused_argmax", "fused_correct", "neural_correct"],
    )
    at_one = attribution[np.isclose(attribution["beta"], 1.0)]
    pn = np.stack(selections["p_neural"].to_numpy())
    result = {}
    for prior in sorted(p for p in at_one["prior_model"].unique() if p != "uniform"):
        shard = pd.read_parquet(inter / "prior_shards" / (prior.replace("/", "_") + ".parquet"),
                                columns=["context_prefix", "p_lm"])
        merged = selections[KEY + ["context_prefix"]].merge(shard, on="context_prefix", how="left", validate="many_to_one")
        pl = np.stack(merged["p_lm"].to_numpy())
        recomputed = np.argmax(np.log(np.maximum(pn, floor)) + np.log(np.maximum(pl, floor)), axis=1)
        stored = at_one[at_one["prior_model"] == prior].set_index(KEY)
        stored_argmax = stored.loc[pd.MultiIndex.from_frame(selections[KEY]), "fused_argmax"].to_numpy()
        net = float((stored["fused_correct"].astype(int) - stored["neural_correct"].astype(int)).mean())
        result[prior] = {
            "agreement_with_stored_argmax": float((recomputed == stored_argmax).mean()),
            "net_gain_beta1": net,
            "fused_correct_beta1": float(stored["fused_correct"].mean()),
            "recomputed_argmax": recomputed,
        }
    return {"floor": floor, "priors": result}


def main() -> None:
    audits = {name: _audit_one(root) for name, root in ROOTS.items()}
    earlier = audits.get("August", {"priors": {}})
    names = sorted(set(audits["R1"]["priors"]) | set(earlier["priors"]))
    rows = []
    for prior in names:
        r1 = audits["R1"]["priors"].get(prior)
        au = earlier["priors"].get(prior)
        same_between_copies = None
        if r1 is not None and au is not None:
            same_between_copies = float((r1["recomputed_argmax"] == au["recomputed_argmax"]).mean())
        rows.append({
            "prior": prior,
            "R1_agreement": None if r1 is None else r1["agreement_with_stored_argmax"],
            "R1_net_gain_beta1": None if r1 is None else r1["net_gain_beta1"],
            "August_agreement": None if au is None else au["agreement_with_stored_argmax"],
            "August_net_gain_beta1": None if au is None else au["net_gain_beta1"],
            "share_same_fused_argmax_between_copies": same_between_copies,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prior_audit.json").write_text(json.dumps(
        {"floors": {k: v["floor"] for k, v in audits.items()}, "rows": rows}, indent=2))
    lines = ["| Prior | R1: recomputed = stored | R1: net gain at beta 1 | August: recomputed = stored | August: net gain at beta 1 | Same fused symbol in both copies |",
             "|---|---|---|---|---|---|"]
    fmt = lambda v, d=4: "absent" if v is None else f"{v:.{d}f}"
    for r in rows:
        lines.append(f"| {r['prior']} | {fmt(r['R1_agreement'])} | {fmt(r['R1_net_gain_beta1'])} | "
                     f"{fmt(r['August_agreement'])} | {fmt(r['August_net_gain_beta1'])} | "
                     f"{fmt(r['share_same_fused_argmax_between_copies'])} |")
    (OUT / "prior_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
