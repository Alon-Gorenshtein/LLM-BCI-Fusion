"""Numerical-precision and independent re-scoring evidence for the stored language-model priors.

(a) GPT-2 is scored on all 951 contexts in 32-bit and in reduced precision with the scorer used for
    the stored priors; the reduced-precision priors are compared with the stored shard, and the effect
    on the fused decision is measured at every weight of the fixed grid.
(b) Two further models (gpt2-large and Qwen2.5-1.5B) are re-scored on the same random 25 contexts in
    32-bit precision and compared with their stored shards.

Writes output/revision/precision_check.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["FUSION_COMPANION_ROOT"])
sys.path.insert(0, str(ROOT))
from authorship.priors import TransformerPrior  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "revision"
SEED = 20260822
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
# model[:dtype]; the cluster scored the larger models in half precision, so a cluster-scored model is re-scored in bfloat16
RESCORE = ("gpt2-large", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B:bfloat16", "Qwen/Qwen2.5-1.5B:float16", "state-spaces/mamba-2.8b-hf:bfloat16")


def _shard(name: str) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(ROOT / "output" / "intermediate" / "prior_shards" / (name.replace("/", "_") + ".parquet"),
                            columns=["context_prefix", "p_lm"])
    return {c: np.asarray(p, dtype=float) for c, p in zip(frame["context_prefix"], frame["p_lm"])}


def _compare(new: dict[str, np.ndarray], stored: dict[str, np.ndarray]) -> dict:
    keys = sorted(new)
    a = np.stack([new[k] for k in keys])
    b = np.stack([stored[k] for k in keys])
    return {
        "n_contexts": len(keys),
        "max_abs_difference": float(np.abs(a - b).max()),
        "mean_abs_difference": float(np.abs(a - b).mean()),
        "top1_symbol_agreement": float((a.argmax(1) == b.argmax(1)).mean()),
    }


def _score(model: str, contexts: list[str], dtype: str | None, device: str) -> tuple[dict[str, np.ndarray], float]:
    start = time.time()
    scorer = TransformerPrior(model, device=device, dtype=dtype)
    scored = {c: np.asarray(scorer.prior(c), dtype=float) for c in contexts}
    return scored, time.time() - start


def _fused_effect(selections: pd.DataFrame, floor: float, stored: dict, new: dict) -> dict:
    pn = np.stack(selections["p_neural"].to_numpy())
    target = selections["target_code"].to_numpy().astype(int) - 1
    ctx = selections["context_prefix"].tolist()
    ps = np.stack([stored[c] for c in ctx])
    pr = np.stack([new[c] for c in ctx])
    neural_correct = (pn.argmax(1) == target)
    rows = {}
    for beta in BETAS:
        base = beta * np.log(np.maximum(pn, floor))
        fs = (base + np.log(np.maximum(ps, floor))).argmax(1)
        fr = (base + np.log(np.maximum(pr, floor))).argmax(1)
        gain_s = float(((fs == target).astype(int) - neural_correct.astype(int)).mean())
        gain_r = float(((fr == target).astype(int) - neural_correct.astype(int)).mean())
        rows[str(beta)] = {
            "share_fused_argmax_changed": float((fs != fr).mean()),
            "n_fused_argmax_changed": int((fs != fr).sum()),
            "n_fused_correctness_changed": int(((fs == target) != (fr == target)).sum()),
            "net_correct_gain_stored": gain_s,
            "net_correct_gain_reduced_precision": gain_r,
            "change_in_net_correct_gain": gain_r - gain_s,
        }
    return rows


def main(only_rescore: bool = False) -> None:
    inter = ROOT / "output" / "intermediate"
    selections = pd.read_parquet(inter / "selections.parquet")
    floor = float(re.search(r"PROBABILITY_FLOOR\s*=\s*([0-9.eE+-]+)", (ROOT / "authorship" / "attribution.py").read_text()).group(1))
    contexts = sorted(selections["context_prefix"].unique())
    result: dict = {"n_contexts_total": len(contexts), "seed": SEED, "floor": floor}

    existing = json.loads((OUT / "precision_check.json").read_text()) if (OUT / "precision_check.json").exists() else {}
    stored = _shard("gpt2")
    gpt2 = existing.get("gpt2_all_contexts", {}) if only_rescore else {}
    for label, dtype, device in (() if only_rescore else (("float32", None, "cpu"), ("bfloat16", "bfloat16", "cpu"), ("float16", "float16", "mps"))):
        try:
            scored, seconds = _score("gpt2", contexts, dtype, device)
        except Exception as error:  # reduced precision can be unsupported on a device
            gpt2[label] = {"skipped": f"{type(error).__name__}: {str(error)[:120]}"}
            print(label, "skipped", gpt2[label]["skipped"], flush=True)
            continue
        entry = {"seconds": seconds, "versus_stored_shard": _compare(scored, stored)}
        entry["fused_effect_by_beta"] = _fused_effect(selections, floor, stored, scored)
        gpt2[label] = entry
        print(label, entry["versus_stored_shard"], flush=True)
    result["gpt2_all_contexts"] = gpt2

    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(contexts, size=25, replace=False).tolist())
    rescored = existing.get("rescored_25_contexts", {}) if only_rescore else {}
    for spec in (sys.argv[2:] if only_rescore else RESCORE):
        model, _, dtype = spec.partition(":")
        scored, seconds = _score(model, sample, dtype or None, "cpu")
        stored_m = _shard(model)
        key = f"{model}:{dtype}" if dtype else model
        rescored[key] = {"model": model, "seconds": seconds, "device": "cpu", "dtype": dtype or "float32",
                         "versus_stored_shard": _compare(scored, {c: stored_m[c] for c in sample})}
        print(key, rescored[key]["versus_stored_shard"], flush=True)
    result["rescored_25_contexts"] = rescored
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_check.json").write_text(json.dumps(result, indent=2))
    print("wrote", OUT / "precision_check.json")


if __name__ == "__main__":
    main(only_rescore=len(sys.argv) > 1 and sys.argv[1] == "--rescore-only")
