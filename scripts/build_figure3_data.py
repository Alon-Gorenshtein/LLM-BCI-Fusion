"""Builds output/figure3_data.json: the four already-computed numbers behind
Figure 3's mechanistic summary panels. No new controller or statistical analysis -- panel (a) is a
cheap aggregation over already-materialized attribution data (no model fitting); panels (b)/(c)/(d)
are parsed directly from this project's already-committed real-run artifacts
(output/candidate_arbitration_nested_loso.txt, output/abstention_baseline_comparison.txt,
output/decoded_history_check.txt), so the figure cannot silently drift from the numbers already
published in Results/Discussion/Supplement.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.candidates import achievable_candidates_by_key
from fusion.data import load_attribution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
PRIORS = ("gpt2", "ngram5_kn")


def _candidate_collapse_distribution() -> dict:
    out = {}
    for prior in PRIORS:
        attribution = load_attribution(prior)
        candidates = achievable_candidates_by_key(attribution)
        counts = candidates["n_distinct_candidates"].value_counts(normalize=True).sort_index()
        out[prior] = {str(int(k)): float(v) for k, v in counts.items()}
    return out


def _parse_kv_sections(path: Path, keys: tuple) -> dict:
    """Parses this project's `=== {prior} ===` / `  key: value` real-run output convention (used by
    both candidate_arbitration_nested_loso.txt and decoded_history_check.txt)."""
    sections: dict = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            current = stripped.strip("= ").split()[0]
            sections[current] = {}
        elif current and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            if key in keys:
                sections[current][key] = float(val.strip())
    return sections


def _parse_signal_enrichment(path: Path, signal_names: tuple) -> dict:
    """Parses run_abstention_baseline_comparison.py's `--- Comparison table ---` block: per prior,
    per signal, the two printed rows (coverage 0.95 then 0.90) of
    `coverage error_fraction_captured enrichment accuracy_retained`."""
    text = path.read_text()
    prior_blocks = re.split(r"^=== (\w+) ===\n", text, flags=re.MULTILINE)
    out: dict = {}
    for i in range(1, len(prior_blocks), 2):
        prior = prior_blocks[i]
        block = prior_blocks[i + 1]
        comparison_section = block.split("--- Per-study sensitivity")[0]
        out[prior] = {}
        for signal in signal_names:
            m = re.search(
                rf"Signal: {re.escape(signal)}.*?\n.*?\n(.*?)\n(.*?)\n",
                comparison_section, re.DOTALL,
            )
            if m is None:
                raise ValueError(f"signal {signal!r} not found for prior {prior!r} in {path}")
            row95 = m.group(1).split()
            row90 = m.group(2).split()
            out[prior][signal] = {
                "coverage_0.95": {"error_fraction_captured": float(row95[1]), "enrichment": float(row95[2])},
                "coverage_0.90": {"error_fraction_captured": float(row90[1]), "enrichment": float(row90[2])},
            }
    return out


def _parse_delta_ci_line(path: Path, key: str) -> dict:
    """Parses this project's `key: point=X, lower=Y, upper=Z` line convention (used by
    decoded_history_check.txt's decoded_minus_ground_truth line)."""
    out: dict = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            current = stripped.strip("= ").split()[0]
        elif current and stripped.startswith(f"{key}:"):
            m = re.search(r"point=([-\d.]+), lower=([-\d.]+), upper=([-\d.]+)", stripped)
            if m is None:
                raise ValueError(f"could not parse {key!r} line for prior {current!r}: {stripped!r}")
            out[current] = {
                "point": float(m.group(1)), "lower": float(m.group(2)), "upper": float(m.group(3)),
            }
    return out


def main() -> None:
    data = {
        "candidate_collapse": _candidate_collapse_distribution(),
        "override_outcomes": _parse_kv_sections(
            OUTPUT_DIR / "candidate_arbitration_nested_loso.txt",
            ("n_helped", "n_hurt", "n_no_effect"),
        ),
        "abstention_enrichment": _parse_signal_enrichment(
            OUTPUT_DIR / "abstention_baseline_comparison.txt",
            (
                "candidate_ranker_p_hat", "fixed_fused_top1_prob", "fixed_fused_margin",
                "fixed_fused_entropy_bits", "neural_top1_prob", "neural_margin",
                "reliability_g_hat", "neural_lm_js_divergence",
            ),
        ),
        "decoded_history": _parse_kv_sections(
            OUTPUT_DIR / "decoded_history_check.txt",
            ("ground_truth_prefix_net_correct_gain", "decoded_history_net_correct_gain"),
        ),
    }
    delta_ci = _parse_delta_ci_line(OUTPUT_DIR / "decoded_history_check.txt", "decoded_minus_ground_truth")
    for prior in PRIORS:
        data["decoded_history"][prior]["decoded_minus_ground_truth"] = delta_ci[prior]
    out_path = OUTPUT_DIR / "figure3_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
