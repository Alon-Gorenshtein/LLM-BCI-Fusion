"""Build supplement eTables (CSV + Markdown) from output/results_digest.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.net_gain import orient_adaptive_minus_fixed  # noqa: E402

TAB_DIR = ROOT / "output" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "output" / "results_digest.json") as f:
    D = json.load(f)

with open(ROOT / "output" / "ladder_full_battery.json") as f:
    LADDER = json.load(f)


def write_md(name: str, header: list[str], rows: list[list[str]]):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    (TAB_DIR / f"{name}.md").write_text("\n".join(lines) + "\n")


def write_csv(name: str, header: list[str], rows: list[list[str]]):
    import csv
    with open(TAB_DIR / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, nd=4):
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-0.") and float(s) == 0 else s


def etable1_arm_rates():
    header = ["Prior", "Arm", "N", "Fused correct", "Neural correct", "Prior capture", "Phantom agreement", "Net correct gain"]
    rows = []
    for prior in ["gpt2", "ngram5_kn"]:
        a1 = D["aim1"][prior]
        for arm, rates in a1["arm_rates"].items():
            n = a1["arm_n"][arm]
            rows.append([
                prior, arm, str(n),
                fmt(rates["fused_correct"]), fmt(rates["neural_correct"]),
                fmt(rates["prior_capture"]), fmt(rates["phantom_agreement"]),
                fmt(rates["net_correct_gain"]),
            ])
    write_md("eTable1_arm_rates", header, rows)
    write_csv("eTable1_arm_rates", header, rows)


def etable2_contrasts():
    header = ["Prior", "Outcome", "Contrast", "Difference", "Lower 95% CI", "Upper 95% CI"]
    labels = {
        "net_correct_gain__adaptive_vs_fixed": ("Net correct gain", "Adaptive vs fixed"),
        "net_correct_gain__adaptive_vs_floor": ("Net correct gain", "Adaptive vs no-language-model floor"),
        "net_correct_gain__oracle_vs_fixed": ("Net correct gain", "Oracle vs fixed"),
        "bootstrap__prior_capture__adaptive_vs_fixed": ("Prior capture", "Adaptive vs fixed"),
        "bootstrap__phantom_agreement__adaptive_vs_fixed": ("Phantom agreement", "Adaptive vs fixed"),
        "bootstrap__fused_correct__adaptive_vs_fixed": ("Fused selection correctness", "Adaptive vs fixed"),
    }
    rows = []
    for prior in ["gpt2", "ngram5_kn"]:
        c = D["aim1"][prior]["contrasts"]
        for key, (outcome, contrast) in labels.items():
            v = c[key]
            rows.append([prior, outcome, contrast, fmt(v["point"]), fmt(v["lower"]), fmt(v["upper"])])
        gap = D["aim1"][prior]["oracle_gap_closure_fused_correct"]
        if gap["ok"]:
            rows.append([prior, "Fused selection correctness", "Oracle-to-fixed gap closed by adaptive", fmt(gap["value"]), "", ""])
    write_md("eTable2_bootstrap_contrasts", header, rows)
    write_csv("eTable2_bootstrap_contrasts", header, rows)


def etable3_gee():
    header = ["Prior", "Outcome", "Arm contrast (vs fixed)", "Odds ratio", "Lower 95% CI", "Upper 95% CI", "P value"]
    rows = []
    import math
    for prior in ["gpt2", "ngram5_kn"]:
        c = D["aim1"][prior]["contrasts"]
        for outcome in ["prior_capture", "phantom_agreement", "fused_correct"]:
            gee = c[f"gee__{outcome}"]
            if not gee["ok"]:
                continue
            for row_key, row in gee["value"].items():
                if "Intercept" in row_key:
                    continue
                arm = "adaptive" if "adaptive" in row_key else ("oracle" if "oracle" in row_key else "floor")
                or_ = math.exp(row["Coef."])
                lo = math.exp(row["[0.025"])
                hi = math.exp(row["0.975]"])
                p = row["P>|z|"]
                p_str = "<0.001" if p < 0.001 else fmt(p, 4)
                rows.append([prior, _OUTCOME_LABELS[outcome], arm, fmt(or_, 3), fmt(lo, 3), fmt(hi, 3), p_str])
    write_md("eTable3_gee_odds_ratios", header, rows)
    write_csv("eTable3_gee_odds_ratios", header, rows)


def etable4_ladder():
    # Digest stores fixed-minus-adaptive (from participant_cluster_bootstrap_rate_diff(..., "fixed",
    # "adaptive")); reported here as adaptive-minus-fixed (negated, with lower/upper swapped) so the
    # sign convention matches Figure 1/Figure 2 and the Results/Discussion prose throughout the paper:
    # prior_capture is a HARM outcome, so positive = adaptive causes more harm than fixed-weight fusion
    # (positive favors fixed).
    header = ["Prior model", "Adaptive minus fixed (prior capture)", "Lower 95% CI", "Upper 95% CI"]
    rows = []
    for r in D["aim4_ladder"]["prior_capture"]:
        rows.append([r["prior_model"], fmt(-r["point"]), fmt(-r["upper"]), fmt(-r["lower"])])
    write_md("eTable4_ladder_sweep", header, rows)
    write_csv("eTable4_ladder_sweep", header, rows)


def etable4b_ladder_fused_correct():
    # fused_correct is a BENEFIT outcome (higher = better), the opposite direction-sense from
    # prior_capture: here positive adaptive-minus-fixed means adaptive is MORE accurate (positive
    # favors adaptive). Same negation as etable4_ladder (digest stores fixed-minus-adaptive).
    header = ["Prior model", "Adaptive minus fixed (fused selection correctness)", "Lower 95% CI", "Upper 95% CI"]
    rows = []
    for r in D["aim4_ladder"]["fused_correct"]:
        rows.append([r["prior_model"], fmt(-r["point"]), fmt(-r["upper"]), fmt(-r["lower"])])
    write_md("eTable4b_ladder_sweep_fused_correct", header, rows)
    write_csv("eTable4b_ladder_sweep_fused_correct", header, rows)


def etable5_transport():
    # Same adaptive-minus-fixed convention as eTable4 (harm outcome, positive favors fixed).
    header = ["Prior", "Held-out study", "Adaptive minus fixed (prior capture)", "Lower 95% CI", "Upper 95% CI"]
    rows = []
    for prior in ["gpt2", "ngram5_kn"]:
        for r in D["aim3_transportability"][prior]["prior_capture"]:
            rows.append([prior, r["held_out_study"], fmt(-r["point"]), fmt(-r["upper"]), fmt(-r["lower"])])
    write_md("eTable5_transportability", header, rows)
    write_csv("eTable5_transportability", header, rows)


def etable5b_transport_fused_correct():
    # Same adaptive-minus-fixed convention as eTable4b (benefit outcome, positive favors adaptive).
    header = ["Prior", "Held-out study", "Adaptive minus fixed (fused selection correctness)", "Lower 95% CI", "Upper 95% CI"]
    rows = []
    for prior in ["gpt2", "ngram5_kn"]:
        for r in D["aim3_transportability"][prior]["fused_correct"]:
            rows.append([prior, r["held_out_study"], fmt(-r["point"]), fmt(-r["upper"]), fmt(-r["lower"])])
    write_md("eTable5b_transportability_fused_correct", header, rows)
    write_csv("eTable5b_transportability_fused_correct", header, rows)


def etable6_rescue():
    header = ["Prior", "Outcome", "Best-fixed rate", "Tuned-adaptive rate", "Tuned-adaptive minus best-fixed", "Lower 95% CI", "Upper 95% CI", "Chosen fixed betas", "Chosen anchors"]
    rows = []
    for prior in ["gpt2", "ngram5_kn"]:
        r = D["rescue"][prior]
        for outcome in ["prior_capture", "fused_correct"]:
            rr = r[outcome]
            rows.append([
                prior, _OUTCOME_LABELS[outcome], fmt(rr["best_fixed_rate"]), fmt(rr["tuned_adaptive_rate"]),
                fmt(-rr["rate_diff_point"], 6), fmt(-rr["rate_diff_upper"], 6), fmt(-rr["rate_diff_lower"], 6),
                str(rr["chosen_fixed_betas"]), str(rr["chosen_anchors"]),
            ])
        ng = r["net_gain_comparison"]
        rows.append([
            prior, "Net correct gain", fmt(ng["best_fixed_net_gain"]), fmt(ng["tuned_adaptive_net_gain"]),
            fmt(-ng["net_gain_diff_point"], 6), fmt(-ng["net_gain_diff_upper"], 6), fmt(-ng["net_gain_diff_lower"], 6),
            "", "",
        ])
    write_md("eTable6_rescue_comparison", header, rows)
    write_csv("eTable6_rescue_comparison", header, rows)


_OUTCOME_LABELS = {
    "prior_capture": "Prior capture",
    "phantom_agreement": "Phantom agreement",
    "fused_correct": "Fused selection correctness",
}


def _clean_bh_label(raw_label: str) -> tuple[str, str]:
    prior, outcome, _ = raw_label.split("__", 2)
    return prior, _OUTCOME_LABELS.get(outcome, outcome)


def etable7_bh():
    bh = D["bh_correction"]
    header = ["Prior", "Outcome", "Contrast", "Raw p", "BH-adjusted p"]
    rows = []
    for lbl, rp, ap in zip(bh["labels"], bh["raw_p"], bh["adjusted_p"]):
        prior, outcome = _clean_bh_label(lbl)
        rows.append([prior, outcome, "Adaptive vs fixed", fmt(rp, 4), fmt(ap, 4)])
    write_md("eTable7_bh_correction", header, rows)
    write_csv("eTable7_bh_correction", header, rows)


def etable15_ladder_fair_comparison():
    """All 22 evaluable priors under the fair, matched-search-space (rescue) comparison (Results:
    "No prior favored adaptive fusion"; Supplement S1.4) -- the 21-evaluation ladder extension
    (ladder_full_battery.json's fair_search_by_prior, which already includes a re-validation of
    ngram5_kn) merged with gpt2's own already-published rescue result (results_digest.json's
    "rescue" key, computed by the identical procedure -- Methods) so all 22 evaluable priors appear
    as one unique set. `net_gain_diff_*` in the stored results is best-fixed MINUS tuned-adaptive
    (the bootstrap is called with arm order best_fixed, tuned_adaptive), so the table converts it
    with orient_adaptive_minus_fixed; positive values favor adaptive."""
    header = [
        "Prior model", "Parameters", "Best-fixed net correct gain", "Tuned-adaptive net correct gain",
        "Adaptive minus fixed", "Lower 95% CI", "Upper 95% CI",
    ]
    rows = []
    fair = dict(LADDER["fair_search_by_prior"])
    fair["gpt2"] = {"net_gain_comparison": D["rescue"]["gpt2"]["net_gain_comparison"]}
    params_by_prior = {name: p["prior_parameters"] for name, p in LADDER["aim1_four_arm_by_prior"].items()}
    params_by_prior["gpt2"] = 124_000_000
    for name in sorted(fair, key=lambda n: params_by_prior[n]):
        ng = fair[name]["net_gain_comparison"]
        oriented = orient_adaptive_minus_fixed(ng)
        rows.append([
            name, str(params_by_prior[name]),
            fmt(ng["best_fixed_net_gain"]), fmt(ng["tuned_adaptive_net_gain"]),
            fmt(oriented["adaptive_minus_fixed_point"]), fmt(oriented["adaptive_minus_fixed_lower"]),
            fmt(oriented["adaptive_minus_fixed_upper"]),
        ])
    write_md("eTable15_ladder_fair_comparison", header, rows)
    write_csv("eTable15_ladder_fair_comparison", header, rows)


if __name__ == "__main__":
    etable1_arm_rates()
    etable2_contrasts()
    etable3_gee()
    etable4_ladder()
    etable4b_ladder_fused_correct()
    etable5_transport()
    etable5b_transport_fused_correct()
    etable6_rescue()
    etable7_bh()
    # eTable 15 onward are built from the revision analyses by build_revision_tables.py
    print("Wrote eTable1-7 (+4b, 5b) to", TAB_DIR)
