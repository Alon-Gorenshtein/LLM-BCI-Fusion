"""Markdown supplementary tables for the revision, built from output/revision/*.json.

Writes output/revision/revision_etables.md (eTable 15 to 22) and one .csv per table. Every number comes
from the JSON files written by the run_revision_* scripts, so the tables cannot drift from the digest.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "output" / "revision"


def load(name):
    path = REV / name
    return json.loads(path.read_text()) if path.exists() else None


def f(x, d=4):
    return "" if x is None else f"{x:.{d}f}"


def pm(x):
    return "" if x is None else f"{x:+.4f}"


def ci(pair, d=4):
    return f"{pair[0]:+.{d}f} to {pair[1]:+.{d}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def params(n):
    return "0" if n == 0 else (f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M")


def etable15_16_17(out):
    d = load("fair_ladder.json")
    if d is None:
        return
    names = sorted(d["priors"], key=lambda n: d["priors"][n]["parameters"])
    s_nested, s_in = d["summary"]["nested"], d["summary"]["insample"]
    rows15 = []
    for n in names:
        e = d["priors"][n]
        ne, ins = e["nested"], e["insample"]
        rows15.append([n, params(e["parameters"]), f(e["mean_fixed"]), f(ins["mean_adaptive"]), f(ne["mean_adaptive"]),
                       pm(ne["diff"]), ci(ne["ci_participant"]), ci(ne["ci_context"]), ci(ne["ci_crossed"]),
                       ci(s_nested["simultaneous_band"][n]),
                       f"{ne['n_adaptive_better']} / {ne['n_fixed_better']}",
                       "yes" if ne["identical_on_every_selection"] else "no"])
    h15 = ["Prior", "Parameters", "Best-fixed net correct gain", "Tuned-adaptive, in-sample anchor", "Tuned-adaptive, nested anchor",
           "Adaptive minus fixed (nested)", "95% CI, participants", "95% CI, contexts", "95% CI, crossed",
           "Simultaneous 95% band", "Selections adaptive better / fixed better", "Identical on every selection"]
    out.append(("eTable 15. Fair comparison for all 24 priors: tuned-adaptive minus best-fixed net correct gain (positive favours adaptive). "
                "The nested column is the main analysis; the in-sample column selects the anchor on the training rows' own reliability estimates.",
                h15, rows15))

    rows16 = []
    for n in names:
        e = d["priors"][n]
        fx = ", ".join(f"{x['fixed_beta']:g}" for x in e["folds_nested"])
        an = ", ".join(f"({x['anchor'][0]:g}, {x['anchor'][1]:g})" for x in e["folds_nested"])
        rows16.append([n, fx, an, f(e["nested"]["share_folds_constant_weight"], 2), f(e["insample"]["share_folds_constant_weight"], 2),
                       f(e["share_folds_same_anchor_insample_vs_nested"], 2),
                       f"{e['insample']['diff'] - e['nested']['diff']:+.4f}"])
    out.append(("eTable 16. Weights and anchors chosen in each of the five folds (nested selection), share of folds that chose a constant weight, "
                "share of folds in which in-sample and nested selection chose the same anchor, and the difference in the adaptive-minus-fixed estimate between the two procedures.",
                ["Prior", "Best-fixed beta per fold", "Anchor per fold (nested)", "Share of folds with a constant weight (nested)",
                 "Share with a constant weight (in-sample)", "Share of folds with the same anchor in both procedures",
                 "In-sample minus nested estimate"], rows16))

    rows17 = []
    for label, s in (("Nested (main)", s_nested), ("In-sample", s_in)):
        for scheme in ("participant", "crossed"):
            c = s["counts"][scheme]
            eq = c["equivalence"]
            key = "ci_" + scheme
            lab = "nested" if label.startswith("Nested") else "insample"
            non = [e for e in d["priors"].values() if not e[lab]["identical_on_every_selection"]]
            inside = lambda m: sum(1 for e in non if e[lab][key][0] > -m and e[lab][key][1] < m)
            rows17.append([label, scheme, c["identical_on_every_selection"], c["ci_crosses_zero"], c["ci_favors_adaptive"], c["ci_favors_fixed"],
                           f"{eq['0.005']['inside_margin']}", f"{eq['0.010']['inside_margin']}", f"{eq['0.020']['inside_margin']}",
                           f"{inside(0.005)} / {inside(0.010)} / {inside(0.020)} of {len(non)}"])
    pooled = [[lab, f(s["pooled_mean_diff"]), ci(s["pooled_mean_diff_ci_crossed"]), f(s["simultaneous_multiplier"], 2),
               ", ".join(s["simultaneous_band_excludes_zero"]) or "none"] for lab, s in (("Nested (main)", s_nested), ("In-sample", s_in))]
    out.append(("eTable 17a. Counts of the 24 intervals by what they show, and the number of intervals inside the equivalence margin "
                "(0.5, 1.0 and 2.0 percentage points of net correct gain; the 9 identical priors count as inside by construction, so the last column counts only the priors where the arms differ).",
                ["Procedure", "Interval", "Identical on every selection", "Cross zero", "Favour adaptive", "Favour fixed",
                 "Inside +/-0.5 pp (all 24)", "Inside +/-1.0 pp (all 24)", "Inside +/-2.0 pp (all 24)", "Inside +/-0.5 / 1.0 / 2.0 pp among priors where the arms differ"], rows17))
    out.append(("eTable 17b. Pooled mean difference across the 24 priors (crossed 95% interval), multiplier of the simultaneous band, and priors whose simultaneous band excludes zero.",
                ["Procedure", "Pooled mean difference", "95% CI (crossed)", "Band multiplier", "Priors whose band excludes zero"], pooled))


def etable18(out):
    d = load("envelope.json")
    if d is None:
        return
    pri = d["priors"]
    ranges = ("grid5", "default_adaptive", "widest_anchor", "full")
    meta = load("fair_ladder.json")
    names = sorted(pri, key=lambda n: (meta["priors"][n]["parameters"] if meta else 0, n))
    rows = []
    for n in names:
        rows.append([n] + [f(pri[n][r]["share_1_reachable"], 3) for r in ranges] + [f(pri[n][r]["headroom"], 3) for r in ranges])
    out.append(("eTable 18. Reachable symbols by weight range. Left: share of selections with exactly one reachable symbol. Right: tie-free ceiling "
                "(share of selections whose target is reachable) minus the fused accuracy of the best-fixed arm. Ranges: grid = five grid weights; default = 0.25 to 4; "
                "widest = 0.1 to 20; full = language-model-only to neural-only. The candidate set always includes the neural-only argmax.",
                ["Prior", "One symbol: grid", "default", "widest", "full", "Headroom: grid", "default", "widest", "full"], rows))


def etable19(out):
    d = load("reliability_baselines.json")
    if d is None:
        return
    rows = [[k, f(v["auroc"], 3), f(v.get("brier"), 3), f(v.get("ece"), 3)] for k, v in d["metrics"].items()]
    rows2 = [[k, f"{v['auroc_three_feature_minus_this']:+.4f}", ci([v["lower"], v["upper"]])] for k, v in d["auroc_contrasts"].items()]
    fs = d["three_feature_auroc_across_random_fold_assignments"]
    out.append(("eTable 19a. Out-of-fold discrimination of whether the neural argmax is correct (neural posterior only; identical across priors). Raw rows use the feature directly as a score; logistic rows fit that one feature.",
                ["Score", "AUROC", "Brier score", "Expected calibration error"], rows))
    out.append((f"eTable 19b. AUROC of the three-feature estimator minus each baseline (95% participant-bootstrap interval). Across {fs['n_assignments']} random participant-fold assignments the three-feature AUROC averaged {fs['mean']:.3f} (range {fs['min']:.3f} to {fs['max']:.3f}).",
                ["Baseline", "Three-feature minus baseline", "95% CI"], rows2))


def etable20(out):
    d = load("selective.json")
    if d is None:
        return
    rows = []
    for prior, p in d["priors"].items():
        for sig, v in {**p["signals"], **{k: {"aurc": r["aurc"], "excess_aurc": r["excess_aurc"]} for k, r in p["references"].items()}}.items():
            vr = v.get("vs_reference")
            diff = None if not vr else {**vr["aurc_difference_vs_reference"], "holm_p": vr["p_holm"]}
            f5, f10 = (v.get("flagged", {}).get("5%"), v.get("flagged", {}).get("10%"))
            def fl(x):
                if not x:
                    return ""
                c = x["error_fraction_captured"]
                return f"{x['errors_among_flagged']}/{x['errors_total']} ({100*c['point']:.1f}%; {100*c['lower']:.1f} to {100*c['upper']:.1f}); {x['enrichment']['point']:.2f}x"
            rows.append([prior, sig, f"{v['aurc']['point']:.4f} ({v['aurc']['interval'][0]:.4f} to {v['aurc']['interval'][1]:.4f})",
                         f"{v['excess_aurc']['point']:.4f}",
                         "" if not diff else f"{diff['point']:+.4f} ({diff['interval'][0]:+.4f} to {diff['interval'][1]:+.4f}); Holm P {diff['holm_p']:.3f}",
                         fl(f5), fl(f10)])
    out.append(("eTable 20. Selective prediction for the fixed-weight fused system's errors, ranked within each held-out study. AURC (lower is better) with 95% participant-bootstrap "
                "intervals, excess over the best possible ranking, paired difference from the fused top-1 probability (Holm-adjusted across seven contrasts), and errors captured when the "
                "least confident 5% and 10% of selections are flagged (errors among flagged selections / errors in total, share of errors with 95% CI, enrichment).",
                ["Prior", "Signal", "AURC (95% CI)", "Excess AURC", "Difference from fused top-1 probability", "5% flagged", "10% flagged"], rows))


def etable21(out):
    rows = []
    for prior in ("gpt2", "ngram5_kn"):
        d = load(f"decoded_fair_{prior}.json")
        if d is None:
            continue
        rows.append([prior, f(d["mean_fixed_net_gain"]), f(d["mean_adaptive_net_gain"]), pm(d["diff_adaptive_minus_fixed"]),
                     ci(d["ci_participant"]), ci(d["ci_crossed"]), f"{d['n_adaptive_better']} / {d['n_fixed_better']}",
                     "; ".join(f"{x['fixed_beta']:g} / ({x['anchor'][0]:g}, {x['anchor'][1]:g})" for x in d["folds"])])
    if rows:
        out.append(("eTable 21. Fair comparison with the language model conditioned on the decoder's own earlier output. Both arms are tuned inside this setting. "
                    "Differences are adaptive minus fixed net correct gain.",
                    ["Prior", "Best-fixed net correct gain", "Tuned-adaptive net correct gain", "Adaptive minus fixed", "95% CI, participants",
                     "95% CI, crossed", "Selections adaptive better / fixed better", "Weight / anchor chosen in each fold"], rows))


def etable22(out):
    d = load("prior_audit.json")
    if d is None:
        return
    rows = [[r["prior"], f(r["R1_agreement"]), f(r["R1_net_gain_beta1"]), f(r["August_agreement"]), f(r["August_net_gain_beta1"]),
             f(r["share_same_fused_argmax_between_copies"])] for r in d["rows"]]
    out.append(("eTable 22. Consistency of the stored posteriors in two copies of the companion data. Agreement is the share of selections whose fused symbol at beta = 1, recomputed from the stored neural posterior and "
                "prior shard, equals the stored fused symbol. The last column is the share of selections with the same fused symbol in the two copies. The current copy is the one used throughout.",
                ["Prior", "Current copy: recomputed equals stored", "Current copy: net correct gain at beta = 1", "Earlier copy: recomputed equals stored",
                 "Earlier copy: net correct gain at beta = 1", "Same fused symbol in both copies"], rows))


def main():
    out: list = []
    for fn in (etable15_16_17, etable18, etable19, etable20, etable21, etable22):
        fn(out)
    text = "\n\n".join(f"**{cap}**\n\n{md_table(h, r)}" for cap, h, r in out)
    (REV / "revision_etables.md").write_text(text + "\n")
    for i, (cap, h, r) in enumerate(out, start=1):
        with open(REV / f"revision_etable_{i:02d}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(h)
            w.writerows(r)
    print(f"wrote {len(out)} tables to {REV / 'revision_etables.md'}")


if __name__ == "__main__":
    main()
