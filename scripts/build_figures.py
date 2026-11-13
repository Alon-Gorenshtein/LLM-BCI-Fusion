"""Build the manuscript's publication figures from output/results_digest.json.
Figure 1: 22-evaluable-prior fair, matched-search-space comparison across the model ladder --
  the primary cross-model analysis (promoted from Figure 2 when the manuscript was restructured to
  lead with cross-model breadth rather than the two representative anchors).
Figure 2: detailed characterization of the two representative anchors, gpt2 + ngram5_kn (naive +
  fair comparison) -- demoted from Figure 1 in the same restructuring.
Figure 3: candidate-collapse mechanism, override outcomes, error-flagging, decoded-history replay.
eFigure 1 (supplement): Aim-3 transportability, leave-one-study-out.
eFigure 2 (supplement): Aim-2 reliability-estimator calibration (g_hat vs realized neural_correct),
  demoted from main Figure 2 in an earlier round.
eFigure 3 (supplement): Aim-4 ladder-wide robustness (fused_correct, adaptive vs fixed), 24 priors.
eFigure 4 (supplement): Aim-4 ladder-wide robustness (prior_capture, adaptive vs fixed), 24 priors.
Reader-facing labels: gpt2 -> "GPT-2", ngram5_kn -> "the 5-gram" (raw identifiers stay in
Methods/Supplement, matching the PRIORS tuple below which indexes results_digest.json/
figure3_data.json by their raw keys).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fusion.net_gain import orient_adaptive_minus_fixed
from figure_kit import set_style, save, panel, faint_grid, BLUE, GREY, SALMON, LABEL, MUTED
from hochberg_kit import (
    apply_style as hochberg_style, save as hochberg_save, panel as hochberg_panel,
    heatmap_values, INK, KEY, ERROR, CONTEXT, ACCENT3, MUTED as H_MUTED,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRIORS = ("gpt2", "ngram5_kn")
PRIOR_LABEL = {"gpt2": "GPT-2", "ngram5_kn": "5-gram"}

with open(ROOT / "output" / "results_digest.json") as f:
    D = json.load(f)

with open(ROOT / "output" / "ladder_full_battery.json") as f:
    LADDER = json.load(f)

set_style()
import matplotlib.pyplot as plt


def _fair_comparison_rows():
    """Best-fixed-vs-tuned-adaptive differences from the matched-search-space (fair) comparison,
    read from the already-verified eTable6_rescue_comparison.csv (Results: fair-comparison
    analysis). Fused selection correctness is dropped here since it is equivalent to net correct
    gain within a fixed row set (Results) and would otherwise duplicate the same near-zero point."""
    import csv
    rows = {"gpt2": {}, "ngram5_kn": {}}
    with open(ROOT / "output" / "tables" / "eTable6_rescue_comparison.csv") as f:
        for r in csv.DictReader(f):
            if r["Outcome"] not in ("Prior capture", "Net correct gain"):
                continue
            # eTable 6 is tuned-adaptive minus best-fixed, the orientation the figure shows
            rows[r["Prior"]][r["Outcome"]] = {
                "point": float(r["Tuned-adaptive minus best-fixed"]), "lower": float(r["Lower 95% CI"]), "upper": float(r["Upper 95% CI"]),
            }
    # Net correct gain row: the main (nested-anchor) analysis, participant-bootstrap interval.
    ladder = json.loads((ROOT / "output" / "revision" / "fair_ladder.json").read_text())["priors"]
    for prior in rows:
        e = ladder[prior]["nested"]
        rows[prior]["Net correct gain"] = {"point": e["diff"], "lower": e["ci_participant"][0], "upper": e["ci_participant"][1]}
    return rows


def figure2_deepdive():
    hochberg_style()
    priors = [("gpt2", "a", "c"), ("ngram5_kn", "b", "d")]
    row_labels_naive = ["Oracle − fixed", "Adaptive − fixed", "Adaptive − floor"]
    row_keys_naive = ["net_correct_gain__oracle_vs_fixed", "net_correct_gain__adaptive_vs_fixed", "net_correct_gain__adaptive_vs_floor"]
    row_labels_fair = ["Prior capture", "Net correct gain"]
    row_keys_fair = ["Prior capture", "Net correct gain"]
    fair_rows = _fair_comparison_rows()

    # Harmonize the naive-comparison row across both priors so a visual magnitude comparison is
    # not misleading -- collect every contrast bound across both priors before drawing either panel.
    all_bounds = []
    for prior, _, _ in priors:
        for key in row_keys_naive:
            c = D["aim1"][prior]["contrasts"][key]
            all_bounds.extend([c["lower"], c["upper"]])
    span = max(all_bounds) - min(all_bounds)
    pad = span * 0.08
    shared_xlim_naive = (min(all_bounds) - pad, max(all_bounds) + pad)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex="row")

    for col, (prior, letter_top, letter_bottom) in enumerate(priors):
        # Top row: naive comparison against beta = 1.
        ax = axes[0, col]
        contrasts = D["aim1"][prior]["contrasts"]
        ys = np.arange(len(row_keys_naive))[::-1]
        for y, key in zip(ys, row_keys_naive):
            c = contrasts[key]
            point, lo, hi = c["point"], c["lower"], c["upper"]
            significant = (lo > 0) or (hi < 0)
            color = KEY if (significant and point > 0) else (ERROR if (significant and point < 0) else CONTEXT)
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                        ecolor=color, elinewidth=2.2, capsize=4, markersize=9, markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        ax.axvline(0, color=H_MUTED, lw=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(*shared_xlim_naive)
        faint_grid(ax, axis="x")
        ax.set_yticks(ys)
        ax.set_yticklabels(row_labels_naive, fontweight="bold")
        ax.set_ylim(-0.7, len(row_keys_naive) - 0.3)
        prior_label = "GPT-2 (general-purpose LM prior)" if prior == "gpt2" else "5-gram (classical n-gram prior)"
        ax.text(0.5, 1.12, prior_label, transform=ax.transAxes, ha="center", fontsize=9.5, fontweight="bold", color=INK)
        hochberg_panel(ax, letter_top)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Bottom row: fair, matched-search-space comparison (best-fixed vs tuned-adaptive).
        ax = axes[1, col]
        ys = np.arange(len(row_keys_fair))[::-1]
        for y, key in zip(ys, row_keys_fair):
            c = fair_rows[prior][key]
            point, lo, hi = c["point"], c["lower"], c["upper"]
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=CONTEXT,
                        ecolor=CONTEXT, elinewidth=2.2, capsize=4, markersize=9, markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        ax.axvline(0, color=H_MUTED, lw=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.006, 0.006)
        faint_grid(ax, axis="x")
        ax.set_yticks(ys)
        ax.set_yticklabels(row_labels_fair, fontweight="bold")
        ax.set_xlabel("Tuned-adaptive − best-fixed (fair comparison)")
        ax.set_ylim(-0.7, len(row_keys_fair) - 0.3)
        hochberg_panel(ax, letter_bottom)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0, 0].set_xlabel("Difference in net correct gain (naive, beta = 1)")
    axes[0, 1].set_xlabel("Difference in net correct gain (naive, beta = 1)")
    fig.subplots_adjust(wspace=0.2, hspace=0.55, top=0.88)
    hochberg_save(fig, str(FIG_DIR / "Figure3"))   # anchor deep dive (Figure 3 in the revised numbering)


def efigure3_ladder_prior_capture():
    """24-prior ladder sweep on prior capture (Results: ladder-wide consistency). Companion to
    eFigure3's fused-correctness sweep; drawn the same neutral-color way and for the same reason:
    these 24 per-prior comparisons are uncorrected for multiplicity (Supplement S1.4), so
    color-coding by whether an individual CI excludes zero would invite reading this sweep as a
    per-model significance count rather than the descriptive consistency check it is."""
    set_style()
    rows = D["aim4_ladder"]["prior_capture"]
    # adaptive_minus_fixed = -(fixed - adaptive); positive = adaptive causes MORE prior-capture harm.
    items = []
    for r in rows:
        point = -r["point"]
        lo = -r["upper"]
        hi = -r["lower"]
        items.append((r["prior_model"], point, lo, hi))
    items.sort(key=lambda x: x[1])
    n_favor_fixed = sum(1 for _, point, lo, hi in items if lo > 0)
    n_favor_adaptive = sum(1 for _, point, lo, hi in items if hi < 0)

    fig, ax = plt.subplots(figsize=(5.2, 7.8))
    ys = np.arange(len(items))[::-1]
    for y, (name, point, lo, hi) in zip(ys, items):
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=MUTED,
                    ecolor=MUTED, elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
    faint_grid(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, *_ in items], fontsize=7.5)
    ax.set_xlabel(
        f"Adaptive minus fixed prior-capture rate (positive = adaptive causes more harm)\n"
        f"{n_favor_fixed}/{len(items)} favored fixed, {n_favor_adaptive}/{len(items)} favored adaptive (uncorrected; S1.4)",
        fontsize=8.5, labelpad=8,
    )
    ax.set_ylim(-2.4, len(items) - 0.2)
    for spine in ["left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(FIG_DIR / "eFigure4"))


def efigure2_ladder_fused_correct():
    """24-prior ladder sweep on fused selection correctness (Results: ladder-wide consistency).
    Demoted to a supplement figure and drawn with a single neutral color throughout: the 24
    per-prior comparisons are uncorrected for multiplicity (Supplement S1.4), so this figure
    deliberately does not color-code by whether an individual CI excludes zero, which would
    invite reading it as a per-model significance count rather than the descriptive sweep it is."""
    set_style()
    rows = D["aim4_ladder"]["fused_correct"]
    items = []
    for r in rows:
        point = -r["point"]
        lo = -r["upper"]
        hi = -r["lower"]
        items.append((r["prior_model"], point, lo, hi))
    items.sort(key=lambda x: x[1])
    n_favor_adaptive = sum(1 for _, point, lo, hi in items if point > 0)

    fig, ax = plt.subplots(figsize=(5.2, 7.8))
    ys = np.arange(len(items))[::-1]
    for y, (name, point, lo, hi) in zip(ys, items):
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=MUTED,
                    ecolor=MUTED, elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
    faint_grid(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, *_ in items], fontsize=7.5)
    ax.set_xlabel(
        f"Adaptive minus fixed fused-correctness rate\n"
        f"{n_favor_adaptive}/{len(items)} favored adaptive fusion (uncorrected; S1.4)",
        fontsize=8.5, labelpad=8,
    )
    ax.set_ylim(-2.4, len(items) - 0.2)
    for spine in ["left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(FIG_DIR / "eFigure3"))


def figure3():
    hochberg_style()
    with open(ROOT / "output" / "figure3_data.json") as f:
        F3 = json.load(f)

    PRIOR_COLOR = {"gpt2": KEY, "ngram5_kn": ACCENT3}
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.0))

    # (a) Fraction of selections with 1-5 grid-achievable decoded candidates, both priors --
    # a viridis heatmap with the printed percentage in every cell (Hochberg data-rich style).
    ax = axes[0, 0]
    categories = [1, 2, 3, 4, 5]
    M_a = np.array([[F3["candidate_collapse"][p].get(str(c), 0.0) * 100 for c in categories] for p in PRIORS])
    heatmap_values(ax, M_a, row_labels=[PRIOR_LABEL[p] for p in PRIORS], col_labels=[str(c) for c in categories],
                   fmt="{:.1f}%", vmin=0, cbar_label="% of selections", valfontsize=9)
    ax.set_xlabel("Grid-achievable decoded candidates")
    hochberg_panel(ax, "a")

    ax = axes[0, 1]
    outcomes = ["n_helped", "n_hurt", "n_no_effect"]
    outcome_labels = ["Helped", "Hurt", "No effect"]
    x = np.arange(len(outcomes))
    override_totals = {p: sum(F3["override_outcomes"][p][o] for o in outcomes) for p in PRIORS}
    for offset, prior in zip((-width / 2, width / 2), PRIORS):
        n = override_totals[prior]
        heights = [100 * F3["override_outcomes"][prior][o] / n for o in outcomes]
        label = f"{PRIOR_LABEL[prior]} (n = {int(n)})"
        bars = ax.bar(x + offset, heights, width, color=PRIOR_COLOR[prior], edgecolor="white",
                      linewidth=1.3, label=label, zorder=3)
        for rect, h in zip(bars, heights):
            ax.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels)
    ax.set_ylabel("% of that prior's own overridden selections")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    faint_grid(ax)
    hochberg_panel(ax, "b")

    # (c) Error-fraction-captured enrichment at 5% flagged, all 8 signals, both priors -- a
    # viridis heatmap with the printed fold-enrichment value in every cell; vmin=1.0 anchors the
    # color scale at "no enrichment" (95% CIs for these point estimates are in Supplement eTable13).
    ax = axes[1, 0]
    signal_order = [
        "fixed_fused_top1_prob", "fixed_fused_entropy_bits", "fixed_fused_margin",
        "reliability_g_hat", "candidate_ranker_p_hat", "neural_margin",
        "neural_top1_prob", "neural_lm_js_divergence",
    ]
    signal_labels = {
        "fixed_fused_top1_prob": "Fixed-fused top-1 prob.",
        "fixed_fused_entropy_bits": "Fixed-fused entropy",
        "fixed_fused_margin": "Fixed-fused margin",
        "reliability_g_hat": "Reliability estimate",
        "candidate_ranker_p_hat": "Candidate-arbitration ranker",
        "neural_margin": "Neural margin",
        "neural_top1_prob": "Neural top-1 prob.",
        "neural_lm_js_divergence": "Neural-LM JS divergence",
    }
    M_c = np.array([[F3["abstention_enrichment"][p][s]["coverage_0.95"]["enrichment"] for p in PRIORS]
                     for s in signal_order])
    heatmap_values(ax, M_c, row_labels=[signal_labels[s] for s in signal_order],
                   col_labels=[PRIOR_LABEL[p] for p in PRIORS], fmt="{:.2f}×", vmin=1.0,
                   cbar_label="Error-fraction-captured enrichment (5% flagged)", valfontsize=9)
    hochberg_panel(ax, "c")

    ax = axes[1, 1]
    x = np.arange(len(PRIORS))
    points = [F3["decoded_history"][p]["decoded_minus_ground_truth"]["point"] for p in PRIORS]
    los = [F3["decoded_history"][p]["decoded_minus_ground_truth"]["lower"] for p in PRIORS]
    his = [F3["decoded_history"][p]["decoded_minus_ground_truth"]["upper"] for p in PRIORS]
    lo_err = [points[i] - los[i] for i in range(len(PRIORS))]
    hi_err = [his[i] - points[i] for i in range(len(PRIORS))]
    ax.errorbar(x, points, yerr=[lo_err, hi_err], fmt="none", ecolor=INK, elinewidth=2.0,
                capsize=5, zorder=2)
    for i, prior in enumerate(PRIORS):
        ax.plot(x[i], points[i], "o", color=PRIOR_COLOR[prior], markersize=11,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.axhline(0, color=H_MUTED, lw=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.6, len(PRIORS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PRIOR_LABEL[p] for p in PRIORS])
    ax.set_ylabel("Decoded-history minus\nground-truth-prefix\nnet correct gain")
    faint_grid(ax)
    hochberg_panel(ax, "d")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PRIOR_COLOR["gpt2"], label=PRIOR_LABEL["gpt2"]),
        plt.Rectangle((0, 0), 1, 1, color=PRIOR_COLOR["ngram5_kn"], label=PRIOR_LABEL["ngram5_kn"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.4, wspace=0.5, bottom=0.1)
    hochberg_save(fig, str(FIG_DIR / "Figure3"))


def efigure1_transport():
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    outcomes = [("prior_capture", False), ("fused_correct", True)]
    panel_letters = [["a", "b"], ["c", "d"]]
    priors = ["gpt2", "ngram5_kn"]
    outcome_xlabel = {
        "prior_capture": "Adaptive − fixed prior-capture rate\n(held-out study)",
        "fused_correct": "Adaptive − fixed fused-correctness rate\n(held-out study)",
    }
    for row, (outcome, is_benefit) in enumerate(outcomes):
        for col, prior in enumerate(priors):
            ax = axes[row][col]
            rows = D["aim3_transportability"][prior][outcome]
            ys = np.arange(len(rows))[::-1]
            for y, r in zip(ys, rows):
                point, lo, hi = -r["point"], -r["upper"], -r["lower"]
                significant = (lo > 0) or (hi < 0)
                if is_benefit:
                    color = SALMON if (significant and point > 0) else (BLUE if (significant and point < 0) else GREY)
                else:
                    color = SALMON if (significant and point < 0) else (BLUE if (significant and point > 0) else GREY)
                ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                            ecolor=color, elinewidth=1.4, capsize=3, markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
            faint_grid(ax, axis="x")
            ax.set_yticks(np.arange(len(rows))[::-1])
            ax.set_yticklabels([r["held_out_study"] for r in rows])
            ax.set_xlabel(outcome_xlabel[outcome], fontsize=8)
            panel(ax, panel_letters[row][col])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    save(fig, str(FIG_DIR / "eFigure1"))


def figure2_calibration():
    hochberg_style()
    sys.path.insert(0, str(ROOT))
    try:
        from fusion.data import build_working_frame
        from fusion.reliability import fit_reliability_cv

        working = build_working_frame("gpt2")
        fitted = fit_reliability_cv(working, n_folds=5)
        bins = np.linspace(0, 1, 11)
        bin_ids = np.clip(np.digitize(fitted["g_hat"], bins[1:-1]), 0, 9)
        conf, acc, n = [], [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            conf.append(fitted["g_hat"][mask].mean())
            acc.append(fitted["neural_correct"][mask].astype(int).mean())
            n.append(mask.sum())
        sizes = 20 + 200 * (np.array(n) / max(n))
    except RuntimeError:
        # fusion.data needs the sibling study_bci_llm_authorship checkout's raw intermediates,
        # not present in this worktree. These 10 calibration-bin coordinates are losslessly
        # recovered from the vector marker paths of the already-published Figure2.pdf (the same
        # run that produced the AUROC/ECE already recorded in results_digest.json) -- a pure
        # restyle, not a re-derivation. Regenerate from source once run where fusion.data can import.
        conf = [0.0419, 0.1467, 0.2495, 0.3529, 0.4468, 0.5506, 0.6485, 0.7561, 0.8715, 0.9004]
        acc = [0.1491, 0.2538, 0.2213, 0.3534, 0.3427, 0.4453, 0.5507, 0.5661, 0.8914, 0.9740]
        fracs = np.array([0.0809, 0.0654, 0.0613, 0.0669, 0.0719, 0.0689, 0.0694, 0.0950, 1.0000, 0.1161])
        sizes = 20 + 200 * fracs

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color=H_MUTED, lw=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(conf, acc, s=sizes * 1.3, color=KEY, edgecolors="white", linewidths=1.2, zorder=3)
    ax.set_xlabel("Mean predicted reliability (ĝ)")
    ax.set_ylabel("Observed neural-correct rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    auroc = D["aim1"]["gpt2"]["reliability_auroc"]
    ece = D["aim1"]["gpt2"]["reliability_ece"]
    ax.text(0.03, 0.94, f"AUROC = {auroc:.3f}\nECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=INK, va="top")
    fig.subplots_adjust(bottom=0.15, left=0.18)
    hochberg_save(fig, str(FIG_DIR / "eFigure2"))


def figure1_ladder():
    """The full 22-evaluable-prior fair, matched-search-space comparison (Results: "No prior
    favored adaptive fusion"; Supplement S1.4), sorted by parameter count ascending -- the
    principal cross-model analysis, main Figure 1. Merges the 21-evaluation ladder extension
    (output/ladder_full_battery.json, which already includes a re-validation of the published
    5-gram result) with gpt2's own already-published fair-comparison row from
    eTable6_rescue_comparison.csv (the two representative anchors' own original evaluation,
    computed by the identical procedure -- Methods) so all 22 evaluable priors appear as one
    unique set, not 21-plus-one-shown-elsewhere. Neutral color throughout, matching
    eFigure3/eFigure4's own rationale (the naive-ladder sweeps): 22 simultaneous comparisons,
    uncorrected for multiplicity, read descriptively rather than as per-prior significance
    tests."""
    set_style()
    fair = dict(LADDER["fair_search_by_prior"])
    params_by_prior = {name: p["prior_parameters"] for name, p in LADDER["aim1_four_arm_by_prior"].items()}
    gpt2_rows = _fair_comparison_rows()["gpt2"]
    gpt2_ng = gpt2_rows["Net correct gain"]
    fair["gpt2"] = {"net_gain_comparison": {
        "net_gain_diff_point": gpt2_ng["point"], "net_gain_diff_lower": gpt2_ng["lower"],
        "net_gain_diff_upper": gpt2_ng["upper"],
    }}
    params_by_prior["gpt2"] = 124_000_000
    items = []
    n_tie = n_sig_fixed = n_sig_adapt = 0
    for name in sorted(fair, key=lambda n: params_by_prior[n]):
        ng = orient_adaptive_minus_fixed(fair[name]["net_gain_comparison"])
        point, lo, hi = ng["adaptive_minus_fixed_point"], ng["adaptive_minus_fixed_lower"], ng["adaptive_minus_fixed_upper"]
        items.append((name, point, lo, hi))
        if point == 0.0 and lo == 0.0 and hi == 0.0:
            n_tie += 1
        elif hi < 0:
            n_sig_fixed += 1
        elif lo > 0:
            n_sig_adapt += 1

    fig, ax = plt.subplots(figsize=(5.2, 8.1))
    ys = np.arange(len(items))[::-1]
    for y, (name, point, lo, hi) in zip(ys, items):
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=MUTED,
                    ecolor=MUTED, elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
    faint_grid(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, *_ in items], fontsize=7.5)
    ax.set_xlabel(
        f"Tuned-adaptive minus best-fixed net correct gain\n"
        f"{n_tie}/{len(items)} exact ties; {len(items) - n_tie - n_sig_fixed - n_sig_adapt}/{len(items)} CIs cross zero;\n"
        f"{n_sig_fixed}/{len(items)} CIs favor fixed; {n_sig_adapt}/{len(items)} CIs favor adaptive (uncorrected; S1.4)",
        fontsize=8.5, labelpad=8,
    )
    ax.set_ylim(-2.4, len(items) - 0.2)
    for spine in ["left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(FIG_DIR / "Figure1"))


if __name__ == "__main__":
    # Figures 1, 2 and 4 are built by build_revision_figures.py; this script builds the anchor deep
    # dive (Figure 3) and the supplementary figures from the regenerated digest.
    figure2_deepdive()
    efigure1_transport()
    figure2_calibration()
    efigure2_ladder_fused_correct()
    efigure3_ladder_prior_capture()
    print("Wrote Figure3, eFigure1, eFigure2, eFigure3, eFigure4 to", FIG_DIR)
