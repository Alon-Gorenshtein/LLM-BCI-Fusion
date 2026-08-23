"""Build the manuscript's publication figures from output/results_digest.json.
Figure 1: Aim-1 primary contrast (net correct gain), gpt2 + ngram5_kn.
Figure 2: Aim-4 ladder-wide robustness (prior_capture, adaptive vs fixed), 24 priors.
eFigure 1 (supplement): Aim-3 transportability, leave-one-study-out.
eFigure 2 (supplement): Aim-2 reliability-estimator calibration (ĝ vs realized neural_correct).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from figure_kit import set_style, save, panel, faint_grid, BLUE, GREY, SALMON, LABEL, MUTED

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "output" / "results_digest.json") as f:
    D = json.load(f)

set_style()
import matplotlib.pyplot as plt


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    priors = [("gpt2", "a"), ("ngram5_kn", "b")]
    row_labels = ["Oracle − fixed", "Adaptive − fixed", "Adaptive − floor"]
    row_keys = ["net_correct_gain__oracle_vs_fixed", "net_correct_gain__adaptive_vs_fixed", "net_correct_gain__adaptive_vs_floor"]
    for ax, (prior, letter) in zip(axes, priors):
        contrasts = D["aim1"][prior]["contrasts"]
        ys = np.arange(len(row_keys))[::-1]
        for y, key in zip(ys, row_keys):
            c = contrasts[key]
            point, lo, hi = c["point"], c["lower"], c["upper"]
            significant = (lo > 0) or (hi < 0)
            color = SALMON if (significant and point > 0) else (BLUE if (significant and point < 0) else GREY)
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                        ecolor=color, elinewidth=1.4, capsize=3, markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
        faint_grid(ax, axis="x")
        ax.set_yticks(np.arange(len(row_keys))[::-1])
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Difference in net correct gain")
        ax.set_ylim(-0.7, len(row_keys) - 0.3)
        prior_label = "gpt2 (strong LLM prior)" if prior == "gpt2" else "ngram5_kn (classical n-gram prior)"
        ax.text(0.5, -0.32, prior_label, transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
        panel(ax, letter)
        for spine in ["left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.15, bottom=0.28)
    save(fig, str(FIG_DIR / "Figure1"))


def figure2():
    rows = D["aim4_ladder"]["prior_capture"]
    # adaptive_minus_fixed = -(fixed - adaptive); positive = adaptive causes MORE prior-capture harm.
    items = []
    for r in rows:
        point = -r["point"]
        lo = -r["upper"]
        hi = -r["lower"]
        items.append((r["prior_model"], point, lo, hi))
    items.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(5.2, 7.8))
    ys = np.arange(len(items))[::-1]
    for y, (name, point, lo, hi) in zip(ys, items):
        significant = (lo > 0) or (hi < 0)
        color = SALMON if (significant and point < 0) else (BLUE if (significant and point > 0) else GREY)
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                    ecolor=color, elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
    faint_grid(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, *_ in items], fontsize=7.5)
    ax.set_xlabel("Adaptive − fixed prior-capture rate (positive = adaptive causes more harm)", fontsize=9, labelpad=8)
    ax.set_ylim(-2.4, len(items) - 0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=6, label="Adaptive significantly worse"),
        plt.Line2D([0], [0], marker="o", color=GREY, linestyle="", markersize=6, label="No significant difference"),
        plt.Line2D([0], [0], marker="o", color=SALMON, linestyle="", markersize=6, label="Adaptive significantly better"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8, frameon=False)
    for spine in ["left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(FIG_DIR / "Figure2"))


def efigure1_transport():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, prior, letter in zip(axes, ["gpt2", "ngram5_kn"], ["a", "b"]):
        rows = D["aim3_transportability"][prior]["prior_capture"]
        ys = np.arange(len(rows))[::-1]
        for y, r in zip(ys, rows):
            point, lo, hi = -r["point"], -r["upper"], -r["lower"]
            significant = (lo > 0) or (hi < 0)
            color = SALMON if (significant and point < 0) else (BLUE if (significant and point > 0) else GREY)
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                        ecolor=color, elinewidth=1.4, capsize=3, markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
        faint_grid(ax, axis="x")
        ax.set_yticks(np.arange(len(rows))[::-1])
        ax.set_yticklabels([r["held_out_study"] for r in rows])
        ax.set_xlabel("Adaptive − fixed prior-capture rate\n(held-out study)")
        panel(ax, letter)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.subplots_adjust(wspace=0.2, bottom=0.32)
    save(fig, str(FIG_DIR / "eFigure1"))


def efigure2_calibration():
    import pandas as pd
    sys.path.insert(0, str(ROOT))
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

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=0.9, linestyle="--", zorder=1)
    sizes = 20 + 200 * (np.array(n) / max(n))
    ax.scatter(conf, acc, s=sizes, color=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
    faint_grid(ax, axis="both")
    ax.set_xlabel("Mean predicted reliability (ĝ)")
    ax.set_ylabel("Observed neural-correct rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    auroc = D["aim1"]["gpt2"]["reliability_auroc"]
    ece = D["aim1"]["gpt2"]["reliability_ece"]
    ax.text(0.03, 0.94, f"AUROC = {auroc:.3f}\nECE = {ece:.3f}", transform=ax.transAxes, fontsize=8.5, color=LABEL, va="top")
    fig.subplots_adjust(bottom=0.15, left=0.18)
    save(fig, str(FIG_DIR / "eFigure2"))


if __name__ == "__main__":
    figure1()
    figure2()
    efigure1_transport()
    efigure2_calibration()
    print("Wrote Figure1, Figure2, eFigure1, eFigure2 to", FIG_DIR)
