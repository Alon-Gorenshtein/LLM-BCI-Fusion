"""Figures added or rebuilt for the revision. Run after the revision analyses have written
output/revision/*.json. Figure 1 is the study flow diagram; Figure 2 is the 24-prior fair
comparison (adaptive minus fixed, nested anchor selection, crossed intervals).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figure_kit import BLUE, GREY, LABEL, MUTED, faint_grid, save, set_style

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures"
REV = ROOT / "output" / "revision"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PITCH = 0.052   # line pitch in axes units (figure is 4.0 in tall; 7.4 pt text)


def _box(ax, cx, cy, w, lines, edge=MUTED, face="#F8F9FA", size=7.4, lw=1.0, bold_first=True):
    """Box of fixed text pitch centred on (cx, cy); returns its left, right, bottom, top edges."""
    h = PITCH * len(lines) + 0.05
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    for i, line in enumerate(lines):
        ax.text(cx, y + h - 0.025 - (i + 0.5) * PITCH, line, ha="center", va="center", fontsize=size,
                fontweight="bold" if (bold_first and i == 0) else "normal", color=LABEL, zorder=3)
    return x, x + w, y, y + h


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=MUTED, zorder=1, shrinkA=1, shrinkB=1))


def figure_flow(n_priors: int = 24):
    """Study flow: data, posteriors, the all-prior comparison, and the analyses on two anchors."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    a = _box(ax, 0.095, 0.50, 0.17, ["BigP3BCI archive", "4 studies", "47 people with ALS", "3,373 selections", "951 contexts"], size=7.2)
    n = _box(ax, 0.360, 0.73, 0.25, ["Neural posterior", "36 symbols per selection", "session-specific decoders", "(calibration data only)"], size=7.2)
    l = _box(ax, 0.360, 0.27, 0.25, [f"{n_priors} language-model priors", "3 character n-grams", "21 causal language models", "124M to 46.7B parameters"], size=7.2)
    f = _box(ax, 0.640, 0.50, 0.21, ["Fair comparison", "best fixed weight", "versus tuned", "adaptive weight,", f"all {n_priors} priors"],
             edge=BLUE, lw=1.6, size=7.2)
    ax.text(0.640, f[2] - 0.025, "Figure 2", ha="center", va="top", fontsize=7.2, color=BLUE, fontweight="bold")

    ax.text(0.885, 0.965, "Two anchors: GPT-2, 5-gram", ha="center", va="center", fontsize=7.4, fontweight="bold", color=LABEL)
    anchors = [
        ["Naive comparison", "(beta = 1), Fig. 3"],
        ["Reachable symbols", "per selection, Fig. 4a"],
        ["Three further", "controllers, Fig. 4b"],
        ["Error flagging", "Fig. 4c"],
        ["Replay with decoded", "history, Fig. 4d"],
    ]
    left_edges = []
    for i, text in enumerate(anchors):
        edges = _box(ax, 0.885, 0.85 - i * 0.18, 0.21, text, bold_first=False, size=7.2)
        left_edges.append((edges[0], (edges[2] + edges[3]) / 2))

    _arrow(ax, a[1], 0.56, n[0], 0.70)
    _arrow(ax, a[1], 0.44, l[0], 0.30)
    _arrow(ax, n[1], 0.70, f[0], 0.58)
    _arrow(ax, l[1], 0.30, f[0], 0.42)
    for x_left, y_mid in left_edges:
        _arrow(ax, f[1], 0.50, x_left, y_mid)
    save(fig, str(FIG_DIR / "Figure1"))


def figure_ladder(path: Path | None = None):
    """Adaptive minus fixed net correct gain for every prior (nested anchor selection), crossed
    participant-by-context 95% intervals, with the simultaneous band drawn lighter."""
    set_style()
    data = json.loads((path or REV / "fair_ladder.json").read_text())
    nested = data["summary"]["nested"]
    items = []
    for name, e in data["priors"].items():
        n = e["nested"]
        band = nested["simultaneous_band"][name]
        items.append((name, e["parameters"], n["diff"], n["ci_crossed"], band, n["identical_on_every_selection"]))
    items.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(5.4, 8.4))
    ys = np.arange(len(items))[::-1]
    for y, (name, params, diff, ci, band, tie) in zip(ys, items):
        ax.plot(band, [y, y], color=GREY, lw=3.0, solid_capstyle="butt", zorder=2)
        ax.errorbar(diff, y, xerr=[[diff - ci[0]], [ci[1] - diff]], fmt="o", color=MUTED, ecolor=MUTED,
                    elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=3)
    ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="--", zorder=1)
    faint_grid(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in items], fontsize=7.5)
    counts = nested["counts"]["crossed"]
    differ = [r for r in items if not r[5]]  # r[5]: identical on every selection
    inside = sum(1 for r in differ if r[3][0] > -0.01 and r[3][1] < 0.01)
    ax.set_xlabel(
        "Tuned-adaptive minus best-fixed net correct gain (positive favours adaptive)\n"
        f"{counts['identical_on_every_selection']} identical on every selection; "
        f"{counts['ci_crosses_zero']} intervals cross zero; {counts['ci_favors_adaptive']} favour adaptive; "
        f"{counts['ci_favors_fixed']} favour fixed\n"
        f"{inside}/{len(differ)} of the priors where the arms differ have intervals inside +/-1 percentage point",
        fontsize=8.2, labelpad=8)
    for side in ("left",):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(FIG_DIR / "Figure2"))


def _parse_arbitration(path: Path) -> dict:
    out, current = {}, None
    for line in path.read_text().splitlines():
        t = line.strip()
        if t.startswith("=== "):
            current = t.strip("= ").split()[0]
            out[current] = {}
        elif current and t.split(":")[0] in ("n_helped", "n_hurt", "n_no_effect"):
            out[current][t.split(":")[0]] = float(t.split(":")[1])
    return out


def figure4():
    """Mechanism and deployment: (a) reachable symbols by weight range, (b) what the arbitration controller's
    overrides did, (c) risk-coverage curves for error flagging, (d) replay with decoded history."""
    from hochberg_kit import apply_style, heatmap_values, panel as hpanel, save as hsave, KEY, ERROR, CONTEXT, ACCENT3, INK, MUTED as HMUTED
    apply_style()
    env = json.loads((REV / "envelope.json").read_text())["priors"]
    sel = json.loads((REV / "selective.json").read_text())["priors"]
    arb = _parse_arbitration(REV / "rerun_candidate_arbitration_clean.txt")
    priors = ("gpt2", "ngram5_kn")
    label = {"gpt2": "GPT-2", "ngram5_kn": "5-gram"}
    color = {"gpt2": KEY, "ngram5_kn": ACCENT3}
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 8.2))

    ax = axes[0, 0]
    ranges = [("grid5", "5 grid weights"), ("default_adaptive", "0.25 to 4"), ("widest_anchor", "0.1 to 20"), ("full", "full range")]
    rows, row_labels = [], []
    for pr in priors:
        for key, name in ranges:
            v = env[pr][key]
            rows.append([100 * v["share_1_reachable"], 100 * v["share_2_reachable"], 100 * v["share_3plus_reachable"]])
            row_labels.append(f"{label[pr]}, {name}")
    heatmap_values(ax, np.array(rows), row_labels=row_labels, col_labels=["1", "2", "3 or more"], fmt="{:.1f}%", vmin=0,
                   cbar_label="% of selections", valfontsize=8)
    ax.set_xlabel("Reachable symbols")
    hpanel(ax, "a")

    ax = axes[0, 1]
    keys, names = ["n_helped", "n_hurt", "n_no_effect"], ["Helped", "Hurt", "No effect"]
    width, x = 0.35, np.arange(3)
    for off, pr in zip((-width / 2, width / 2), priors):
        n = sum(arb[pr][k] for k in keys)
        heights = [100 * arb[pr][k] / n for k in keys]
        bars = ax.bar(x + off, heights, width, color=color[pr], edgecolor="white", linewidth=1.3, label=f"{label[pr]} (n = {int(n)})", zorder=3)
        for rect, h in zip(bars, heights):
            ax.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% of the prior's overridden selections")
    ax.set_ylim(0, 72)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    faint_grid(ax)
    hpanel(ax, "b")

    ax = axes[1, 0]
    show = [("fixed_fused_top1_prob", "Fused top-1 probability", KEY, "-"), ("candidate_ranker_p_hat", "Candidate ranker", ERROR, "-"),
            ("reliability_g_hat", "Reliability estimate", CONTEXT, "-")]
    for pr, ls in (("gpt2", "-"), ("ngram5_kn", "--")):
        for sig, name, c, _ in show:
            rc = sel[pr]["signals"][sig]["risk_coverage"]
            keep = [i for i, cv in enumerate(rc["coverage"]) if cv >= 0.1]
            risk = rc.get("risk", rc.get("selective_risk"))
            ax.plot([rc["coverage"][i] for i in keep], [risk[i] for i in keep], color=c, lw=1.8, linestyle=ls, label=f"{name}" if pr == "gpt2" else None)
        ax.axhline(sel[pr]["error_rate"], color=HMUTED, lw=0.8, linestyle=":")
    ax.set_xlabel("Coverage (share of selections kept)")
    ax.set_ylabel("Error rate among kept selections")
    ax.set_ylim(0, 0.32)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK, lw=1.6, linestyle="-"), plt.Line2D([], [], color=INK, lw=1.6, linestyle="--")]
    labels += ["GPT-2", "5-gram"]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="upper left")
    faint_grid(ax)
    hpanel(ax, "c")

    ax = axes[1, 1]
    rows = []
    for pr in priors:
        d = json.loads((REV / f"decoded_fair_{pr}.json").read_text()) if (REV / f"decoded_fair_{pr}.json").exists() else None
        if d is None:
            continue
        fp = d["fixed_policy_replay"]["decoded_minus_ground_truth"]
        rows.append((f"{label[pr]}: fixed policy,\ndecoded minus true prefix", fp["point"], fp["lower"], fp["upper"], KEY))
        rows.append((f"{label[pr]}: adaptive minus fixed,\ndecoded history", d["diff_adaptive_minus_fixed"], d["ci_crossed"][0], d["ci_crossed"][1], CONTEXT))
    ys = np.arange(len(rows))[::-1]
    for y, (name, pt, lo, hi, c) in zip(ys, rows):
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=9, markeredgecolor="white", zorder=3)
    ax.axvline(0, color=HMUTED, lw=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("Difference in net correct gain")
    faint_grid(ax, axis="x")
    hpanel(ax, "d")

    fig.subplots_adjust(hspace=0.38, wspace=0.62, left=0.2, right=0.97, top=0.97, bottom=0.07)
    hsave(fig, str(FIG_DIR / "Figure4"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["flow", "ladder", "mechanism"]
    if "mechanism" in which:
        figure4()
    if "flow" in which:
        figure_flow()
    if "ladder" in which:
        figure_ladder()
