"""Journal-grade matplotlib kit (scientific-figures skill standard).
Helvetica, colorblind-safe palette, 600 DPI PDF+PNG, no in-figure titles, panel labels, error bars,
invisible gridlines, top/right spines removed. Import set_style()/save()/panel() and build figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lancet palette (3-colour, colorblind-safe) + Okabe-Ito for many categories
BLUE, GREY, SALMON = "#00468B", "#ADB6B6", "#FDAF91"
LABEL, MUTED = "#2B2B2B", "#6B7280"
OKABE = ["#E69F00", "#D55E00", "#CC79A7", "#0072B2", "#009E73", "#56B4E9", "#999999"]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })


def save(fig, path_no_ext):
    """Write both PDF (vector, journal-preferred) and PNG at 600 DPI."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_no_ext}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def panel(ax, letter):
    """Lowercase bold panel label (a/b/c) at the top-left, Nature convention."""
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)


def faint_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#F3F4F6", lw=0.6, zorder=0)


# Reminders encoded as comments (the traps from the origin run):
# - Bars/points that compare estimates MUST show 95% CI error bars (yerr=[lo,hi]).
# - For >3 categories use OKABE, not the 3-colour Lancet set.
# - Never put a title inside the figure; the caption (in the manuscript) carries it.
# - Smooth noisy hourly time-series with a centered rolling mean before plotting.
