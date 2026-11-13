"""Figure style helpers: near-monochrome with one saturated accent, red reserved for errors, individual points
and shaded 95% CI bands, lowercase bold panel letters, thin quiet chrome, no in-figure titles. Kept in this
repository so the figure pipeline is self-contained.

Usage:
    from hochberg_kit import apply_style, save, panel, perf_over_x, cluster_matrix, \
        decoded_example, neural_heatmap, INK, KEY, ERROR, CONTEXT
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    perf_over_x(ax, x, series=[...])
    save(fig, '/path/figure1')
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager as _fm

# ---- palette ----
INK     = '#1A1A1A'   # axes, text, primary data (near-black)
KEY     = '#2C5FAA'   # the one accent: headline / best result
ERROR   = '#C0392B'   # RESERVED for errors / the negative or worse outcome ONLY
CONTEXT = '#9AA0A6'   # secondary / context / "other" (gray)
ACCENT2 = '#1B7340'   # a second restrained accent when two good arms must differ (green)
ACCENT3 = '#C65102'   # third restrained accent (burnt orange) / trajectory start markers
MUTED   = '#6B7280'

def _ensure_arial():
    """Register the system Arial TTFs so matplotlib resolves 'Arial' (not a DejaVu
    fallback). Returns True if Arial is available after registration."""
    import os
    for p in ('/System/Library/Fonts/Supplemental/Arial.ttf',
              '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
              '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
              '/Library/Fonts/Arial.ttf'):
        if os.path.exists(p):
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    return 'Arial' in {f.name for f in _fm.fontManager.ttflist}


def apply_style():
    has_arial = _ensure_arial()
    family = ['Arial', 'Helvetica', 'Helvetica Neue', 'DejaVu Sans']
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': family,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'font.size': 8,
        'axes.titlesize': 8, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.edgecolor': INK, 'axes.linewidth': 0.8,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })
    if not has_arial:
        import warnings
        warnings.warn('Arial not found; falling back to Helvetica/DejaVu Sans.')
    return has_arial

def save(fig, path_no_ext, dpi=600):
    """Export 600 DPI PDF (vector) + PNG, journal-ready."""
    from pathlib import Path
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{p}.{ext}", dpi=dpi, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return f"{p}.png"

def panel(ax, letter, dx=-0.16, dy=1.06, fontsize=14):
    """Bold lowercase panel letter, ~14pt, consistent top-left placement."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left', color=INK)

def _despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def refline(ax, y, label=None, horizontal=True):
    """Light dashed ceiling/chance reference line."""
    if horizontal:
        ax.axhline(y, ls=(0, (4, 3)), lw=0.8, color=MUTED, zorder=1)
        if label:
            ax.text(0.012, y, label, transform=ax.get_yaxis_transform(),
                    va='bottom', ha='left', fontsize=7, color=MUTED, style='italic')
    else:
        ax.axvline(y, ls=(0, (4, 3)), lw=0.8, color=MUTED, zorder=1)

def perf_over_x(ax, x, series, xlabel='', ylabel='', chance=None, chance_label='chance',
                annotate_last=True, lw=2.6, ms=42, band_alpha=0.22, annotate_all=False,
                num_dx=5, num_dy=7):
    """The workhorse performance-over-x plot (bold / data-rich variant).

    series: list of dicts, each:
      { 'y': array, 'lo': array|None, 'hi': array|None, 'label': str,
        'color': hex (default INK), 'marker': 'o'|'s'|'D'|'v' (distinct shapes),
        'open': bool, 'ls': '-'|'--' }
    Thick lines, large distinct markers, darker SEM/CI bands; optional printed value on
    every datum (annotate_all) or just the endpoint (annotate_last)."""
    x = np.asarray(x)
    for s in series:
        col = s.get('color', INK)
        y = np.asarray(s['y'])
        lo, hi = s.get('lo'), s.get('hi')
        if lo is not None and hi is not None:
            ax.fill_between(x, np.asarray(lo), np.asarray(hi), color=col, alpha=band_alpha,
                            linewidth=0, zorder=1)
        ax.plot(x, y, color=col, linestyle=s.get('ls', '-'), lw=lw, zorder=3,
                label=s.get('label'), solid_capstyle='round')
        mk = s.get('marker', 'o')
        if s.get('open', False):
            ax.scatter(x, y, s=ms, facecolors='white', edgecolors=col, linewidths=1.6,
                       marker=mk, zorder=4)
        else:
            ax.scatter(x, y, s=ms, color=col, edgecolors='white', linewidths=0.9,
                       marker=mk, zorder=4)
        if annotate_all:
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords='offset points',
                            xytext=(num_dx, num_dy), va='center', ha='center',
                            fontsize=7.2, fontweight='bold', color=col)
        elif annotate_last:
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords='offset points',
                        xytext=(num_dx, 0), va='center', fontsize=7.5, fontweight='bold',
                        color=col)
    if chance is not None:
        refline(ax, chance, chance_label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _despine(ax)
    return ax

def cluster_matrix(ax, M, labels=None, cmap='Blues', show_values=False, cbar_label=''):
    """Confusion/distance/agreement matrix, optionally reordered by hierarchical clustering."""
    M = np.asarray(M, float)
    order = np.arange(M.shape[0])
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1 - M if (M.max() <= 1 and (np.diag(M) > 0.9).all()) else M
        D = (D + D.T) / 2; np.fill_diagonal(D, 0)
        order = leaves_list(linkage(squareform(D, checks=False), method='average'))
        M = M[np.ix_(order, order)]
    except Exception:
        pass
    im = ax.imshow(M, cmap=cmap, aspect='equal')
    ax.set_xticks(range(M.shape[0])); ax.set_yticks(range(M.shape[0]))
    if labels is not None:
        lab = [labels[i] for i in order]
        ax.set_xticklabels(lab, rotation=90, fontsize=6.5)
        ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_xticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='minor', length=0)
    if show_values:
        for i in range(M.shape[0]):
            for j in range(M.shape[0]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha='center', va='center', fontsize=6,
                        color='white' if M[i, j] > M.max()*0.6 else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=7.5)
    return ax

def neural_heatmap(ax, M, xlabel='time', ylabel='channel / component', cmap='Greys'):
    """Per-panel-normalized neural activity heatmap (spatiotemporal)."""
    M = np.asarray(M, float)
    im = ax.imshow(M, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    return ax

def decoded_example(ax, lines, xlabel=''):
    """Monospace verbatim decoded output; errors in red.
    lines: list of (text, list_of_error_char_indices). Space shown as '>'.
    """
    ax.axis('off')
    y = 0.92
    for text, errs in lines:
        x = 0.02
        for i, ch in enumerate(text):
            disp = '>' if ch == ' ' else ch
            ax.text(x, y, disp, transform=ax.transAxes, family='monospace', fontsize=8.5,
                    color=(ERROR if i in errs else INK),
                    fontweight=('bold' if i in errs else 'normal'))
            x += 0.018
        y -= 0.16
    if xlabel:
        ax.set_title(xlabel, fontsize=7.5, loc='left', color=MUTED)
    return ax

def perctext(ax, x, y, s, color=None):
    ax.annotate(s, (x, y), textcoords='offset points', xytext=(0, 7), ha='center',
                fontsize=7.5, color=color or INK)


# ============================================================================
# Data-rich builders (viridis heatmaps, confusion, time-series)
# ============================================================================
import matplotlib.patches as _mpatches

VIRIDIS = 'viridis'
DIVERGE = 'RdBu_r'              # signed/z-scored spectral features (NEJM)
CTRL_ORANGE = '#E2A33C'         # healthy controls (Claassen)
CMD_BLUE    = '#5B8FC9'         # patients WITH the signal (CMD)
NOCMD_BLACK = '#1A1A1A'         # patients without
# bold, saturated, distinguishable set for many conditions (one per condition)
BOLD = ['#1F3FB0', '#D1352B', '#1B7F3B', '#E07B1A', '#19A0C0', '#C8A21A',
        '#6A3D9A', '#E15A99', '#5BA84A', '#8C6D31']
# distinct hues for category-group brackets
CAT = ['#C0392B', '#1B7340', '#6A4C93', '#C65102', '#2C5FAA', '#117A8B', '#A61E4D']


def _rel_lum(rgb):
    """WCAG relative luminance of an sRGB triple in [0,1]."""
    def _lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(l1, l2):
    a, b = max(l1, l2), min(l1, l2)
    return (a + 0.05) / (b + 0.05)


def _txt_on_cmap(norm, cmap=None):
    """Luminance-aware label color keyed to the ACTUAL facecolor of the cell.

    Computes the WCAG relative luminance of the colormap color at `norm` (the cell's
    position on the [vmin,vmax] scale) and returns whichever of near-black / white wins
    the >=4.5:1 contrast comparison against that background. This is the single rule
    shared by heatmap_values() and confusion_matrix(), so every cell -- including both
    bright-green confusion diagonals and the mid-blue off-diagonals -- is keyed to its
    own facecolor luminance, never to anything else."""
    import matplotlib as _mpl
    cmap = _mpl.colormaps['viridis'] if cmap is None else cmap
    if isinstance(cmap, str):
        cmap = _mpl.colormaps[cmap]
    bg = cmap(float(np.clip(norm, 0.0, 1.0)))
    lbg = _rel_lum(bg)
    c_black = _contrast(_rel_lum((0.04, 0.04, 0.04)), lbg)
    c_white = _contrast(1.0, lbg)
    return '#0A0A0A' if c_black >= c_white else 'white'


def _txt_on_viridis(norm, cmap=VIRIDIS):
    """Back-compat shim -> luminance-aware text keyed to the cell facecolor."""
    return _txt_on_cmap(norm, cmap)


def heatmap_values(ax, M, row_labels=None, col_labels=None, cmap=VIRIDIS,
                   vmin=None, vmax=None, fmt='{:.2f}', col_groups=None,
                   row_groups=None, excluded=None, cbar_label='', valfontsize=10.0,
                   col_group_title=None, row_group_title=None, sep_rows=None,
                   sep_cols=None, cbar='horizontal', cbar_ticks=None,
                   row_bracket_x=-0.18, row_label_x=-0.27):
    """Viridis matrix with a bold printed value in every cell + grouped category brackets.

    M: 2D array. col_groups/row_groups: list of (label, start, end) inclusive-exclusive.
    excluded: list of (i, j) cells to mark with a red X (excluded / not-applicable).
    sep_rows/sep_cols: extra thin black separator rules at these boundary indices.
    cbar: 'horizontal' (beneath, divider-tied) | 'vertical' | None.
    Reads as a table: hairline gridlines between every cell + 1pt black frame.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    M = np.asarray(M, float)
    vmin = np.nanmin(M) if vmin is None else vmin
    vmax = np.nanmax(M) if vmax is None else vmax
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    nr, nc = M.shape
    rng = (vmax - vmin) or 1.0
    for i in range(nr):
        for j in range(nc):
            if np.isnan(M[i, j]):
                continue
            norm = (M[i, j] - vmin) / rng
            ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                    fontsize=valfontsize, fontweight='bold', color=_txt_on_viridis(norm))
    if excluded:
        for (i, j) in excluded:
            ax.text(j, i, '✕', ha='center', va='center', fontsize=valfontsize+3,
                    color='#C0392B', fontweight='bold')
    ax.set_xticks(range(nc)); ax.set_yticks(range(nr))
    ax.set_xticklabels(col_labels if col_labels else range(nc), rotation=0, fontsize=8)
    ax.set_yticklabels(row_labels if row_labels else range(nr), fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # hairline gridlines between every cell (reads as a table)
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    ax.grid(which='minor', color='black', lw=0.5)
    ax.tick_params(which='minor', length=0)
    # 1pt black frame
    ax.add_patch(_mpatches.Rectangle((-0.5, -0.5), nc, nr, fill=False,
                 edgecolor='black', lw=1.0, clip_on=False, zorder=5))
    # extra separator rules (e.g. group divider, Chance offset)
    for r in (sep_rows or []):
        ax.axhline(r - 0.5, color='black', lw=0.85, zorder=6)
    for c in (sep_cols or []):
        ax.axvline(c - 0.5, color='black', lw=0.85, zorder=6)
    # column group headers (bracket + bold label) + group separators
    if col_groups:
        for k, (lab, a, b) in enumerate(col_groups):
            ax.plot([a-0.5+0.04, b-0.5-0.04], [-0.78, -0.78], color=CAT[k % len(CAT)],
                    lw=3.2, clip_on=False, solid_capstyle='butt')
            ax.text((a+b-1)/2, -1.18, lab, ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=INK, clip_on=False)
            if a > 0:
                ax.axvline(a-0.5, color='black', lw=0.85, zorder=6)
        if col_group_title:
            ax.text((nc-1)/2, -1.9, col_group_title, ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=INK, clip_on=False)
    if row_groups:
        import matplotlib.transforms as _mtrans
        # blended transform: x in axes fraction (stable left margin), y in data coords
        bt = _mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        for k, (lab, a, b) in enumerate(row_groups):
            ax.plot([row_bracket_x, row_bracket_x], [a-0.5+0.05, b-0.5-0.05],
                    color=CAT[k % len(CAT)], lw=3.4, clip_on=False, solid_capstyle='butt',
                    transform=bt, zorder=7)
            # group label sits horizontally to the LEFT of its own bracket, at the
            # bracket's vertical center -> never overprints the neighbouring group's
            # label and clears the row tick text by a comfortable margin.
            ax.text(row_label_x, (a+b-1)/2, lab, ha='right', va='center', rotation=0,
                    fontsize=10, fontweight='bold', color=INK, clip_on=False,
                    transform=bt, zorder=7)
            if a > 0:
                ax.axhline(a-0.5, color='black', lw=0.85, zorder=6)
    if cbar == 'horizontal':
        div = make_axes_locatable(ax)
        cax = div.append_axes('bottom', size='6%', pad=0.42)
        cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal',
                                ticks=cbar_ticks)
        cb.outline.set_linewidth(0.6); cb.ax.tick_params(labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8, fontweight='bold')
    elif cbar == 'vertical':
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=cbar_ticks)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8, fontweight='bold')
    return im


def confusion_matrix(ax, M, groups, cmap=VIRIDIS, title='', labels=None,
                     show_values=True, overall_acc=None):
    """Confusion matrix on a DARK VIRIDIS field with per-category diagonal boxes outlined
    in the category color. Overall accuracy printed in the title.

    Every cell's printed value is colored by the SHARED _txt_on_cmap luminance rule keyed
    to that cell's OWN viridis facecolor (vmin=0, vmax=1, so value==norm). Both bright-green
    diagonals therefore get the same near-black glyph, and the mid-blue off-diagonals get
    whichever of black/white clears 4.5:1 against their actual facecolor -- no inverted or
    per-cell-inconsistent decision.

    groups: list of (label, start, end). M assumed row-normalized (fraction of trials)."""
    M = np.asarray(M, float)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    n = M.shape[0]
    if show_values:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=9,
                        fontweight='bold', color=_txt_on_cmap(M[i, j], cmap))
    for k, (lab, a, b) in enumerate(groups):
        ax.add_patch(_mpatches.Rectangle((a-0.5, a-0.5), b-a, b-a, fill=False,
                     edgecolor=CAT[k % len(CAT)], lw=2.4, zorder=5))
        ax.plot([a-0.5, b-0.5], [n-0.4]*2, color=CAT[k % len(CAT)], lw=3, clip_on=False)
    if labels is not None:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.tick_params(length=0)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if overall_acc is None:
        overall_acc = float(np.trace(M) / max(M.sum(), 1e-9))
    ttl = title or 'Confusion'
    ax.set_title(f"{ttl}  (overall acc {overall_acc:.2f})", fontsize=8.5,
                 fontweight='bold', loc='center')
    return ax


# back-compat alias for the old name
confusion_boxes = confusion_matrix


def timeseries_multicolor(ax, t, series, event=0.0, xlabel='Time (s)', ylabel=''):
    """Many conditions, each bold color, thick line + shaded SEM/CI band, dashed event line.
    series: list of dict { 'mean': arr, 'lo': arr, 'hi': arr, 'label': str, 'color': hex }.
    Returns (handles, labels) for a legend_below call."""
    t = np.asarray(t)
    H, L = [], []
    for i, s in enumerate(series):
        col = s.get('color', BOLD[i % len(BOLD)])
        m = np.asarray(s['mean'])
        if s.get('lo') is not None:
            ax.fill_between(t, np.asarray(s['lo']), np.asarray(s['hi']), color=col,
                            alpha=0.20, linewidth=0, zorder=1)
        line, = ax.plot(t, m, color=col, lw=1.8, zorder=3)
        H.append(line); L.append(s.get('label', f'cond {i}'))
    if event is not None:
        ax.axvline(event, ls='--', lw=0.9, color=INK, zorder=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _despine(ax)
    return H, L


def despine_all(axes):
    """Remove top + right spines on every axis (uniform chrome)."""
    for ax in np.ravel(np.asarray(axes, dtype=object)):
        if ax is not None:
            _despine(ax)


def legend_below(fig, handles, labels, ncol=5, y=-0.02, fontsize=8):
    leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y),
                     ncol=ncol, fontsize=fontsize, frameon=False, handletextpad=0.6,
                     columnspacing=1.8)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    return leg


def swatch(color, marker='o', ls='-', lw=2.6):
    """A bold colored line+marker handle for a horizontal legend row."""
    return mlines.Line2D([], [], color=color, lw=lw, marker=marker, ms=7,
                         mfc=color, mec='white', mew=0.6, ls=ls)


def _block_step(centers, values, half=0.5):
    """Turn one value per block into a stepped/plateau trace: each block value is held
    flat across [center-half, center+half] so the trace reads as Move-high / Rest-low
    plateaus (the Claassen blocked look) instead of a point-to-point sawtooth.
    Returns (xs, ys) ready for ax.plot / ax.fill_between."""
    centers = np.asarray(centers, float)
    values = np.asarray(values, float)
    xs, ys = [], []
    for c, v in zip(centers, values):
        xs.extend([c - half, c + half]); ys.extend([v, v])
    return np.asarray(xs), np.asarray(ys)


def decoding_over_trials(axes, move_centers, rest_centers, groups,
                         ylabel='P(imagery) decoded', scale_bar=None, scale_label=''):
    """Claassen NEJM TWO-STRIP signature, rebuilt as the true layout the DoC PI asked for.

    `axes` is a (ax_move, ax_rest) pair from plt.subplots(2,1, sharex=True): the TOP strip
    is its own band showing decoded P(imagery) during MOVE blocks, the BOTTOM strip its own
    band showing it during REST blocks. Each block value is held flat across its block
    (blocked envelope, not a sawtooth). A 0.5 INK dashed midline sits in BOTH strips. Awake
    sits clearly ABOVE 0.5 on Move and BELOW 0.5 on Rest; the intermediate group is damped
    toward 0.5 in both; deep runs flat at ~0.5 in both.

    Task markers live OFF the data gridlines: SOLID gray Move bars in a thin header lane
    just ABOVE the top strip's data, DOTTED gray Rest bars in a thin lane just BELOW the
    bottom strip's data, each spanning the real block x-extents at full opacity.

    groups: list of dict { 'move': arr (one per Move block), 'rest': arr (one per Rest
            block), optional 'move_lo'/'move_hi'/'rest_lo'/'rest_hi', 'label', 'color',
            'flat': bool, 'lw': float }.
    move_centers / rest_centers: block-center x-positions for the Move and Rest strips.
    """
    ax_m, ax_r = axes
    GRAYBAR = '#73787E'
    H, L = [], []
    for ax in (ax_m, ax_r):
        ax.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_ylim(0.0, 1.0)
    # --- plot each group's blocked plateau in each strip ---
    for g in groups:
        col = g['color']; lw = g.get('lw', 2.6)
        ls = g.get('ls', (0, (5, 2)) if g.get('flat') else '-')
        for ax, key, lo_k, hi_k, cen in (
                (ax_m, 'move', 'move_lo', 'move_hi', move_centers),
                (ax_r, 'rest', 'rest_lo', 'rest_hi', rest_centers)):
            v = np.asarray(g[key], float)
            xs, ys = _block_step(cen, v)
            if (not g.get('flat')) and g.get(lo_k) is not None:
                xlo, ylo = _block_step(cen, np.asarray(g[lo_k], float))
                _, yhi = _block_step(cen, np.asarray(g[hi_k], float))
                ax.fill_between(xlo, ylo, yhi, color=col, alpha=0.13, linewidth=0, zorder=3)
            line, = ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=4,
                            solid_capstyle='round')
            # block-center markers so each plateau reads as a block estimate
            ax.plot(cen, v, color=col, ls='none', marker='o', ms=3.6,
                    mfc=col, mec='white', mew=0.5, zorder=5)
            if ax is ax_m:
                H.append(line); L.append(g['label'])
    # --- task-marker lanes OFF the data gridlines (clip_on=False, above/below the band) ---
    half = 0.5
    for c in move_centers:
        ax_m.add_patch(_mpatches.Rectangle((c - half, 1.015), 2 * half, 0.07,
                       facecolor=GRAYBAR, edgecolor='none', alpha=1.0,
                       clip_on=False, zorder=6, transform=ax_m.get_xaxis_transform()))
    for c in rest_centers:
        # dotted Rest bar: a thick dotted line in a lane just BELOW the bottom strip
        ax_r.plot([c - half + 0.04, c + half - 0.04], [-0.085, -0.085],
                  color=GRAYBAR, lw=5.0, ls=(0, (1.4, 1.2)), solid_capstyle='butt',
                  clip_on=False, zorder=6, transform=ax_r.get_xaxis_transform())
    # --- bold muted rotated strip super-labels ---
    ax_m.set_ylabel('Move', rotation=90, fontsize=10, fontweight='bold', color=MUTED,
                    labelpad=8)
    ax_r.set_ylabel('Rest', rotation=90, fontsize=10, fontweight='bold', color=MUTED,
                    labelpad=8)
    # the quantitative y-axis label belongs on the figure, set by the caller
    ax_m.set_yticks([0.0, 0.5, 1.0]); ax_r.set_yticks([0.0, 0.5, 1.0])
    # task-marker legend handles
    H.append(mlines.Line2D([], [], color=GRAYBAR, lw=6.0))
    L.append('Move block (top lane)')
    H.append(mlines.Line2D([], [], color=GRAYBAR, lw=4.0, ls=(0, (1.4, 1.2))))
    L.append('Rest block (bottom lane)')
    # optional time scale bar on the bottom strip, off the data
    if scale_bar:
        x1 = max(np.max(move_centers), np.max(rest_centers))
        x0 = x1 - scale_bar
        ax_r.plot([x0, x1], [0.12, 0.12], color=INK, lw=2.4, solid_capstyle='butt',
                  zorder=7)
        ax_r.text((x0 + x1) / 2, 0.06, scale_label, ha='center', va='top',
                  fontsize=7, color=INK)
    _despine(ax_m); _despine(ax_r)
    return H, L


def spectral_bands(fig, axes, bands, row_labels=None, col_labels=None,
                   show_values=True, cbar_label='z-scored band power', vlim=None,
                   fmt='{:+.1f}', row_groups=None, row_bracket_x=-0.78,
                   row_label_x=-0.92, per_panel_vlim=False, null_band_thresh=None):
    """Row of diverging RdBu_r heatmaps (delta/theta/alpha/beta), centered zero=white,
    feeding the SVM->AUC pipeline. A printed value per cell.
    bands: list of (name, 2D array). vlim sets a shared symmetric color range.

    per_panel_vlim: if True, each panel gets its OWN symmetric color range scaled to that
      band's data (min floor 0.04) so a compressed band like beta (all |z|<0.05) still maps
      its small imagery-minus-rest sign to a visible blue/red instead of a blank white strip.
      A small per-panel scale tag is printed under each such panel's title. The shared
      colorbar then reflects the widest band.
    null_band_thresh: if set, any band whose max |value| is below this is treated as a true
      null: it keeps the shared scale, gets a light gray hatch overlay and a centered
      'no band separation' annotation, so a flat panel reads as an intended null result.

    row_groups: list of (label, start, end) region brackets (e.g. frontal/central/parieto-
    occipital) drawn on the LEFT of the first panel, matching the fig3 row-bracket grammar."""
    axes = np.ravel(axes)
    band_max = [float(np.nanmax(np.abs(np.asarray(M)))) for _, M in bands]
    if vlim is None:
        vlim = max(band_max)
    im = None
    for k, (ax, (name, M)) in enumerate(zip(axes, bands)):
        M = np.asarray(M, float)
        bmax = band_max[k]
        is_null = (null_band_thresh is not None) and (bmax < null_band_thresh)
        # choose this panel's color range
        if is_null:
            pvlim = vlim                       # keep shared scale, annotate as null
        elif per_panel_vlim:
            pvlim = max(bmax * 1.05, 0.04)     # tight symmetric range so small signs show
        else:
            pvlim = vlim
        imk = ax.imshow(M, cmap=DIVERGE, vmin=-pvlim, vmax=pvlim, aspect='auto')
        if not (per_panel_vlim or is_null):
            im = imk                            # shared-scale colorbar source
        elif im is None:
            im = imk
        if per_panel_vlim and not is_null:
            # fold the per-panel symmetric range into the title (two lines) so it never
            # overprints the band name; the small beta range then reads as intentional.
            ax.set_title(name, fontsize=9, fontweight='bold', pad=12)
            ax.text(0.5, 1.02, f'(scale +/-{pvlim:.2f})', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=6.4, color=MUTED, clip_on=False)
        else:
            ax.set_title(name, fontsize=9, fontweight='bold')
        nr, nc = M.shape
        if show_values:
            for i in range(nr):
                for j in range(nc):
                    val = M[i, j]
                    tc = 'white' if abs(val) > 0.62 * pvlim else '#0A0A0A'
                    ax.text(j, i, fmt.format(val), ha='center', va='center',
                            fontsize=7, fontweight='bold', color=tc)
        if is_null:
            ax.add_patch(_mpatches.Rectangle((-0.5, -0.5), nc, nr, fill=True,
                         facecolor='none', edgecolor='#9AA0A6', hatch='////', lw=0.0,
                         alpha=0.55, zorder=4))
            ax.text(0.5, 0.5, 'no band\nseparation\n(|z| < %.2f)' % bmax,
                    transform=ax.transAxes, ha='center', va='center', fontsize=6.6,
                    fontweight='bold', color='#3A3A3A', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#9AA0A6',
                              lw=0.6, alpha=0.85))
        ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
        ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
        ax.grid(which='minor', color='black', lw=0.4)
        ax.tick_params(which='minor', length=0)
        ax.add_patch(_mpatches.Rectangle((-0.5, -0.5), nc, nr, fill=False,
                     edgecolor='black', lw=0.9, clip_on=False))
        if col_labels is not None:
            ax.set_xticks(range(nc)); ax.set_xticklabels(col_labels, fontsize=7, rotation=0)
        else:
            ax.set_xticks([])
        if row_labels is not None and k == 0:
            ax.set_yticks(range(nr)); ax.set_yticklabels(row_labels, fontsize=7)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # left-side region brackets on the channel axis of the first panel only
        if row_groups and k == 0:
            import matplotlib.transforms as _mtrans
            bt = _mtrans.blended_transform_factory(ax.transAxes, ax.transData)
            for gi, (lab, a, b) in enumerate(row_groups):
                ax.plot([row_bracket_x, row_bracket_x], [a-0.5+0.05, b-0.5-0.05],
                        color=CAT[gi % len(CAT)], lw=3.4, clip_on=False,
                        solid_capstyle='butt', transform=bt, zorder=7)
                ax.text(row_label_x, (a+b-1)/2, lab, ha='center', va='center',
                        rotation=90, fontsize=7.5, fontweight='bold', color=INK,
                        clip_on=False, transform=bt, zorder=7)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), orientation='vertical',
                          fraction=0.025, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        cb.set_label(cbar_label, fontsize=8, fontweight='bold')
    return axes
