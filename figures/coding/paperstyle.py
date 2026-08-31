"""Shared figure style for the paper. Every figure script in for_paper/ imports this so the
figures are one visual system and match the LaTeX body text.

    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paperstyle import COL, FULL, C, use_paper_style, save
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ...
    save(fig, "fig2_learning_curves")     # -> for_paper/figures/fig2_learning_curves.pdf

See for_paper/AGENTS.md for the rules these defaults implement.
"""
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- venue geometry, inches. Set ONCE for the target venue, then never rescale in LaTeX. ---
#   ICML / NeurIPS (two-column): COL 3.25, FULL 6.75
#   ACL / EMNLP:                 COL 3.15, FULL 6.50
#   NeurIPS single-column:       COL 5.50, FULL 5.50
COL = 3.25
FULL = 6.75

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# semantic palette, identical to reports/intern_pointers (see its AGENTS.md sec. 4)
C = {
    "ours":       "#d62728",   # ours / DPPO / the win
    "gated":      "#1f77b4",   # ESS-gated
    "baseline":   "#7f7f7f",   # baseline / prior SOTA
    "baseline2":  "#999999",
    "baseline3":  "#c0c0c0",
    "eval":       "#000000",   # single-series eval curve
    "alt1":       "#2ca02c",
    "alt2":       "#ff7f0e",
    "alt3":       "#9467bd",
    "alt4":       "#17becf",
}

# Curated multi-series palette, for figures where every series needs its own hue (the dynamics
# panels). Deeper and less saturated than tab10, so several hues sit together without clashing and
# each still reads against the grey grid. Use in order; do not reorder per figure.
FAM = [
    "#4C72B0",   # blue
    "#DD8452",   # amber
    "#55A868",   # green
    "#C44E52",   # brick, reserve for the collapse / the thing to notice
    "#8172B3",   # violet
    "#937860",   # taupe
]
# baseline dashed / ours solid, so the figures survive greyscale printing
LS = {"ours": "-", "gated": "-", "baseline": "--", "ref": ":"}


def format_sig(value, digits=3):
    """Format a finite scalar with a fixed number of significant digits."""
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return f"{value:.{digits - 1}f}"
    decimals = digits - 1 - math.floor(math.log10(abs(value)))
    rounded = round(value, decimals)
    return f"{rounded:.{max(decimals, 0)}f}"


def use_paper_style():
    plt.rcParams.update({
        # NO savefig.bbox="tight": it crops the canvas, so the PDF comes out narrower than COL
        # and \includegraphics[width=\linewidth] would silently scale the 8pt text up. Constrained
        # layout keeps the canvas exactly figsize, so LaTeX places it at scale 1.0.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "pdf.fonttype": 42,          # TrueType, not Type-3: required by most venues
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 2.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def save(fig, name):
    """Write for_paper/figures/<name>.pdf (vector, final size) and report the path.

    `name` may contain a subfolder, e.g. save(fig, "motivation/ess_governs").
    """
    out = os.path.join(FIGDIR, name + ".pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print("saved", out)
    plt.close(fig)
    return out
