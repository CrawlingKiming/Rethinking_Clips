#!/usr/bin/env python
"""Core Qwen3-30B-A3B trajectories for Section 6.

All conditional curves use TIS 3 while ESS is high and activate the named
clipping safeguard when ESS is low. The first three panels compare this
template with an always-on safeguard; the fourth compares it directly with
ungated TIS 3, exposing the collapse that conditioning prevents.

Outputs:
  figures_mains/result/q30ba3b/curves/overall.pdf      (main-text panels a--c)
  figures_mains/result/q30ba3b/curves/tis3.pdf         (standalone panel d)
  figures_mains/result/q30ba3b/curves/overall_all.pdf  (all four panels)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle
from paperstyle import COL, FULL, C, format_sig, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

STATIC = "#4d4d4d"
CONDITIONAL = C["ours"]
PANELS = [
    ("GRPO", "grpo_base", "grpo_ess_clip", (-5, 6), "grpo"),
    ("Clip-Higher", "dapo_base", "dapo_ess", (-5, 6), "grpo_cliphigher"),
    ("DPPO", "dppo_base", "dppo_ess", (-5, 6), "dppo"),
    ("TIS-3", "cispo3_nogate", "dapo_ess", (8, 6), "tis3"),
]


def annotate_endpoint(ax, x, y, color, offset, compact=False):
    ax.annotate(
        f"{format_sig(100 * y)}%",
        xy=(x, 100 * y),
        xytext=(3 if compact else 5, offset),
        textcoords="offset points",
        color=color,
        fontsize=5.9 if compact else 6.5,
        ha="left",
        va="center",
        clip_on=False,
    )


def draw(ax, static_run, conditional_run, offsets, compact=False,
         reference_color=STATIC):
    for run, color, linestyle, linewidth, offset in [
        (static_run, reference_color, "--", 1.2, offsets[0]),
        (conditional_run, CONDITIONAL, "-", 1.6, offsets[1]),
    ]:
        xs, ys = series(run, "eval")
        ax.plot(
            xs,
            [100 * value for value in ys],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=2.7,
        )
        annotate_endpoint(ax, xs[-1], ys[-1], color, offset, compact)
    ax.set_xlim(-3, 235 if compact else 222)
    ax.set_ylim(-1.5, 50)


clipped_handles = [
    Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.2,
           marker="o", markersize=2.7, label="Clipped"),
    Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.6,
           marker="o", markersize=2.7, label="ESS-conditioned"),
]

tis3_handles = [
    Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.2,
           marker="o", markersize=2.7, label="TIS-3"),
    Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.6,
           marker="o", markersize=2.7, label="ESS-conditioned"),
]

use_paper_style()

# Main-text figure: the three clipping rules only.
fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.42), sharex=True, sharey=True)
for index, (title, static_run, conditional_run, offsets, _slug) in enumerate(PANELS[:3]):
    axis = axes[index]
    draw(axis, static_run, conditional_run, offsets)
    axis.set_title(
        f"({'abc'[index]}) {title}",
        loc="left",
        fontsize=8.4,
        fontweight="bold",
    )
    axis.set_xticks([0, 100, 200])
    axis.set_yticks([0, 25, 50])
    axis.tick_params(axis="both", labelsize=7.2)
    axis.grid(True, which="major", alpha=0.26)
    if index == 0:
        axis.set_ylabel("AIME-2024 mean@16 (%)")
        axis.yaxis.label.set_size(8.0)
fig.supxlabel("training step", fontsize=8.0)
fig.legend(
    handles=clipped_handles,
    loc="outside upper center",
    ncol=2,
    frameon=False,
    fontsize=8.2,
    handlelength=2.3,
)
save(fig, "result/q30ba3b/curves/overall")

# Archival all-in-one view. Panel (d) uses a distinct reference color so its
# ungated TIS-3 baseline cannot be mistaken for the clipped baselines in (a--c).
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.36), sharex=True, sharey=True)
for index, (title, static_run, conditional_run, offsets, _slug) in enumerate(PANELS):
    axis = axes[index]
    reference_color = C["alt2"] if index == 3 else STATIC
    draw(axis, static_run, conditional_run, offsets, compact=True,
         reference_color=reference_color)
    axis.set_title(
        f"({'abcd'[index]}) {title}", loc="left", fontsize=8.0,
        fontweight="bold",
    )
    axis.set_xticks([0, 100, 200])
    axis.set_yticks([0, 25, 50])
    axis.tick_params(axis="both", labelsize=6.8)
    if index == 0:
        axis.set_ylabel("AIME-2024 mean@16 (%)", fontsize=7.4)
fig.supxlabel("training step", fontsize=7.4)
fig.legend(
    handles=[
        clipped_handles[0],
        Line2D([0], [0], color=C["alt2"], linestyle="--", linewidth=1.2,
               marker="o", markersize=2.7, label="TIS-3"),
        clipped_handles[1],
    ],
    loc="outside upper center", ncol=3, frameon=False, fontsize=7.5,
    handlelength=2.0,
)
save(fig, "result/q30ba3b/curves/overall_all")

for title, static_run, conditional_run, offsets, slug in PANELS:
    fig, axis = plt.subplots(figsize=(COL, 2.35))
    draw(axis, static_run, conditional_run, offsets)
    axis.set_title(title, loc="left")
    axis.set_ylabel("AIME-2024 mean@16 (%)")
    axis.set_xlabel("training step")
    fig.legend(
        handles=tis3_handles if slug == "tis3" else clipped_handles,
        loc="outside lower center", ncol=2, frameon=False, fontsize=8.0,
        handlelength=2.2,
    )
    save(fig, f"result/q30ba3b/curves/{slug}")
