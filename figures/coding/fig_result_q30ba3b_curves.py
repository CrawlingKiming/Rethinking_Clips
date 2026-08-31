#!/usr/bin/env python
"""Core Qwen3-30B-A3B trajectories for Section 6.

All conditional curves use TIS 3 while ESS is high and activate the named
clipping safeguard when ESS is low. The first three panels compare this
template with an always-on safeguard; the fourth compares it directly with
ungated TIS 3, exposing the collapse that conditioning prevents.

Output:
  figures_mains/result/q30ba3b/curves/overall.pdf
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
    ("TIS 3", "cispo3_nogate", "dapo_ess", (8, 6), "tis3"),
]


def annotate_endpoint(ax, x, y, color, offset):
    ax.annotate(
        f"{format_sig(100 * y)}%",
        xy=(x, 100 * y),
        xytext=(5, offset),
        textcoords="offset points",
        color=color,
        fontsize=6.5,
        ha="left",
        va="center",
        clip_on=False,
    )


def draw(ax, static_run, conditional_run, offsets):
    for run, color, linestyle, linewidth, offset in [
        (static_run, STATIC, "--", 1.2, offsets[0]),
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
        annotate_endpoint(ax, xs[-1], ys[-1], color, offset)
    ax.set_xlim(-3, 222)
    ax.set_ylim(-1.5, 50)


handles = [
    Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.2,
           marker="o", markersize=2.7, label="static comparison"),
    Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.6,
           marker="o", markersize=2.7,
           label=r"TIS 3 $\rightarrow$ low-ESS safeguard"),
]

use_paper_style()
fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.15), sharex=True, sharey=True)
for index, (title, static_run, conditional_run, offsets, _slug) in enumerate(PANELS):
    axis = axes.ravel()[index]
    draw(axis, static_run, conditional_run, offsets)
    axis.set_title(f"({'abcd'[index]}) {title}", loc="left")
    if index % 2 == 0:
        axis.set_ylabel("AIME-2024 mean@16 (%)")
    if index >= 2:
        axis.set_xlabel("training step")
fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
save(fig, "result/q30ba3b/curves/overall")

for title, static_run, conditional_run, offsets, slug in PANELS:
    fig, axis = plt.subplots(figsize=(COL, 2.35))
    draw(axis, static_run, conditional_run, offsets)
    axis.set_title(title, loc="left")
    axis.set_ylabel("AIME-2024 mean@16 (%)")
    axis.set_xlabel("training step")
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"result/q30ba3b/curves/{slug}")
