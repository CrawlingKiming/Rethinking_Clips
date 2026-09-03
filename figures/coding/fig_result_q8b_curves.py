#!/usr/bin/env python
"""Qwen3-8B recipe comparisons for Section 6.

The two panels use the same visual grammar as the primary Qwen3-30B-A3B
result: an always-clipped recipe in dashed gray and its ESS-conditioned
counterpart in solid red.

Output:
  figures_mains/result/8b/curves/overall.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle
from paperstyle import FULL, C, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

STATIC = "#4d4d4d"
CONDITIONAL = C["ours"]
PANELS = [
    ("GRPO", "q8b_grpo_base", "q8b_grpo_ess", (-6, 7)),
    ("Clip-Higher", "q8b_dapo_base", "q8b_dapo_ess_nonorm", (-5, 6)),
]


def annotate_endpoint(ax, x, y, color, offset):
    ax.annotate(
        f"{100 * y:.1f}%",
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
    ax.set_ylim(10, 36)


use_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.42), sharex=True, sharey=True)
for index, (title, static_run, conditional_run, offsets) in enumerate(PANELS):
    draw(axes[index], static_run, conditional_run, offsets)
    axes[index].set_title(
        f"({'ab'[index]}) {title}",
        loc="left",
        fontsize=8.4,
        fontweight="bold",
    )
    axes[index].set_xticks([0, 100, 200])
    axes[index].set_yticks([10, 20, 30])
    axes[index].tick_params(axis="both", labelsize=7.2)
    axes[index].grid(True, which="major", alpha=0.26)
axes[0].set_ylabel("AIME-2024 mean@16 (%)")
axes[0].yaxis.label.set_size(8.0)
fig.supxlabel("training step", fontsize=8.0)

handles = [
    Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.2,
           marker="o", markersize=2.7, label="Clipped"),
    Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.6,
           marker="o", markersize=2.7, label="ESS-conditioned"),
]
fig.legend(
    handles=handles, loc="outside upper center", ncol=2, frameon=False,
    fontsize=8.2, handlelength=2.3,
)
save(fig, "result/8b/curves/overall")
