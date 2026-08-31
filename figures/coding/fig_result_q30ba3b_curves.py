#!/usr/bin/env python
"""Core Qwen3-30B-A3B trajectories for Section 6.

The main figure keeps the two comparisons that most directly expose the
static-versus-conditional transition. GRPO and clip-higher band-matched
comparisons remain numerical supporting evidence in the text; placing them in
the same figure would make four panels look like four identical gate ablations.

Output:
  figures_mains/result/q30ba3b/curves/overall.pdf
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
    ("DPPO", "dppo_base", "dppo_ess", (-1.5, 1.2)),
    ("TIS 3", "cispo3_nogate", "dapo_ess", (1.8, 1.0)),
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
    ax.set_ylim(-1.5, 50)
    ax.set_xlabel("training step")


use_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35), sharex=True, sharey=True)
for index, (title, static_run, conditional_run, offsets) in enumerate(PANELS):
    draw(axes[index], static_run, conditional_run, offsets)
    axes[index].set_title(f"({'ab'[index]}) {title}", loc="left")
axes[0].set_ylabel("AIME-2024 mean@16 (%)")

handles = [
    Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.2,
           marker="o", markersize=2.7, label="static update"),
    Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.6,
           marker="o", markersize=2.7, label="ESS-conditioned update"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
save(fig, "result/q30ba3b/curves/overall")
