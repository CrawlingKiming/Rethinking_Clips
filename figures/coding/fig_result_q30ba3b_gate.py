#!/usr/bin/env python
"""Realized ESS-conditioned intervention schedule on Qwen3-30B-A3B.

Each panel pairs the gate's ESS statistic with the raw intervention fraction.
GRPO and DPPO use raw complete-sequence ESS; the Clip-Higher construction uses
post-cap shaped ESS. Thin black traces are per-step intervention fractions and
thick black traces are trailing seven-step means.

Output:
  figures_mains/result/q30ba3b/gate/overall.pdf
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

PANELS = [
    ("GRPO", "grpo_ess_clip", "ess"),
    ("Clip-Higher", "dapo_ess", "ess_shaped"),
    ("DPPO", "dppo_ess", "ess"),
]


def trailing_mean(values, window=7):
    output = []
    for index in range(len(values)):
        current = values[max(0, index - window + 1): index + 1]
        output.append(sum(current) / len(current))
    return output


def draw(ax, run, ess_metric, show_left, show_right):
    tx, trip = series(run, "trip")
    ax.plot(tx, trip, color=C["eval"], linewidth=0.55, alpha=0.18)
    ax.plot(tx, trailing_mean(trip), color=C["eval"], linewidth=1.45)
    ax.set_xlim(-3, 203)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training step")
    ax.tick_params(axis="y", labelleft=show_left)
    if show_left:
        ax.set_ylabel("intervention fraction")

    ex, ess = series(run, ess_metric)
    right = ax.twinx()
    right.plot(ex, ess, color=C["ours"], linewidth=1.0)
    right.axhline(0.1, color=C["ours"], linestyle=(0, (1, 2)), linewidth=0.9)
    right.set_ylim(0, 0.68)
    right.grid(False)
    right.spines["right"].set_visible(show_right)
    right.spines["right"].set_color(C["ours"])
    right.tick_params(
        axis="y",
        colors=C["ours"],
        right=show_right,
        labelright=show_right,
    )
    if show_right:
        right.set_ylabel("ESS", color=C["ours"])


use_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.25))
for index, (title, run, ess_metric) in enumerate(PANELS):
    draw(
        axes[index],
        run,
        ess_metric,
        show_left=index == 0,
        show_right=index == len(PANELS) - 1,
    )
    axes[index].set_title(f"({'abc'[index]}) {title}", loc="left")

handles = [
    Line2D([0], [0], color=C["eval"], linewidth=1.45,
           label="intervention (7-step mean)"),
    Line2D([0], [0], color=C["ours"], linewidth=1.0, label="ESS"),
    Line2D([0], [0], color=C["ours"], linestyle=(0, (1, 2)), linewidth=0.9,
           label="threshold 0.1"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
save(fig, "result/q30ba3b/gate/overall")
