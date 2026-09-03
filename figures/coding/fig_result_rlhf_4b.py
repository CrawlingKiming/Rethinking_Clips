#!/usr/bin/env python
"""Open-ended RLHF transfer on Qwen3-4B-Instruct-2507.

The main figure reports only the optimized reward-model score. Thin curves are
per-step observations; thick curves are trailing seven-step means.

Output:
  figures_mains/result/rlhf_4b/overall.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle
from paperstyle import FULL, C, format_sig, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

RUNS = [
    ("rlhf_grpo", "Clipped", "#4d4d4d", "--", -5),
    ("rlhf_cispo3_ess", "ESS-conditioned", C["ours"], "-", 6),
]


def trailing_mean(values, window=7):
    output = []
    for index in range(len(values)):
        current = values[max(0, index - window + 1): index + 1]
        output.append(sum(current) / len(current))
    return output


def annotate_endpoint(ax, x, y, color, offset):
    ax.annotate(
        format_sig(y),
        xy=(x, y),
        xytext=(5, offset),
        textcoords="offset points",
        color=color,
        fontsize=6.5,
        ha="left",
        va="center",
        clip_on=False,
    )


def draw(ax, metric, ylim, ylabel):
    for run, _label, color, linestyle, offset in RUNS:
        xs, ys = series(run, metric)
        ax.plot(xs, ys, color=color, linewidth=0.55, alpha=0.20)
        ax.plot(
            xs,
            trailing_mean(ys),
            color=color,
            linestyle=linestyle,
            linewidth=1.55 if linestyle == "-" else 1.25,
        )
        annotate_endpoint(ax, xs[-1], ys[-1], color, offset)
    ax.set_xlim(0, 220)
    ax.set_ylim(*ylim)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)


use_paper_style()
fig, axis = plt.subplots(figsize=(0.72 * FULL, 2.35))
draw(axis, "reward", (0, 66), "reward-model score")
axis.set_title("Open-ended RLHF", loc="left")

handles = [
    Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.25,
           label="Clipped"),
    Line2D([0], [0], color=C["ours"], linestyle="-", linewidth=1.55,
           label="ESS-conditioned"),
]
fig.legend(
    handles=handles, loc="outside lower center", ncol=2, frameon=False,
    fontsize=8.0, handlelength=2.2,
)
save(fig, "result/rlhf_4b/overall")
