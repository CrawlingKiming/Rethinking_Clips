#!/usr/bin/env python
"""Qwen3-1.7B scale-transfer plots for Section 6.

Only the matched learning-rate comparison is shown: GRPO and the run reported
in the paper as TIS-3 + Clip. The stored run key retains its original internal
name; display labels use the paper terminology.

Outputs:
  figures_mains/result/q17b/aime.pdf
  figures_mains/result/q17b/ess.pdf
  figures_mains/result/q17b/overall.pdf
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

RUNS = [
    ("GRPO", "q17b_grpo_lr1e6", "#4d4d4d", "--", -7),
    ("TIS-3 + Clip", "q17b_cispo3_essdppo_lr1e6", C["ours"], "-", 7),
]
def trailing_mean(values, window=7):
    return [
        sum(values[max(0, index - window + 1): index + 1])
        / len(values[max(0, index - window + 1): index + 1])
        for index in range(len(values))
    ]


def annotate_endpoint(ax, x, y, color, offset, suffix=""):
    ax.annotate(
        f"{format_sig(y)}{suffix}",
        xy=(x, y), xytext=(4, offset), textcoords="offset points",
        color=color, fontsize=6.8, ha="left", va="center", clip_on=False,
    )


def method_handles():
    return [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.6,
               marker="o", markersize=2.7, label=label)
        for label, _key, color, linestyle, _offset in RUNS
    ]


def make_aime():
    fig, axis = plt.subplots(figsize=(COL, 2.48))
    for _label, key, color, linestyle, offset in RUNS:
        xs, ys = series(key, "eval")
        values = [100 * value for value in ys]
        axis.plot(
            xs, values, color=color, linestyle=linestyle, linewidth=1.6,
            marker="o", markersize=2.7,
        )
        annotate_endpoint(axis, xs[-1], values[-1], color, offset, "%")
    axis.set_xlim(-5, 515)
    axis.set_ylim(2, 13.5)
    axis.set_xlabel("training step")
    axis.set_ylabel("AIME-2024 mean@16 (%)")
    axis.set_title("AIME", loc="left", fontweight="bold")
    fig.legend(
        handles=method_handles(), loc="outside lower center", ncol=2,
        frameon=False, fontsize=8.0, handlelength=2.2,
    )
    return fig


def make_ess():
    fig, axis = plt.subplots(figsize=(COL, 2.48))
    for _label, key, color, linestyle, offset in RUNS:
        xs, ys = series(key, "ess")
        smooth = trailing_mean(ys)
        axis.plot(xs, ys, color=color, linewidth=0.5, alpha=0.16)
        axis.plot(xs, smooth, color=color, linestyle=linestyle, linewidth=1.6)
        annotate_endpoint(axis, xs[-1], smooth[-1], color, offset)
    axis.set_xlim(-5, 515)
    axis.set_ylim(0.24, 0.68)
    axis.set_xlabel("training step")
    axis.set_ylabel("ESS")
    axis.set_title("ESS", loc="left", fontweight="bold")
    fig.legend(
        handles=method_handles(), loc="outside lower center", ncol=2,
        frameon=False, fontsize=8.0, handlelength=2.2,
    )
    return fig


def make_overall():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), sharex=True)
    aime_axis, ess_axis = axes

    for _label, key, color, linestyle, offset in RUNS:
        eval_x, eval_y = series(key, "eval")
        ess_x, ess_y = series(key, "ess")
        eval_values = [100 * value for value in eval_y]
        ess_values = trailing_mean(ess_y)
        aime_axis.plot(
            eval_x, eval_values, color=color,
            linestyle=linestyle, linewidth=1.65, marker="o", markersize=2.6,
        )
        ess_axis.plot(ess_x, ess_y, color=color, linewidth=0.5, alpha=0.16)
        ess_axis.plot(
            ess_x, ess_values, color=color,
            linestyle=linestyle, linewidth=1.45,
        )
        annotate_endpoint(
            aime_axis, eval_x[-1], eval_values[-1], color, offset, "%"
        )
        annotate_endpoint(ess_axis, ess_x[-1], ess_values[-1], color, offset)

    aime_axis.set_xlim(-5, 515)
    aime_axis.set_ylim(2, 13.5)
    aime_axis.set_xlabel("training step")
    aime_axis.set_ylabel("AIME-2024 mean@16 (%)")
    aime_axis.set_title("(a) AIME", loc="left", fontweight="bold")

    ess_axis.set_xlim(-5, 515)
    ess_axis.set_ylim(0.24, 0.68)
    ess_axis.set_xlabel("training step")
    ess_axis.set_ylabel("ESS")
    ess_axis.set_title("(b) ESS", loc="left", fontweight="bold")

    fig.legend(
        handles=method_handles(), loc="outside lower center", ncol=2,
        frameon=False, fontsize=8.5, handlelength=2.4, columnspacing=1.8,
    )
    return fig


use_paper_style()
save(make_aime(), "result/q17b/aime")
# Keep the legacy path synchronized with the new two-run AIME plot.
save(make_aime(), "result/q17b/curve")
save(make_ess(), "result/q17b/ess")
save(make_overall(), "result/q17b/overall")
print("q17b figures done")
