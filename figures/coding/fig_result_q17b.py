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
from paperstyle import COL, C, format_sig, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

RUNS = [
    ("GRPO", "q17b_grpo_lr1e6", "#4d4d4d", "--", -7),
    ("TIS-3 + Clip", "q17b_cispo3_essdppo_lr1e6", C["ours"], "-", 7),
]
AIME_COLOR = "#202020"
ESS_COLOR = C["gated"]


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
    axis.set_title("AIME-2024", loc="left", fontweight="bold")
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
    axis.set_ylabel("normalized sequence ESS")
    axis.set_title("Sequence ESS", loc="left", fontweight="bold")
    fig.legend(
        handles=method_handles(), loc="outside lower center", ncol=2,
        frameon=False, fontsize=8.0, handlelength=2.2,
    )
    return fig


def make_overall():
    fig, left = plt.subplots(figsize=(COL, 2.72))
    right = left.twinx()

    for _label, key, _color, linestyle, _offset in RUNS:
        eval_x, eval_y = series(key, "eval")
        ess_x, ess_y = series(key, "ess")
        left.plot(
            eval_x, [100 * value for value in eval_y], color=AIME_COLOR,
            linestyle=linestyle, linewidth=1.65, marker="o", markersize=2.6,
        )
        right.plot(
            ess_x, trailing_mean(ess_y), color=ESS_COLOR,
            linestyle=linestyle, linewidth=1.45,
        )

    left.set_xlim(-5, 515)
    left.set_ylim(2, 13.5)
    right.set_ylim(0.24, 0.68)
    left.set_xlabel("training step")
    left.set_ylabel("AIME-2024 mean@16 (%)", color=AIME_COLOR)
    right.set_ylabel("normalized sequence ESS", color=ESS_COLOR)
    left.tick_params(axis="y", colors=AIME_COLOR)
    right.tick_params(axis="y", colors=ESS_COLOR)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(ESS_COLOR)
    right.grid(False)
    left.set_title("Qwen3-1.7B", loc="left", fontweight="bold")

    handles = [
        Line2D([0], [0], color=AIME_COLOR, linewidth=1.65, marker="o",
               markersize=2.6, label="AIME"),
        Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.5,
               label="GRPO"),
        Line2D([0], [0], color=ESS_COLOR, linewidth=1.45, label="ESS"),
        Line2D([0], [0], color="#4d4d4d", linestyle="-", linewidth=1.5,
               label="TIS-3 + Clip"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, frameon=False,
        fontsize=7.8, handlelength=2.2, columnspacing=1.3,
    )
    return fig


use_paper_style()
save(make_aime(), "result/q17b/aime")
# Keep the legacy path synchronized with the new two-run AIME plot.
save(make_aime(), "result/q17b/curve")
save(make_ess(), "result/q17b/ess")
save(make_overall(), "result/q17b/overall")
print("q17b figures done")
