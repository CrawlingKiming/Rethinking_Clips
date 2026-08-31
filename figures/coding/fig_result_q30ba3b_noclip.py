#!/usr/bin/env python
"""Gate-action ablation for the unclipped Qwen3-30B-A3B update.

Both runs use the same unclipped high-support update. One leaves it unchanged;
the other activates conventional clipping on a low-ESS minibatch. The skip
variant is reported later with the other alternative ESS actions.

Output:
  figures_mains/result/q30ba3b/noclip/overall.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import FULL, C, format_sig, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

VARIANTS = [
    ("noclip_ungated", "no ESS action", "#4d4d4d", "--", 1.2, -8),
    ("noclip_ess_clip", r"ESS $\rightarrow$ clip", C["ours"], "-", 1.6, 7),
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


use_paper_style()
fig, ax = plt.subplots(figsize=(0.72 * FULL, 2.35))
for run, label, color, linestyle, linewidth, offset in VARIANTS:
    xs, ys = series(run, "eval")
    ax.plot(
        xs,
        [100 * value for value in ys],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker="o",
        markersize=2.7,
        label=label,
    )
    annotate_endpoint(ax, xs[-1], ys[-1], color, offset)

ax.set_xlim(-3, 224)
ax.set_ylim(10, 48)
ax.set_xlabel("training step")
ax.set_ylabel("AIME-2024 mean@16 (%)")
ax.set_title("Qwen3-30B-A3B, unclipped update", loc="left")
fig.legend(loc="outside lower center", ncol=2, frameon=False)
save(fig, "result/q30ba3b/noclip/overall")
