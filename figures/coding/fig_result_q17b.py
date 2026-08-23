#!/usr/bin/env python
"""Qwen3-1.7B-base RL (interactive verl-gopo-train, p4d). AIME-2024 mean@16 (val-core; metrics.txt).
The dominant axis is learning rate: lr 1e-5 collapses, lr 1e-6 is stable. Among lr 1e-6, cispo3+ESS
(dppo-latch variant) edges the GRPO baseline; ESS-clip ran short. Small base -> low absolute AIME.

  lr 1e-5:  GRPO `q17b_grpo_lr1e5`  (collapses to 0)          [dashed]
  lr 1e-6:  GRPO `q17b_grpo_lr1e6` · cispo3 ESS-clip `..._essclip_lr1e6` · cispo3 ESS-dppo `..._essdppo_lr1e6`  [solid]

-> for_paper/figures_mains/result/q17b/curve.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (label, runlog key, colour, linestyle)  — dashed = lr 1e-5, solid = lr 1e-6
CURVES = [
    ("GRPO (lr 1e-5)",            "q17b_grpo_lr1e5",           FAM[3], "--"),
    ("GRPO (lr 1e-6)",            "q17b_grpo_lr1e6",           FAM[2], "-"),
    ("cispo3 ESS-clip (lr 1e-6)", "q17b_cispo3_essclip_lr1e6", FAM[0], "-"),
    ("cispo3 ESS-dppo (lr 1e-6)", "q17b_cispo3_essdppo_lr1e6", FAM[1], "-"),
]

use_paper_style()


def fig_eval():
    fig, a = plt.subplots(figsize=(COL * 1.3, 2.6))
    for lbl, key, col, ls in CURVES:
        xs, ys = series(key, "eval")
        a.plot(xs, [v * 100 for v in ys], color=col, ls=ls, lw=1.6, marker="o", ms=2.5,
               label=f"{lbl}: pk {max(ys) * 100:.1f} / fin {ys[-1] * 100:.1f}")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_ylim(-0.5, 15)
    a.set_title("Qwen3-1.7B-base — lr 1e-5 collapses, 1e-6 stable", loc="left")
    a.legend(loc="upper left", fontsize=7)
    return fig, a


def fig_ess():
    fig, a = plt.subplots(figsize=(COL * 1.3, 2.6))
    for lbl, key, col, ls in CURVES:
        try:
            xs, ys = series(key, "ess")
        except KeyError:
            continue
        a.plot(xs, ys, color=col, ls=ls, lw=1.3, label=f"{lbl}: fin {ys[-1]:.2f}")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")
    a.set_ylim(0, 0.8)
    a.set_title("Qwen3-1.7B-base — sequence ESS", loc="left")
    a.legend(loc="upper right", fontsize=7)
    return fig, a


for maker, slug in [(fig_eval, "curve"), (fig_ess, "ess")]:
    fig, a = maker()
    a.set_xlabel("training step")
    save(fig, f"result/q17b/{slug}")
print("q17b figures done")

