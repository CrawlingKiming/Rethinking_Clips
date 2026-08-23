#!/usr/bin/env python
"""Batch-size (PPO minibatch) ablation on Qwen3-30B, cispo3 + ESS-clip, only PPO_MINI_BATCH_SIZE
varies. Updates per rollout = 256 / PPO_MINI_BATCH_SIZE:

  mb32  `uz5xrdzr9k`   8 updates (default)   -> runlog key q30b_mb32
  mb16  `3tw5bvbqiu`  16 updates             -> runlog key q30b_mb16
  mb8   `g5q2wcdp9q`  32 updates             -> runlog key q30b_mb8

Two panels: (a) AIME-2024 mean@16, (b) normalized sequence ESS (with the 0.1 gate line) — more
updates/rollout should push ESS lower (more off-policy) and stress the gate harder.

NOTE: needs the 3 mb-run CSVs in only_for_figures/data/ (fetch the runs from Bolt, then
`python coding/build_figure_data.py`). Until then this raises KeyError.

-> for_paper/figures_mains/result/q30ba3b/batchsize/{overall,eval,ess}.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run key, label, colour)
RUNS = [
    ("q30b_mb8",  "mb8  (32 updates)", FAM[3]),
    ("q30b_mb16", "mb16 (16 updates)", FAM[1]),
    ("q30b_mb32", "mb32 (8 updates)",  FAM[0]),
]

use_paper_style()


def draw_eval(a):
    for run, lbl, col in RUNS:
        xs, ys = series(run, "eval", cut=200)
        a.plot(xs, [v * 100 for v in ys], color=col, lw=1.6, marker="o", ms=3,
               label=f"{lbl}: pk {max(ys) * 100:.1f} / fin {ys[-1] * 100:.1f}")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.legend(loc="lower right")


def draw_ess(a):
    for run, lbl, col in RUNS:
        xs, ys = series(run, "ess", cut=200)
        frac = 100.0 * sum(v < 0.1 for v in ys) / len(ys)
        a.plot(xs, ys, color=col, lw=1.3, label=f"{lbl}: {frac:.0f}% < 0.1")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")
    a.legend(loc="upper right")


# --- A) overall: eval + ESS ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.6))
for a, (fn, tag) in zip(ax, [(draw_eval, "(a) AIME-2024"), (draw_ess, "(b) sequence ESS")]):
    fn(a)
    a.set_title(tag, loc="left")
    a.set_xlim(-3, 205)
    a.set_xlabel("training step")
save(fig, "result/q30ba3b/batchsize/overall")

# --- B) separates ---
for fn, slug in [(draw_eval, "eval"), (draw_ess, "ess")]:
    fig, a = plt.subplots(figsize=(COL, 2.5))
    fn(a)
    a.set_xlim(-3, 205)
    a.set_xlabel("training step")
    save(fig, f"result/q30ba3b/batchsize/{slug}")
print("batchsize figures done")
