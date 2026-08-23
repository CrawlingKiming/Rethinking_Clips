#!/usr/bin/env python
"""Result figure (Qwen2.5-7B): GRPO vs CISPO-3 + ESS gate.

Two runs, both 200 steps / 2 epochs, 8xB200, git sha 02c84dfc. Validation is MATH-500 mean@1
(there is NO AIME val set in these runs), reward is critic/score/mean logged every step.

  GRPO            w5rwzuttpv   -> runlog q257b_grpo
  CISPO-3 + ESS   qzrn8vpezj   -> runlog q257b_cispo3_ess   (clip-latch)

-> for_paper/figures_mains/result/qwen257b/reward.{pdf,png}
-> for_paper/figures_mains/result/qwen257b/math500_eval.{pdf,png}
-> for_paper/figures_mains/result/qwen257b/overall.{pdf,png}   (reward | eval, side by side)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (short name, label, colour, linestyle)
RUNS = [
    ("q257b_grpo",       "GRPO",          C["baseline2"], "--"),
    ("q257b_cispo3_ess", "CISPO-3 + ESS", C["gated"],     "-"),
]


def save_both(fig, name):
    """PDF (paper) + PNG (quick view) into figures_mains/<name>."""
    png = os.path.join(paperstyle.FIGDIR, name + ".png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=200)
    save(fig, name)  # writes the .pdf and closes the figure


def draw_reward(a):
    for run, lbl, col, ls in RUNS:
        xs, ys = series(run, "reward")
        a.plot(xs, ys, color=col, ls=ls, lw=1.3, label=f"{lbl}: {ys[-1]:.3f}")
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
    a.set_ylabel("train reward (mean)")
    a.legend(loc="lower right")


def draw_eval(a):
    for run, lbl, col, ls in RUNS:
        xs, ys = series(run, "eval_math500")
        pct = [v * 100 for v in ys]
        peak, final = max(pct), pct[-1]
        a.plot(xs, pct, color=col, ls=ls, marker="o",
               lw=1.6 if ls == "-" else 1.2, label=f"{lbl}: pk {peak:.1f} / fin {final:.1f}")
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
    a.set_ylabel("MATH-500 acc (%)")
    a.legend(loc="lower right")


use_paper_style()

# --- reward (single) ---
fig, a = plt.subplots(figsize=(COL, 2.35))
draw_reward(a)
a.set_title("Qwen2.5-7B: training reward", loc="left")
save_both(fig, "result/qwen257b/reward")

# --- MATH-500 eval (single) ---
fig, a = plt.subplots(figsize=(COL, 2.35))
draw_eval(a)
a.set_title("Qwen2.5-7B: MATH-500", loc="left")
save_both(fig, "result/qwen257b/math500_eval")

# --- combined (reward | eval) ---
fig, (a0, a1) = plt.subplots(1, 2, figsize=(FULL, 2.4))
draw_reward(a0)
a0.set_title("(a) training reward", loc="left")
draw_eval(a1)
a1.set_title("(b) MATH-500 accuracy", loc="left")
save_both(fig, "result/qwen257b/overall")
