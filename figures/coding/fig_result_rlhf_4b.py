#!/usr/bin/env python
"""RLHF result (Qwen3-4B-Instruct, Anthropic HH-RLHF): GRPO baseline vs cispo3+ESS.
Metric is the reward-model score (critic/score/mean), not AIME. Panels: reward (headline),
entropy, response length. Data from only_for_figures/data/rlhf_{grpo,cispo3_ess}.csv.

-> for_paper/figures_mains/result/rlhf_4b/{overall,reward,entropy,length}.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run, label, colour, linestyle, lw)
SERIES = [
    ("rlhf_grpo",       "GRPO",         FAM[2], "--", 1.2),
    ("rlhf_cispo3_ess", "cispo3 + ESS", FAM[1], "-",  1.6),
]
# (metric, ylabel, panel title, slug)
PANELS = [
    ("reward",  "reward-model score", "(a) reward",          "reward"),
    ("entropy", "policy entropy",     "(b) entropy",         "entropy"),
    ("length",  "mean response length (tokens)", "(c) response length", "length"),
]


def draw(a, metric, ylab):
    for run, lbl, col, ls, lw in SERIES:
        xs, ys = series(run, metric)
        tag = f"{lbl}: {ys[-1]:.1f}" if metric == "reward" else lbl
        a.plot(xs, ys, color=col, ls=ls, lw=lw, marker="o" if metric == "reward" else None,
               ms=2.5, label=tag)
    a.set_ylabel(ylab)
    a.set_xlabel("training step")


use_paper_style()

# --- A) combined ---
fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.5))
for a, (metric, ylab, tag, _slug) in zip(ax, PANELS):
    draw(a, metric, ylab)
    a.set_title(tag, loc="left")
ax[0].legend(loc="lower right")
save(fig, "result/rlhf_4b/overall")

# --- B) one per metric ---
for metric, ylab, tag, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.4))
    draw(a, metric, ylab)
    a.set_title(tag, loc="left")
    a.legend(loc="lower right")
    save(fig, f"result/rlhf_4b/{slug}")
print("rlhf figures done")
