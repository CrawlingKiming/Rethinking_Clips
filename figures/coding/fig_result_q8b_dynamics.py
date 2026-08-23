#!/usr/bin/env python
"""Training dynamics (Qwen3-8B): what the gate does to the rollouts themselves.

  response length  -> response_length/mean
  entropy          -> actor/entropy, LINEAR axis
  reward           -> critic/score/mean

Each method as a baseline (dashed) and with the ESS gate (solid). Colours from paperstyle.FAM.
No TIS-only no-gate run was trained at 8B, so (unlike 30B) there is no ungated-collapse line here.

-> for_paper/figures_mains/result/8b/dynamics/overall.pdf              (A)
-> for_paper/figures_mains/result/8b/dynamics/{length,entropy,reward}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run, label, colour, linestyle, linewidth)
SERIES = [
    ("q8b_grpo_base",        "GRPO",                   FAM[0], "--", 1.1),
    ("q8b_grpo_ess",         "GRPO + ESS",             FAM[0], "-",  1.5),
    ("q8b_dapo_base",        "GRPO clip-higher",       FAM[1], "--", 1.1),
    ("q8b_dapo_ess_nonorm",  "GRPO clip-higher + ESS", FAM[1], "-",  1.5),
    ("q8b_dppo_alwayslatch", "DPPO",                   FAM[2], "--", 1.1),
    ("q8b_dppo_ess",         "DPPO + ESS",             FAM[2], "-",  1.5),
]
# (metric, ylabel, panel title, output slug)
PANELS = [
    ("length",  "mean response length (tokens)", "(a) response length", "length"),
    ("entropy", "policy entropy",                "(b) entropy",         "entropy"),
    ("reward",  "training reward",               "(c) reward",          "reward"),
]


def draw(a, metric, ylab):
    for run, lbl, col, ls, lw in SERIES:
        xs, ys = series(run, metric)
        a.plot(xs, ys, color=col, ls=ls, lw=lw, label=lbl)
    a.set_ylabel(ylab)
    a.set_xlabel("training step")
    a.set_xlim(-3, 203)


use_paper_style()

# --- A) one standalone single-column figure per metric ---
for metric, ylab, _tag, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.7))
    draw(a, metric, ylab)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"result/8b/dynamics/{slug}")

# --- B) the same three panels combined, full width ---
fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.5))
for a, (metric, ylab, tag, _slug) in zip(ax, PANELS):
    draw(a, metric, ylab)
    a.set_title(tag, loc="left")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
save(fig, "result/8b/dynamics/overall")
