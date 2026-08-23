#!/usr/bin/env python
"""Training dynamics (Qwen3-30B-A3B): what the gate does to the rollouts themselves.

  response length  -> response_length/mean
  entropy          -> actor/entropy, LINEAR axis
  reward           -> critic/score/mean

Each GRPO variant and DPPO as a baseline (dashed) and with the ESS gate (solid), plus the ungated
TIS 3 collapse run in brick. Colours come from paperstyle.FAM.

TIS 3's gated counterpart is `328rfu6eb2`, already on the plot as "GRPO clip-higher + ESS" (same
run, matched to that baseline by clip band), so it is not drawn twice.

Pairing caveat: the GRPO variants are matched on the clip band, not on an identical loss. Only the
DPPO pair toggles the gate alone. See results_sweep_A_B1.md.

Emits both forms from one set of panel definitions:
  A) one standalone single-column figure per metric
  B) one combined full-width figure with all three

-> for_paper/figures/result/q30ba3b/dynamics/overall.pdf              (A)
-> for_paper/figures/result/q30ba3b/dynamics/{length,entropy,reward}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from paperstyle import COL, FULL, FAM, use_paper_style, save
from runlog import series

# (run, label, colour, linestyle, linewidth)
SERIES = [
    ("grpo_base",       "GRPO",                   FAM[0], "--", 1.1),
    ("grpo_ess_clip",   "GRPO + ESS",             FAM[0], "-",  1.5),
    ("dapo_base",       "GRPO clip-higher",       FAM[1], "--", 1.1),
    ("dapo_ess",        "GRPO clip-higher + ESS", FAM[1], "-",  1.5),
    ("dppo_base",       "DPPO",                   FAM[2], "--", 1.1),
    ("dppo_ess",        "DPPO + ESS",             FAM[2], "-",  1.5),
    ("cispo3_nogate",   "TIS 3, no gate",         FAM[3], "--", 1.3),
]
# (metric, ylabel, panel title, output slug, ylim)
PANELS = [
    ("length",  "mean response length (tokens)", "(a) response length", "length",  (700, 8400)),
    ("entropy", "policy entropy",                "(b) entropy",         "entropy", (0, 5.1)),
    ("reward",  "training reward",               "(c) reward",          "reward",  (0, 0.78)),
]


def draw(a, metric, ylab, ylim):
    for run, lbl, col, ls, lw in SERIES:
        xs, ys = series(run, metric)
        a.plot(xs, ys, color=col, ls=ls, lw=lw, label=lbl)
    a.set_ylabel(ylab)
    a.set_xlabel("training step")
    a.set_xlim(-3, 203)
    a.set_ylim(*ylim)


use_paper_style()

# --- A) one standalone single-column figure per metric ---
for metric, ylab, _tag, slug, ylim in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.7))
    draw(a, metric, ylab, ylim)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"result/q30ba3b/dynamics/{slug}")

# --- B) the same three panels combined, full width ---
fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.5))
for a, (metric, ylab, tag, _slug, ylim) in zip(ax, PANELS):
    draw(a, metric, ylab, ylim)
    a.set_title(tag, loc="left")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
save(fig, "result/q30ba3b/dynamics/overall")
