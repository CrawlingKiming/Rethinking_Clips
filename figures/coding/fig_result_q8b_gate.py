#!/usr/bin/env python
"""Gate diagnostics (Qwen3-8B): what the gate sees and how often it clips.

  (a) normalized sequence ESS per step, linear axis, with the 0.1 gate threshold
  (b) fraction of updates clipped, corrected by runlog.gate_fraction()

The three ESS-gate variants (clip mode). Colours match the dynamics figure:
FAM[0] GRPO, FAM[1] clip-higher, FAM[2] DPPO.

-> for_paper/figures_mains/result/8b/gate/overall.pdf  (A)
-> for_paper/figures_mains/result/8b/gate/{ess,clipped}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series, gate_fraction

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run, label, colour)
GATED = [
    ("q8b_grpo_ess",        "GRPO + ESS",             FAM[0]),
    ("q8b_dapo_ess_nonorm", "GRPO clip-higher + ESS", FAM[1]),
    ("q8b_dppo_ess",        "DPPO + ESS",             FAM[2]),
]


def draw_ess(a):
    for run, lbl, col in GATED:
        xs, ys = series(run, "ess")
        a.plot(xs, ys, color=col, lw=1.2, label=lbl)
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")


def draw_clipped(a):
    for run, lbl, col in GATED:
        xs, ys = gate_fraction(run, "clipped")
        a.plot(xs, ys, color=col, lw=1.0, alpha=0.95, label=lbl)
    a.set_ylabel("fraction of updates clipped")
    a.set_ylim(-0.03, 1.06)


PANELS = [(draw_ess, "what the gate measures", "ess"),
          (draw_clipped, "how often the gate clips", "clipped")]

use_paper_style()

# --- A) overall ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.3))
for i, (fn, tag, _slug) in enumerate(PANELS):
    fn(ax[i])
    ax[i].set_title(f"({'ab'[i]}) {tag}", loc="left")
    ax[i].set_xlim(-3, 203)
    ax[i].set_xlabel("training step")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
save(fig, "result/8b/gate/overall")

# --- B) one per panel ---
for fn, tag, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.5))
    fn(a)
    a.set_title(tag, loc="left")
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"result/8b/gate/{slug}")
