#!/usr/bin/env python
"""Result figure (Qwen3-8B): baselines vs gate-conditional updates, one panel per method.

Baselines are the **conventional token-mean** GRPO / GRPO-clip-higher runs (verl's implemented
loss_agg_mode = seq-mean-token-mean), NOT the 0813 `*_alwaysclip_ref` runs (those are sum-norm).

  (a) GRPO, band 0.2/0.2         `yfs6ms6w6a` GRPO token-mean   vs `n2bu8xky6c` ESS-clip 0.2/0.2
  (b) GRPO clip-higher, 0.2/0.28 `5sra49tycr` DAPO token-mean   vs `bxjnvy6f3s` ESS-clip 0.2/0.28
  (c) DPPO (total-variation)     `zrmqamex4j` latch/step        vs `vdfa57r99z` dppo-tv + ESS

Aggregation match: (b) is clean — baseline AND ESS run are both token-mean (`5sra49tycr` /
`bxjnvy6f3s`). (a) baseline is token-mean but its ESS partner `n2bu8xky6c` is sum-norm (no
token-mean plain-GRPO-band ESS run was trained — an open cell). (c) DPPO is not a verl-conventional
loss, kept as-is. No cispo/TIS-only no-gate run at 8B, so no (d) panel; not needed.
Legend numbers are final in-training AIME-2024 mean@16. Saves to figures_mains/result/8b/curves/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, use_paper_style, save
from runlog import series

# route output into figures_mains/ (where the paper's main figures live), not figures/
paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (panel title, baseline run, baseline label, gated run, gated label, slug)
PANELS = [
    ("GRPO", "q8b_grpo_base", "GRPO",
     "q8b_grpo_ess", "ESS-conditional clip", "grpo"),
    ("GRPO clip-higher", "q8b_dapo_base", "GRPO clip-higher",
     "q8b_dapo_ess_nonorm", "ESS-conditional clip", "grpo_cliphigher"),
    ("DPPO", "q8b_dppo_alwayslatch", "DPPO", "q8b_dppo_ess", "+ ESS gate", "dppo"),
]


def draw(a, b_run, b_lbl, g_run, g_lbl):
    for run, lbl, col, ls in [(b_run, b_lbl, C["baseline2"], "--"),
                              (g_run, g_lbl, C["gated"], "-")]:
        xs, ys = series(run, "eval")
        a.plot(xs, [v * 100 for v in ys], color=col, ls=ls, marker="o",
               lw=1.6 if ls == "-" else 1.2, label=f"{lbl}: {ys[-1] * 100:.1f}%")
    a.set_ylim(8, 38)
    a.set_xlim(-3, 203)
    a.legend(loc="lower right")

use_paper_style()

# --- A) overall (1x3) ---
fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.1), sharey=True)
for i, (tag, b_run, b_lbl, g_run, g_lbl, _slug) in enumerate(PANELS):
    a = ax[i]
    draw(a, b_run, b_lbl, g_run, g_lbl)
    a.set_title(f"({'abc'[i]}) {tag}", loc="left")
    a.set_xlabel("training step")
    if i == 0:
        a.set_ylabel("AIME-2024 mean@16 (%)")
save(fig, "result/8b/curves/overall")

# --- B) one per method ---
for tag, b_run, b_lbl, g_run, g_lbl, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.35))
    draw(a, b_run, b_lbl, g_run, g_lbl)
    a.set_title(tag, loc="left")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_xlabel("training step")
    save(fig, f"result/8b/curves/{slug}")
