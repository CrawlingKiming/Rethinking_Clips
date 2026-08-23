#!/usr/bin/env python
"""Result figure (Qwen3-30B-A3B): baselines vs gate-conditional updates.

Read results_sweep_A_B1.md "Gate ablation" before relabelling this figure. Only (c) and (d) are gate toggles; (a) and (b) are matched on the **clip band**, not on the loss:

  (a) GRPO, band 0.2/0.2        `udg7vbgfsn` clips every step  vs `ircyhpdmku` TIS 3 base + ESS
  (b) GRPO clip-higher, 0.2/0.28 `t82djeyx43` clips every step  vs `328rfu6eb2` TIS 3 base + ESS
  (c) DPPO                       `z95e8ih6mr` no gate           vs `52iya9e2hr` same loss + ESS
  (d) TIS 3                      `sjjc7dcpzf` no gate           vs `328rfu6eb2` + ESS (with latch)

(a) and (b) therefore compare clipping on *every* step against falling back to that same clip *only*
when ESS says the batch has gone off-policy. An identical-loss with-and-without-gate pair does not
exist for either GRPO variant; no such run was trained. TIS 5 is plotted separately in tis5.pdf,
because its gated partner has no held-out number yet.

Note `328rfu6eb2` appears in both (b) and (d): it is simultaneously the clip-higher-band
ESS-conditional run and the TIS-3-based gated run. That is a property of the run matrix, not a
duplication error.

Legend numbers are the final in-training validation step, not the held-out sweep, and the two
differ (STAR: 44.8 in-training vs 42.1 held-out). Held-out numbers live in results_sweep_A_B1.md.

-> for_paper/figures/result/q30ba3b/curves/overall.pdf  (A)
-> for_paper/figures/result/q30ba3b/curves/{grpo,grpo_cliphigher,dppo,tis3}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (panel title, baseline run, baseline label, gated run, gated label)
PANELS = [
    ("GRPO", "grpo_base", "GRPO",
     "grpo_ess_clip", "ESS-conditional clip", "grpo"),
    ("GRPO clip-higher", "dapo_base", "GRPO clip-higher",
     "dapo_ess", "ESS-conditional clip", "grpo_cliphigher"),
    ("DPPO", "dppo_base", "DPPO", "dppo_ess", "+ ESS gate", "dppo"),
    ("TIS 3", "cispo3_nogate", "TIS 3, no gate", "dapo_ess", "+ ESS gate", "tis3"),
]


def draw(a, b_run, b_lbl, g_run, g_lbl):
    for run, lbl, col, ls in [(b_run, b_lbl, C["baseline2"], "--"),
                              (g_run, g_lbl, C["gated"], "-")]:
        xs, ys = series(run, "eval")
        a.plot(xs, [v * 100 for v in ys], color=col, ls=ls, marker="o",
               lw=1.6 if ls == "-" else 1.2, label=f"{lbl}: {ys[-1] * 100:.1f}%")
    a.set_ylim(-1.5, 52)
    a.set_xlim(-3, 203)
    a.legend(loc="lower left")

use_paper_style()

# --- A) overall ---
fig, ax = plt.subplots(2, 2, figsize=(FULL, 4.2), sharey=True, sharex=True)
flat = ax.ravel()
for i, (tag, b_run, b_lbl, g_run, g_lbl, _slug) in enumerate(PANELS):
    a = flat[i]
    draw(a, b_run, b_lbl, g_run, g_lbl)
    a.set_title(f"({'abcd'[i]}) {tag}", loc="left")
    if i % 2 == 0:
        a.set_ylabel("AIME-2024 mean@16 (%)")
    if i >= 2:
        a.set_xlabel("training step")
save(fig, "result/q30ba3b/curves/overall")

# --- B) one per panel ---
for tag, b_run, b_lbl, g_run, g_lbl, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.35))
    draw(a, b_run, b_lbl, g_run, g_lbl)
    a.set_title(tag, loc="left")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_xlabel("training step")
    save(fig, f"result/q30ba3b/curves/{slug}")
