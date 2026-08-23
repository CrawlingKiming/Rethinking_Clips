#!/usr/bin/env python
"""Skip-gate ablation (Qwen3-30B-A3B): what a veto-only gate buys, on two different bases.

The skip gate does not modify the update, it drops it: when ESS falls below 0.1 the step is
vetoed. Two bases, gate off against ESS.skip.0.1:

    no-clip PG (cap inf)   `bvrscfn6u8` ungated  ->  `vm7vcynvy7` (92.5 skips)
    TIS 5                  `ayv2ajeuqk` no gate  ->  `rdq6r5yy83` (only the gate toggles)

The overall figure is eval only; grad_norm is a diagnostic and has its own plot, on a log axis
because it spans 26.8 to 5.9e5 here.

Read the two columns differently. On no-clip PG the skip gate buys stability at a real cost in peak
(44.4 -> 35.6) while removing the grad_norm blow-up entirely (5.9e5 -> 26.8): a veto keeps the run
safe but throws away the updates that were driving the peak. On TIS 5 the same gate converts an
outright collapse (0.0%) into 39.4%. So a skip gate is a rescue for a run that would die and a
handicap for one that would not, which is the argument for clipping rather than skipping when the
estimator is merely unreliable instead of hopeless.

-> for_paper/figures/result/q30ba3b/skip_gate/overall.pdf  (A, eval)
-> for_paper/figures/result/q30ba3b/skip_gate/{grad_norm,noclip_*,tis5_*}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series

# (column title, ungated run, ungated label, skip-gated run, skip-gated label)
COLUMNS = [
    ("no-clip PG (cap inf)", "noclip_ungated", "ungated",
     "noclip_ess_skip", "+ ESS skip", "noclip"),
    ("TIS 5", "cispo5_nogate", "no gate", "cispo5_ess", "+ ESS skip", "tis5"),
]

def draw(a, metric, u_run, u_lbl, g_run, g_lbl):
    for run, lbl, col, ls, lw in [(u_run, u_lbl, C["baseline2"], "--", 1.2),
                                  (g_run, g_lbl, FAM[2], "-", 1.5)]:
        xs, ys = series(run, metric)
        if metric == "eval":
            ys = [v * 100 for v in ys]
            lbl = f"{lbl}: pk {max(ys):.1f} / fin {ys[-1]:.1f}"
        else:
            lbl = f"{lbl}: max {max(ys):.2g}"
        a.plot(xs, ys, color=col, ls=ls, lw=lw,
               marker="o" if metric == "eval" else None, label=lbl)
    if metric == "grad_norm":
        a.set_yscale("log")
    else:
        a.set_ylim(-1.5, 50)
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
    a.legend(loc="lower left" if metric == "eval" else "lower right")


YLAB = {"eval": "AIME-2024 mean@16 (%)", "grad_norm": "grad_norm"}

use_paper_style()

# --- A) overall: eval for both bases. grad_norm is a diagnostic and gets its own plot. ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
for i, (tag, u_run, u_lbl, g_run, g_lbl, _slug) in enumerate(COLUMNS):
    draw(ax[i], "eval", u_run, u_lbl, g_run, g_lbl)
    ax[i].set_title(f"({'ab'[i]}) {tag}", loc="left")
ax[0].set_ylabel(YLAB["eval"])
save(fig, "result/q30ba3b/skip_gate/overall")

# --- B) separates: grad_norm for both bases, then one figure per base ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
for i, (tag, u_run, u_lbl, g_run, g_lbl, _slug) in enumerate(COLUMNS):
    draw(ax[i], "grad_norm", u_run, u_lbl, g_run, g_lbl)
    ax[i].set_title(f"({'ab'[i]}) {tag}", loc="left")
ax[0].set_ylabel(YLAB["grad_norm"])
save(fig, "result/q30ba3b/skip_gate/grad_norm")

for tag, u_run, u_lbl, g_run, g_lbl, slug in COLUMNS:
    for metric in ["eval", "grad_norm"]:
        fig, a = plt.subplots(figsize=(COL, 2.5))
        draw(a, metric, u_run, u_lbl, g_run, g_lbl)
        a.set_ylabel(YLAB[metric])
        a.set_title(tag, loc="left")
        save(fig, f"result/q30ba3b/skip_gate/{slug}_{metric}")
