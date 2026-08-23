#!/usr/bin/env python
"""What governs a performance update in the UNGATED runs? Scope: the three no-gate runs that vary
only the ratio cap — pure TIS cap-3 (`sjjc7dcpzf`), pure TIS cap-5 (`ayv2ajeuqk`), GRPO no-clip
cap-inf (`bvrscfn6u8`). A run is considered collapsed once validation falls below its initial
(step-0) value; all windows from that step on are dropped (cispo3 @100, cispo5 @110, no-clip never).
Pooled over the surviving eval windows, the per-window change in validation tracks the window's mean
sequence ESS far more than grad_norm:

-> for_paper/figures_mains/result/q30ba3b/ungated_governs/ess_governs_update.pdf
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import FULL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (label, runlog key, colour) — the three UNGATED runs, cap 3 / 5 / inf
RUNS = [
    ("pure TIS, cap 3",     "cispo3_nogate", FAM[3]),
    ("pure TIS, cap 5",     "cispo5_nogate", FAM[1]),
    ("GRPO no-clip, cap ∞", "noclip_ungated", FAM[0]),
]


def windows(key):
    """per-eval-window (Δ AIME %, mean ESS, max grad_norm, mean frac_upper), post-collapse dropped.

    Collapse = validation falls below its initial (step-0) value; everything from that step on is
    discarded (the run has regressed past where it started, so its ESS trace is not meaningful)."""
    xe, ev = series(key, "eval")
    esd = dict(zip(*series(key, "ess")))
    gnd = dict(zip(*series(key, "grad_norm")))
    frd = dict(zip(*series(key, "frac_upper")))
    base = ev[0]
    collapse_start = next((xe[i] for i in range(1, len(xe)) if ev[i] < base), None)
    out = []
    for i in range(len(xe) - 1):
        if collapse_start is not None and xe[i] >= collapse_start:
            continue
        s0, s1 = xe[i], xe[i + 1]
        we = [esd[s] for s in esd if s0 < s <= s1]
        wg = [gnd[s] for s in gnd if s0 < s <= s1]
        wf = [frd[s] for s in frd if s0 < s <= s1]
        if we and wg and wf:
            out.append(((ev[i + 1] - ev[i]) * 100.0,
                        sum(we) / len(we), max(wg), sum(wf) / len(wf)))
    return out


def pear(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x) ** .5
    vy = sum((b - my) ** 2 for b in y) ** .5
    return cov / (vx * vy)


# gather
data = {lbl: windows(key) for lbl, key, _ in RUNS}
D  = [w[0] for lbl in data for w in data[lbl]]
E  = [w[1] for lbl in data for w in data[lbl]]
G  = [w[2] for lbl in data for w in data[lbl]]
Fu = [w[3] for lbl in data for w in data[lbl]]

use_paper_style()
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.7), sharey=True)

r_ess = pear(E, D)
r_gn  = pear([math.log10(max(g, 1e-3)) for g in G], D)

def panel(a, xi, xlab, logx, thresh, title, r):
    for lbl, key, col in RUNS:
        ws = data[lbl]
        a.scatter([w[xi] for w in ws], [w[0] for w in ws],
                  s=15, color=col, alpha=0.85, edgecolors="none", label=lbl)
    a.axhline(0, color=C["baseline"], lw=0.8)
    if logx:
        a.set_xscale("log")
    if thresh is not None:
        a.axvline(thresh, color=C["baseline"], ls=(0, (1, 2)), lw=0.9)
    a.set_xlabel(xlab)
    a.set_title(f"{title}  (r={r:+.2f})", loc="left", fontsize=8)

panel(ax[0], 1, "window-mean ESS",      False, 0.1, "(a) ESS", r_ess)
panel(ax[1], 2, "window-max grad_norm", True,  None, "(b) grad_norm", r_gn)
ax[0].set_ylabel(r"$\Delta$ AIME mean@16 (pts)")
ax[0].legend(loc="lower right", fontsize=6.5, handletextpad=0.2)

save(fig, "result/q30ba3b/ungated_governs/ess_governs_update")
print(f"n={len(D)}  r(ESS)={r_ess:+.3f}  r(log grad_norm)={r_gn:+.3f}")
