#!/usr/bin/env python
"""No-clip PG (cap inf) on Qwen3-30B-A3B: does removing clipping alone destabilize training?

    ungated  `bvrscfn6u8`  pk 44.4 / fin 34.6,  grad_norm max 5.9e5
    + ESS clip-latch     `q2m6j822id`  pk 43.3 / fin 38.1,  grad_norm max 1.7e5
    + ESS skip           `vm7vcynvy7`  pk 35.6 / fin 35.6,  grad_norm max 26.8

`bvrscfn6u8` was launched with a frac.skip.0.015 gate, but its upper-ratio fraction peaks at 1.20%
and never crosses the 1.5% trigger: `actor/gate/skipped` sums to 0.62 across the whole run, so no
step was meaningfully modified and it stands as the ungated datapoint. It is labelled that way
rather than "no gate", because a gate was nominally configured.

This is the "no-clip alone" evidence. Unclipped reaches the highest peak of the family (44.4%), so
removing the clip is not what caps performance, but grad_norm runs to 5.9e5 and the run fades from
44.4 to 34.6 with nothing to catch it. Both gates keep grad_norm orders of magnitude lower; the
clip-latch keeps most of the peak (43.3) while the skip gate trades peak for flatness.

grad_norm is a diagnostic, not the headline, so it is not in the overall figure; it has its own
plot. It is on a log axis: it spans 26.8 to 5.9e5 here, which no linear axis can show.

-> for_paper/figures_mains/result/q30ba3b/noclip/overall.pdf  (A: eval + ESS)
-> for_paper/figures_mains/result/q30ba3b/noclip/{eval,ess,grad_norm}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series

# route into figures_mains/ (the paper's main figures), matching the 8B curves script
paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run, label, colour, linestyle, linewidth)
VARIANTS = [
    ("noclip_ungated",  "ungated", FAM[3], "--", 1.3),
    ("noclip_ess_clip", "+ ESS clip-latch",    FAM[0], "-",  1.5),
    ("noclip_ess_skip", "+ ESS skip",          FAM[2], "-",  1.5),
]

def draw_eval(a):
    for run, lbl, col, ls, lw in VARIANTS:
        xs, ys = series(run, "eval")
        a.plot(xs, [v * 100 for v in ys], color=col, ls=ls, lw=lw, marker="o",
               label=f"{lbl}: pk {max(ys) * 100:.1f} / fin {ys[-1] * 100:.1f}")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_ylim(10, 50)
    a.legend(loc="lower left")


def draw_ess(a):
    """Normalized sequence ESS per step — the headline: without a gate ESS is free to
    crater (ungated), while both gates keep it off the floor. 0.1 = gate threshold."""
    for run, lbl, col, ls, lw in VARIANTS:
        xs, ys = series(run, "ess")
        frac = 100.0 * sum(v < 0.1 for v in ys) / len(ys)
        a.plot(xs, ys, color=col, ls=ls, lw=lw, label=f"{lbl}: {frac:.0f}% steps < 0.1")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")
    a.set_ylim(0, 0.66)
    a.legend(loc="upper right")


def draw_grad_norm(a):
    for run, lbl, col, ls, lw in VARIANTS:
        xs, ys = series(run, "grad_norm")
        a.plot(xs, ys, color=col, ls=ls, lw=1.0, label=f"{lbl}: max {max(ys):.2g}")
    a.set_ylabel("grad_norm")
    a.set_yscale("log")
    a.legend(loc="lower right")


use_paper_style()

# --- A) overall: eval result + ESS (the two headline panels). grad_norm is a log-axis
#     diagnostic and gets its own plot. ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.5))
for a, (fn, tag) in zip(ax, [(draw_eval, "(a) AIME-2024"), (draw_ess, "(b) sequence ESS")]):
    fn(a)
    a.set_title(tag, loc="left")
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
save(fig, "result/q30ba3b/noclip/overall")

# --- B) separates ---
for fn, slug in [(draw_eval, "eval"), (draw_ess, "ess"), (draw_grad_norm, "grad_norm")]:
    fig, a = plt.subplots(figsize=(COL, 2.5))
    fn(a)
    a.set_xlim(-3, 203)
    a.set_xlabel("training step")
    save(fig, f"result/q30ba3b/noclip/{slug}")
