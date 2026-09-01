#!/usr/bin/env python
"""ESS and gradient-norm timing for two permissive Qwen3-30B-A3B updates.

The columns compare an unclipped update (`bvrscfn6u8`) with ungated TIS 3
(`sjjc7dcpzf`). The top row pairs raw and trailing-average normalized ESS with
AIME-2024; the bottom row pairs gradient norm with the same evaluation curve.
All smoothing is trailing, so the diagnostic uses no future observations.

-> figures/result/q30ba3b/noclip/ess_predicts_val.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle
import matplotlib.pyplot as plt
from paperstyle import FULL, C, FAM, use_paper_style, save
from runlog import series

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures")


def smooth(ys, w=7):
    """Trailing moving average with window ``w``."""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w + 1), i + 1
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def sustained_onset(xs, ys, threshold=0.1, length=5):
    """Retrospective start of the first ``length``-point low-ESS episode."""
    return next(
        (xs[i] for i in range(len(xs) - length + 1)
         if all(v < threshold for v in ys[i:i + length])),
        None,
    )


def load(run):
    xe, ev = series(run, "eval")
    xs, es = series(run, "ess")
    xg, gn = series(run, "grad_norm")
    ess_trailing = smooth(es, 7)
    peak_step = xe[ev.index(max(ev))]
    onset = sustained_onset(xs, ess_trailing)
    return {
        "xe": xe, "ev": ev, "xs": xs, "es": es,
        "xg": xg, "gn": gn, "ess_trailing": ess_trailing,
        "peak_step": peak_step, "onset": onset,
    }


use_paper_style()

runs = [
    ("Unclipped update", load("noclip_ungated"), 203, (10, 50), None),
    ("TIS 3", load("cispo3_nogate"), 173, (-1, 35), 77),
]
c_val, c_ess, c_gn = FAM[2], FAM[0], FAM[3]

fig, ax = plt.subplots(2, 2, figsize=(FULL, 4.55), sharex="col")
letters = (("a", "b"), ("c", "d"))

for col, (name, d, xmax, val_ylim, ess_end) in enumerate(runs):
    # Top row: ESS and held-out performance.
    a = ax[0, col]
    end = len(d["xs"]) if ess_end is None else next(
        i for i, x in enumerate(d["xs"]) if x > ess_end
    )
    a.plot(d["xs"][:end], d["es"][:end], color=c_ess, lw=0.6, alpha=0.32,
           label="raw ESS")
    a.plot(d["xs"][:end], d["ess_trailing"][:end], color=c_ess, lw=1.7,
           label="trailing ESS")
    if ess_end is not None:
        post = max(0, end - 1)
        a.plot(d["xs"][post:], d["es"][post:], color=C["baseline"],
               lw=0.6, alpha=0.20)
        a.plot(d["xs"][post:], d["ess_trailing"][post:],
               color=C["baseline"], lw=1.4, alpha=0.65)
        a.text(145, 0.67, "post-collapse\noverlap", color=C["baseline"],
               fontsize=6.6, ha="center", va="bottom")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9)
    a.axvline(d["onset"], color=c_ess, ls=":", lw=1.0)
    a.axvline(d["peak_step"], color=c_val, ls="--", lw=1.0)
    a.text(d["onset"] + 3, 0.13,
           f"first sustained\nsub-0.1 episode: {d['onset']}",
           color=c_ess, fontsize=6.6, va="bottom")
    a.set_ylim(0, 0.9)
    a.set_xlim(-3, xmax)
    a.tick_params(axis="y", labelcolor=c_ess)
    a.set_title(f"({letters[0][col]}) {name}: ESS", loc="left")
    if col == 0:
        a.set_ylabel("normalized ESS", color=c_ess)
        a.legend(loc="upper right", frameon=False)

    av = a.twinx()
    av.plot(d["xe"], [100 * v for v in d["ev"]], color=c_val,
            lw=1.4, marker="o", ms=3)
    av.set_ylim(*val_ylim)
    av.tick_params(axis="y", labelcolor=c_val)
    av.grid(False)
    av.spines["right"].set_visible(True)
    av.spines["right"].set_color(c_val)
    if col == 1:
        av.set_ylabel("AIME mean@16 (%)", color=c_val)

    # Bottom row: gradient norm and the same held-out performance.
    b = ax[1, col]
    b.plot(d["xg"], [max(v, 1e-3) for v in d["gn"]], color=c_gn, lw=0.9)
    b.set_yscale("log")
    b.axvline(d["peak_step"], color=c_val, ls="--", lw=1.0)
    b.set_xlim(-3, xmax)
    b.tick_params(axis="y", labelcolor=c_gn)
    b.set_title(f"({letters[1][col]}) {name}: gradient norm", loc="left")
    b.set_xlabel("training step")
    if col == 0:
        b.set_ylabel("gradient norm", color=c_gn)

    if col == 0:
        b.axvline(57, color=c_gn, ls=":", lw=1.0)
        b.text(60, 2.2e4, "early spike", color=c_gn, fontsize=6.6, va="top")
        b.axvline(176, color=c_gn, ls=":", lw=1.0)
        b.text(173, 2.5e5, "post-peak max", color=c_gn,
               fontsize=6.6, ha="right", va="top")
    bv = b.twinx()
    bv.plot(d["xe"], [100 * v for v in d["ev"]], color=c_val,
            lw=1.4, marker="o", ms=3)
    bv.set_ylim(*val_ylim)
    bv.tick_params(axis="y", labelcolor=c_val)
    bv.grid(False)
    bv.spines["right"].set_visible(True)
    bv.spines["right"].set_color(c_val)
    if col == 1:
        bv.set_ylabel("AIME mean@16 (%)", color=c_val)

save(fig, "result/q30ba3b/noclip/ess_predicts_val")
for name, d, _xmax, _ylim, _ess_end in runs:
    print(f"{name}: peak {d['peak_step']}; sustained sub-0.1 onset {d['onset']}")
