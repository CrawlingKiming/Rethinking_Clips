#!/usr/bin/env python
"""ESS timing for two permissive Qwen3-30B-A3B updates.

The panels compare an unclipped update (`bvrscfn6u8`) with ungated TIS 3
(`sjjc7dcpzf`). Each panel aligns raw and trailing-average normalized ESS with
AIME-2024. All smoothing is trailing, so the diagnostic uses no future
observations.

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
    ess_trailing = smooth(es, 7)
    peak_step = xe[ev.index(max(ev))]
    onset = sustained_onset(xs, ess_trailing)
    return {
        "xe": xe, "ev": ev, "xs": xs, "es": es,
        "ess_trailing": ess_trailing, "peak_step": peak_step,
        "onset": onset,
    }


use_paper_style()
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.7,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.7,
    "lines.linewidth": 1.55,
})

runs = [
    ("Unclipped update", load("noclip_ungated"), 203, (10, 50), None),
    ("TIS 3", load("cispo3_nogate"), 173, (-1, 35), 77),
]
c_val, c_ess = FAM[2], FAM[0]

fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35))

for col, (name, d, xmax, val_ylim, ess_end) in enumerate(runs):
    axis = axes[col]
    end = len(d["xs"]) if ess_end is None else next(
        i for i, x in enumerate(d["xs"]) if x > ess_end
    )
    axis.plot(
        d["xs"][:end],
        d["es"][:end],
        color=c_ess,
        lw=0.75,
        alpha=0.28,
        label="raw ESS",
    )
    axis.plot(
        d["xs"][:end],
        d["ess_trailing"][:end],
        color=c_ess,
        lw=2.0,
        label="7-step ESS",
    )
    if ess_end is not None:
        post = max(0, end - 1)
        axis.plot(
            d["xs"][post:],
            d["es"][post:],
            color=C["baseline"],
            lw=0.75,
            alpha=0.18,
        )
        axis.plot(
            d["xs"][post:],
            d["ess_trailing"][post:],
            color=C["baseline"],
            lw=1.75,
            alpha=0.68,
        )
        axis.text(
            145,
            0.67,
            "post-collapse\noverlap",
            color=C["baseline"],
            fontsize=6.5,
            ha="center",
            va="bottom",
        )
    axis.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=1.0)
    axis.axvline(d["onset"], color=c_ess, ls=":", lw=1.2)
    axis.axvline(d["peak_step"], color=c_val, ls="--", lw=1.2)
    axis.text(
        d["onset"] + 3,
        0.13,
        f"ESS < 0.1\nstep {d['onset']}",
        color=c_ess,
        fontsize=6.6,
        va="bottom",
    )
    axis.set_ylim(0, 0.9)
    axis.set_xlim(-3, xmax)
    axis.set_xlabel("training step")
    axis.tick_params(axis="y", labelcolor=c_ess)
    axis.set_title(f"({'ab'[col]}) {name}", loc="left", fontweight="bold")
    if col == 0:
        axis.set_ylabel("normalized ESS", color=c_ess)
        axis.legend(loc="upper right", frameon=False)

    validation_axis = axis.twinx()
    validation_axis.plot(
        d["xe"],
        [100 * value for value in d["ev"]],
        color=c_val,
        lw=1.7,
        marker="o",
        ms=3.2,
    )
    validation_axis.set_ylim(*val_ylim)
    validation_axis.tick_params(axis="y", labelcolor=c_val)
    validation_axis.grid(False)
    validation_axis.spines["right"].set_visible(True)
    validation_axis.spines["right"].set_color(c_val)
    if col == 1:
        validation_axis.set_ylabel("AIME mean@16 (%)", color=c_val)

save(fig, "result/q30ba3b/noclip/ess_predicts_val")
for name, d, _xmax, _ylim, _ess_end in runs:
    print(f"{name}: peak {d['peak_step']}; sustained sub-0.1 onset {d['onset']}")
