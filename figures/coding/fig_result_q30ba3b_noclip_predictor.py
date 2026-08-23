#!/usr/bin/env python
"""Ungated no-clip (`bvrscfn6u8`): what predicts the validation fade — ESS or grad_norm?

(a) smoothed ESS vs validation. Smoothed ESS falls through the 0.1 gate threshold ~50 steps
    BEFORE the AIME peak (step 140) and stays there — the sustained ESS collapse is the leading
    indicator of the fade to 34.6.
(b) grad_norm (log) vs validation. Its spikes are episodic and coincident: the step-57 spike is a
    false alarm (val kept rising), and the biggest spike (5.9e5) fires at ~180, i.e. AS the final
    crash happens, not before it. grad_norm is a symptom, not an early predictor.

-> for_paper/figures_mains/result/q30ba3b/noclip/ess_predicts_val.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")


def smooth(ys, w=7):
    """centered moving average, window w (odd)."""
    h = w // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - h), min(len(ys), i + h + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


use_paper_style()

xe, ev = series("noclip_ungated", "eval")
xs, es = series("noclip_ungated", "ess")
xg, gn = series("noclip_ungated", "grad_norm")
ess_s = smooth(es, 7)

pk_step = xe[ev.index(max(ev))]                      # validation peak = step 140
# ESS "onset": first step after which the SMOOTHED ESS stays below the 0.1 gate threshold
onset = next((xs[i] for i in range(len(xs))
              if all(v < 0.1 for v in ess_s[i:i + 5])), None)

c_val, c_ess, c_gn = FAM[2], FAM[0], FAM[3]

fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.7), sharex=True)

# --- (a) ESS (smoothed) vs validation ---
a = ax[0]
a.plot(xs, es, color=c_ess, lw=0.6, alpha=0.35)
a.plot(xs, ess_s, color=c_ess, lw=1.7, label="ESS (smoothed)")
a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9)
a.set_ylabel("ESS (normalized)", color=c_ess)
a.tick_params(axis="y", labelcolor=c_ess)
a.set_ylim(0, 0.66)
if onset is not None:
    a.axvspan(onset, 203, color=c_ess, alpha=0.07)
    a.axvline(onset, color=c_ess, ls=":", lw=1.0)
    a.text(onset + 2, 0.6, f"ESS sub-0.1\nonset ~{onset}", color=c_ess, fontsize=7, va="top")
a.axvline(pk_step, color=c_val, ls="--", lw=1.0)
a.set_title("(a) ESS leads", loc="left")
a.set_xlabel("training step")
av = a.twinx()
av.plot(xe, [v * 100 for v in ev], color=c_val, lw=1.4, marker="o", ms=3)
av.set_ylabel("AIME mean@16 (%)", color=c_val)
av.tick_params(axis="y", labelcolor=c_val)
av.set_ylim(10, 50)
av.text(pk_step - 3, 46, f"peak @ {pk_step}", color=c_val, fontsize=7, ha="right")

# --- (b) grad_norm (log) vs validation ---
b = ax[1]
b.plot(xg, [max(v, 1e-3) for v in gn], color=c_gn, lw=0.9, label="grad_norm")
b.set_yscale("log")
b.set_ylabel("grad_norm", color=c_gn)
b.tick_params(axis="y", labelcolor=c_gn)
b.axvline(57, color=c_gn, ls=":", lw=1.0)
b.text(59, 3e4, "false alarm\n@57", color=c_gn, fontsize=7, va="top")
b.axvline(pk_step, color=c_val, ls="--", lw=1.0)
b.set_title("(b) grad_norm coincides", loc="left")
b.set_xlabel("training step")
b.set_xlim(-3, 203)
bv = b.twinx()
bv.plot(xe, [v * 100 for v in ev], color=c_val, lw=1.4, marker="o", ms=3)
bv.set_ylabel("AIME mean@16 (%)", color=c_val)
bv.tick_params(axis="y", labelcolor=c_val)
bv.set_ylim(10, 50)

save(fig, "result/q30ba3b/noclip/ess_predicts_val")
print(f"val peak @ {pk_step}; ESS sub-0.1 onset @ {onset}")
