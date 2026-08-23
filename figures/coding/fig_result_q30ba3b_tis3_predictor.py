#!/usr/bin/env python
"""Ungated CISPO-3 / TIS-3 (`sjjc7dcpzf`, cap-3, no gate): what predicts the validation collapse?

Contrast with the no-clip case. Here the finite cap-3 keeps gradients BOUNDED (grad_norm maxes at
~34, never blows up), so grad_norm gives essentially no early warning. ESS is the only signal that
moves early:

(a) smoothed ESS vs validation. ESS falls through the 0.1 gate threshold right at the AIME peak
    (step 60) and craters to ~0.003 by step 80 — ~80 steps before terminal collapse (0.4 @150).
(b) grad_norm (log) vs validation. It stays < 1 through the peak and only exceeds 10 at ~step 125,
    i.e. well AFTER validation has already fallen. The cap masks the grad signal; ESS does not.

-> for_paper/figures_mains/result/q30ba3b/tis3_dynamics/ess_predicts_val.pdf
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
    h = w // 2
    return [sum(ys[max(0, i - h):min(len(ys), i + h + 1)]) /
            len(ys[max(0, i - h):min(len(ys), i + h + 1)]) for i in range(len(ys))]


use_paper_style()

RUN = "cispo3_nogate"
xe, ev = series(RUN, "eval")
xs, es = series(RUN, "ess")
xg, gn = series(RUN, "grad_norm")
ess_s = smooth(es, 7)

pk_step = xe[ev.index(max(ev))]                      # validation peak = step 60
onset = next((xs[i] for i in range(len(xs))
              if all(v < 0.1 for v in ess_s[i:i + 5])), None)
gn10 = next((s for s, v in zip(xg, gn) if v > 10), None)
XMAX = xe[-1] + 13

c_val, c_ess, c_gn = FAM[2], FAM[0], FAM[3]

fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.7), sharex=True)

# --- (a) ESS (smoothed) vs validation ---
a = ax[0]
a.plot(xs, es, color=c_ess, lw=0.6, alpha=0.35)
a.plot(xs, ess_s, color=c_ess, lw=1.7, label="ESS (smoothed)")
a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9)
a.set_ylabel("ESS (normalized)", color=c_ess)
a.tick_params(axis="y", labelcolor=c_ess)
a.set_ylim(0, 0.9)
if onset is not None:
    a.axvspan(onset, XMAX, color=c_ess, alpha=0.07)
    a.axvline(onset, color=c_ess, ls=":", lw=1.0)
    a.text(onset + 2, 0.2, f"ESS sub-0.1\nonset ~{onset}", color=c_ess, fontsize=7, va="bottom")
a.axvline(pk_step, color=c_val, ls="--", lw=1.0)
a.set_title("(a) ESS leads", loc="left")
a.set_xlabel("training step")
av = a.twinx()
av.plot(xe, [v * 100 for v in ev], color=c_val, lw=1.4, marker="o", ms=3)
av.set_ylabel("AIME mean@16 (%)", color=c_val)
av.tick_params(axis="y", labelcolor=c_val)
av.set_ylim(-1, 35)
av.text(pk_step - 3, 33, f"peak @ {pk_step}", color=c_val, fontsize=7, ha="right", va="top")

# --- (b) grad_norm (log) vs validation ---
b = ax[1]
b.plot(xg, [max(v, 1e-3) for v in gn], color=c_gn, lw=0.9, label="grad_norm")
b.set_yscale("log")
b.set_ylabel("grad_norm", color=c_gn)
b.tick_params(axis="y", labelcolor=c_gn)
b.axhline(10, color=c_gn, ls=(0, (1, 2)), lw=0.8)
if gn10 is not None:
    b.axvline(gn10, color=c_gn, ls=":", lw=1.0)
    b.text(gn10 + 2, 15, f"grad>10 only @{gn10}\n(after collapse)", color=c_gn, fontsize=7, va="bottom")
b.axvline(pk_step, color=c_val, ls="--", lw=1.0)
b.set_title("(b) grad_norm gives no early warning", loc="left")
b.set_xlabel("training step")
b.set_xlim(-3, XMAX)
bv = b.twinx()
bv.plot(xe, [v * 100 for v in ev], color=c_val, lw=1.4, marker="o", ms=3)
bv.set_ylabel("AIME mean@16 (%)", color=c_val)
bv.tick_params(axis="y", labelcolor=c_val)
bv.set_ylim(-1, 35)

save(fig, "result/q30ba3b/tis3_dynamics/ess_predicts_val")
print(f"val peak @ {pk_step}; ESS sub-0.1 onset @ {onset}; grad>10 @ {gn10}")
