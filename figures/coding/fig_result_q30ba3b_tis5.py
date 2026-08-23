#!/usr/bin/env python
"""TIS 5 (Qwen3-30B-A3B), kept separate from curves.pdf.

    no gate    `ayv2ajeuqk`  peak 28.3% @60, then 0.0% from step 130 on
    + ESS gate `rdq6r5yy83`  39.4% @200 (ESS skip at threshold 0.1)

This is the one row in results_sweep_A_B1.md where *only* the gate toggles, so it is the cleanest
single-variable evidence in the suite. It is plotted on its own because the gated partner was not in
the first held-out sweep: its held-out number is still pending, while the no-gate run has a held-out
AIME-2024 of 0.0. Both curves below are the in-training validation, so they are comparable to each
other but not to the held-out table.

  (a) eval, and (b) the ESS that explains it. ESS in (b) is truncated by runlog.ess_cut().

-> for_paper/figures/result/q30ba3b/tis5/overall.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from paperstyle import FULL, C, use_paper_style, save
from runlog import series, ess_cut

FLOOR = 0.01
XMAX = 203
RUNS_ = [("cispo5_nogate", "TIS 5, no gate", C["baseline2"], "--"),
         ("cispo5_ess",    "TIS 5 + ESS gate", C["gated"], "-")]

use_paper_style()
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.3))

for run, lbl, col, ls in RUNS_:
    xs, ys = series(run, "eval")
    ax[0].plot(xs, [v * 100 for v in ys], color=col, ls=ls, marker="o",
               lw=1.6 if ls == "-" else 1.2, label=f"{lbl}: {ys[-1] * 100:.1f}%")
    cut = ess_cut(run, FLOOR)
    xs, ys = series(run, "ess", cut=cut)
    ax[1].plot(xs, ys, color=col, ls=ls, lw=1.1, label=lbl)
    if cut is not None:
        ax[1].axvspan(cut, XMAX, color=C["baseline3"], alpha=0.35, lw=0)
        ax[1].annotate(f"ESS $<$ {FLOOR:g} after step {cut}", xy=(cut + 4, 0.655),
                       fontsize=6.5, color=C["baseline"], va="top")

ax[0].set_ylabel("AIME-2024 mean@16 (%)")
ax[0].set_ylim(-1.5, 46)
ax[0].set_title("(a) eval", loc="left")
ax[0].legend(loc="center left")

ax[1].axhline(0.1, color=C["ours"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
ax[1].set_ylabel("ESS (normalized)")
ax[1].set_ylim(0, 0.68)
ax[1].set_title("(b) ESS", loc="left")
ax[1].legend(loc="upper right")

for a in ax:
    a.set_xlim(-3, XMAX)
    a.set_xlabel("training step")

save(fig, "result/q30ba3b/tis5/overall")
