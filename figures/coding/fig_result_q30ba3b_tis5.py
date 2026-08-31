#!/usr/bin/env python
"""Performance-only TIS-5 collapse-prevention comparison.

    no gate    `ayv2ajeuqk`  peak 28.3% @60, then 0.0% from step 130 on
    + ESS gate `rdq6r5yy83`  39.4% @200 (ESS skip at threshold 0.1)

The pair differs only in whether a low-ESS update is skipped. Both curves are
in-training validation trajectories. The figure intentionally omits ESS,
which has already been established as the diagnostic in Section 5.

-> for_paper/figures/result/q30ba3b/tis5/overall.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import FULL, C, format_sig, use_paper_style, save
from runlog import series

XMAX = 203
RUNS_ = [("cispo5_nogate", "TIS 5, no gate", C["baseline2"], "--"),
         ("cispo5_ess",    r"TIS 5 + ESS $\rightarrow$ skip", C["gated"], "-")]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

use_paper_style()
fig, ax = plt.subplots(figsize=(0.72 * FULL, 2.35))

for index, (run, lbl, col, ls) in enumerate(RUNS_):
    xs, ys = series(run, "eval")
    values = [value * 100 for value in ys]
    ax.plot(
        xs,
        values,
        color=col,
        linestyle=ls,
        marker="o",
        linewidth=1.6 if ls == "-" else 1.2,
        label=lbl,
    )
    ax.annotate(
        f"{format_sig(values[-1])}%",
        xy=(xs[-1], values[-1]),
        xytext=(5, 7 if index else 6),
        textcoords="offset points",
        color=col,
        fontsize=6.5,
        ha="left",
        va="center",
        clip_on=False,
    )

ax.set_ylabel("AIME-2024 mean@16 (%)")
ax.set_ylim(-1.5, 46)
ax.set_xlim(-3, XMAX)
ax.set_xlabel("training step")
ax.set_title("Qwen3-30B-A3B, TIS 5", loc="left")
fig.legend(loc="outside lower center", ncol=2, frameon=False)

save(fig, "result/q30ba3b/tis5/overall")
