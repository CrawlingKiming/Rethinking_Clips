#!/usr/bin/env python
"""Ungated no-clip (`bvrscfn6u8`): ESS vs reward on a shared step axis.

The point: training reward stays healthy the whole run (~0.63 final), giving no warning, while the
normalized sequence ESS collapses (crosses the 0.1 gate threshold at step ~57, then sits mostly
below it). Reward is blind to the off-policy blow-up; ESS is not. Same run whose grad_norm hits
5.9e5 and whose AIME fades 44.4 -> 34.6.

-> for_paper/figures_mains/result/q30ba3b/noclip/ess_vs_reward.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

use_paper_style()

xe, ess = series("noclip_ungated", "ess")
xr, rew = series("noclip_ungated", "reward")
c_ess, c_rew = FAM[0], FAM[2]

fig, ax1 = plt.subplots(figsize=(COL * 1.4, 2.6))
ax1.plot(xe, ess, color=c_ess, lw=1.4)
ax1.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9)
ax1.set_ylabel("ESS (normalized)", color=c_ess)
ax1.tick_params(axis="y", labelcolor=c_ess)
ax1.set_ylim(0, 0.66)
ax1.set_xlim(-3, 203)
ax1.set_xlabel("training step")

ax2 = ax1.twinx()
ax2.plot(xr, rew, color=c_rew, lw=1.4)
ax2.set_ylabel("training reward", color=c_rew)
ax2.tick_params(axis="y", labelcolor=c_rew)
ax2.set_ylim(0, 0.78)

save(fig, "result/q30ba3b/noclip/ess_vs_reward")
