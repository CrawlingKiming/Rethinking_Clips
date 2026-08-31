#!/usr/bin/env python
"""Motivation (Qwen3-30B-A3B): without a gate, ESS collapses and the eval collapses after it.

Two truncated-importance-sampling caps, both with no gate:

    TIS 3, no gate      task sjjc7dcpzf   peak 31.7% @60, then down to 0.4% @150
    TIS 5, no gate      task ayv2ajeuqk   peak 28.3% @60, then down to 0.0% @130

In both, ESS falls through the 0.1 threshold while the eval is still near its peak, so the ESS drop
leads the collapse rather than trailing it.

The ESS trace is truncated by runlog.ess_cut(), the same test in both panels: stop at the first step
where ESS < 0.01 and the eval never afterwards recovers. Beyond that step the measured ESS rises
again while the eval stays at zero, which is not informative about the update and reads as a
contradiction if drawn. Cut steps are annotated on the panels and belong in the caption.

Emits:
  A) one standalone single-column figure per panel
  B) one combined full-width figure with both

-> for_paper/figures/motivation/overall.pdf        (A, both panels)
-> for_paper/figures/motivation/tis{3,5}_nogate.pdf  (B, one per panel)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, use_paper_style, save
from runlog import series, ess_cut

FLOOR = 0.01
ESS_REFERENCE = 0.01
XMAX = 203
# (run, panel title, output slug)
PANELS = [
    ("cispo3_nogate", "TIS 3, no gate", "tis3_nogate"),
    ("cispo5_nogate", "TIS 5, no gate", "tis5_nogate"),
]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")


def draw(a, run, show_right_label=True):
    """Eval on the left axis, ESS on the right. Returns the twin axis."""
    xs, ys = series(run, "eval")
    a.plot(xs, [v * 100 for v in ys], color=C["eval"], ls="-", marker="o",
           label="AIME-2024 mean@16 (left)")
    a.set_ylim(-1.5, 40)
    a.set_xlim(-3, XMAX)
    a.set_xlabel("training step")

    cut = ess_cut(run, FLOOR)
    ex_, ey = series(run, "ess", cut=cut)
    a2 = a.twinx()
    a2.plot(ex_, ey, color=C["ours"], lw=1.1, label="ESS (right)")
    a2.axhline(
        ESS_REFERENCE,
        color=C["ours"],
        ls=(0, (1, 2)),
        lw=0.9,
        label="ESS reference 0.01",
    )
    a2.set_ylim(0, 0.68)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    a2.spines["right"].set_color(C["ours"])
    a2.tick_params(axis="y", colors=C["ours"], labelright=show_right_label)
    if show_right_label:
        a2.set_ylabel("ESS (normalized)", color=C["ours"])
    if cut is not None:
        a2.axvline(cut, color=C["ours"], ls=":", lw=0.9)
        a2.axvspan(cut, XMAX, color=C["ours"], alpha=0.06, lw=0)
        a2.annotate(f"ESS $<$ {FLOOR:g} after step {cut}", xy=(cut + 4, 0.665),
                    fontsize=6.5, color=C["ours"], va="top")
    return a2


use_paper_style()

# --- A) one standalone single-column figure per panel ---
for run, tag, slug in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.35))
    a2 = draw(a, run)
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_title(tag, loc="left")
    h, l = a.get_legend_handles_labels()
    for hh, ll in zip(*a2.get_legend_handles_labels()):
        h.append(hh), l.append(ll)
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"motivation/{slug}")

# --- B) both panels combined ---
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.35), sharey=True)
for i, (run, tag, _slug) in enumerate(PANELS):
    a = ax[i]
    a2 = draw(a, run, show_right_label=(i == len(PANELS) - 1))
    a.set_title(f"({'ab'[i]}) {tag}", loc="left")
    if i == 0:
        a.set_ylabel("AIME-2024 mean@16 (%)")
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(*a2.get_legend_handles_labels()):
            h.append(hh), l.append(ll)
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
save(fig, "motivation/overall")
