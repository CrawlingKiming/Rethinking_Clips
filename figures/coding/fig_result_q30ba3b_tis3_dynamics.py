#!/usr/bin/env python
"""TIS 3 dynamics (Qwen3-30B-A3B): what the collapse looks like, and what the gate prevents.

    no gate    `sjjc7dcpzf`  peak 31.7% @60, then 0.4% @170
    + ESS gate `328rfu6eb2`  44.8% @200 (ESS -> clip-higher band latch, dual-clip)

Four panels, both runs in each:
  (a) mean response length   the ungated run runs away to 8053 tokens
  (b) policy entropy         LINEAR axis; the ungated run spikes to 4.9 before crashing to 0.007
  (c) ESS                    truncated by runlog.ess_cut(), which fires only for the ungated run
  (d) gate action            fraction of updates clipped, corrected by runlog.gate_fraction();
                             flat zero without a gate, by construction

Panel (b) is the reason to look at this run: entropy blowing up and then collapsing, together with
the length runaway in (a), is the signature the gate removes. On a linear axis the gated run is
compressed near zero because the ungated excursion is two orders of magnitude larger.

Emits both forms:
  A) one standalone single-column figure per panel
  B) one combined 2x2 figure

-> for_paper/figures/result/q30ba3b/tis3_dynamics/overall.pdf                 (A)
-> for_paper/figures/result/q30ba3b/tis3_dynamics/{length,entropy,ess,gate}.pdf  (B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, use_paper_style, save
from runlog import series, ess_cut, gate_fraction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paperstyle.FIGDIR = os.path.join(ROOT, "figures_mains")

FLOOR = 0.01
XMAX = 203
RUNS_ = [("cispo3_nogate", "TIS 3, no gate", C["baseline2"], "--", 1.2),
         ("dapo_ess",      "TIS 3 + ESS gate", C["gated"], "-", 1.5)]
# (metric, ylabel, title, slug, ylim, truncate_ess)
PANELS = [
    ("length",  "mean response length (tokens)", "(a) response length", "length",  (700, 8400), False),
    ("entropy", "policy entropy",                "(b) entropy",         "entropy", (0, 5.1),    False),
    ("ess",     "ESS",                           "(c) ESS",             "ess",     (0, 0.68),   True),
    ("clipped", "fraction of updates clipped",   "(d) gate action",     "gate",    (-0.03, 1.03), False),
]


def draw(a, metric, ylab, ylim, truncate):
    for run, lbl, col, ls, lw in RUNS_:
        cut = ess_cut(run, FLOOR) if truncate else None
        if metric == "clipped":
            xs, ys = gate_fraction(run, "clipped")
        else:
            xs, ys = series(run, metric, cut=cut)
        a.plot(xs, ys, color=col, ls=ls, lw=lw, label=lbl)
        if truncate and cut is not None:
            a.axvspan(cut, XMAX, color=C["baseline3"], alpha=0.35, lw=0)
            a.annotate(f"ESS $<$ {FLOOR:g} after step {cut}", xy=(cut + 4, ylim[1] * 0.96),
                       fontsize=6.5, color=C["baseline"], va="top")
    if metric == "ess":
        a.axhline(0.1, color=C["ours"], ls=(0, (1, 2)), lw=0.9)
    a.set_ylabel(ylab)
    a.set_xlabel("training step")
    a.set_xlim(-3, XMAX)
    a.set_ylim(*ylim)


use_paper_style()

# --- A) standalone single-column figures ---
for metric, ylab, tag, slug, ylim, trunc in PANELS:
    fig, a = plt.subplots(figsize=(COL, 2.35))
    draw(a, metric, ylab, ylim, trunc)
    a.set_title(tag, loc="left")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    save(fig, f"result/q30ba3b/tis3_dynamics/{slug}")

# --- B) combined 2x2 ---
fig, ax = plt.subplots(2, 2, figsize=(FULL, 4.1), sharex=True)
flat = ax.ravel()
for i, (metric, ylab, tag, _slug, ylim, trunc) in enumerate(PANELS):
    a = flat[i]
    draw(a, metric, ylab, ylim, trunc)
    a.set_title(tag, loc="left")
    if i < 2:
        a.set_xlabel("")
    if i == 0:
        h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
save(fig, "result/q30ba3b/tis3_dynamics/overall")
