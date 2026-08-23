#!/usr/bin/env python
"""cispo3 ESS-threshold ablation on Qwen3-30B (gpd_shape cap3, sum-norm, raw ESS, clip mode).
Only GATE_ESS varies: 0.05 / 0.1 / 0.2. Unlike the GSPO ablation, the gate is REAL here — cispo3's
dense coefficient loss has no tight built-in clip, so the ESS gate's 0.2 band actually binds.

  0.05  `k9ec6cfvkg`  -> cispo3_ess005
  0.1   `ircyhpdmku`  -> grpo_ess_clip (same config, existing)
  0.2   `mfw7j84534`  -> cispo3_ess02

Panels: (a) AIME-2024 mean@16, (b) normalized sequence ESS (with the 0.1 gate line, % steps < 0.1).

-> for_paper/figures_mains/result/q30ba3b/cispo3_ess_threshold/{overall,eval,ess}.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, FULL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (run key, label, colour)
RUNS = [
    ("cispo3_ess005", "ESS 0.05", FAM[0]),
    ("grpo_ess_clip", "ESS 0.1",  FAM[2]),
    ("cispo3_ess02",  "ESS 0.2",  FAM[1]),
]

use_paper_style()


def draw_eval(a):
    for run, lbl, col in RUNS:
        xs, ys = series(run, "eval", cut=200)
        a.plot(xs, [v * 100 for v in ys], color=col, lw=1.6, marker="o", ms=3,
               label=f"{lbl}: pk {max(ys) * 100:.1f} / fin {ys[-1] * 100:.1f}")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.legend(loc="lower right")


def draw_ess(a):
    for run, lbl, col in RUNS:
        xs, ys = series(run, "ess", cut=200)
        frac = 100.0 * sum(v < 0.1 for v in ys) / len(ys)
        a.plot(xs, ys, color=col, lw=1.3, label=f"{lbl}: {frac:.0f}% < 0.1")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")
    a.set_ylim(0, 0.66)
    a.legend(loc="upper right")


fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.6))
for a, (fn, tag) in zip(ax, [(draw_eval, "(a) AIME-2024"), (draw_ess, "(b) sequence ESS")]):
    fn(a)
    a.set_title(tag, loc="left")
    a.set_xlim(-3, 205)
    a.set_xlabel("training step")
save(fig, "result/q30ba3b/cispo3_ess_threshold/overall")

for fn, slug in [(draw_eval, "eval"), (draw_ess, "ess")]:
    fig, a = plt.subplots(figsize=(COL, 2.5))
    fn(a)
    a.set_xlim(-3, 205)
    a.set_xlabel("training step")
    save(fig, f"result/q30ba3b/cispo3_ess_threshold/{slug}")
print("cispo3 ess-threshold figures done")
