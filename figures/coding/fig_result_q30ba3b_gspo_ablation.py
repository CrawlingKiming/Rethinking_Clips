#!/usr/bin/env python
"""GSPO ablation on Qwen3-30B (token-mean). Three GSPO runs — a no-gate baseline plus nominal ESS
thresholds 0.05/0.1/0.2. The ESS gate is effectively inert on GSPO (its 3e-4 sequence clip is ~600x
tighter than the gate's 0.2 band, so the gate rarely/never binds); spread is GSPO run-to-run variance.

Separate single-panel figures (one per metric):
  curve.pdf  = AIME-2024 mean@16
  ess.pdf    = normalized sequence ESS (gated runs only; the no-gate baseline logs no ESS)
  reward.pdf = training reward (critic/score/mean)

  GSPO no gate `q3cfydj8eu` gspo_base · +ESS0.05 `c332ayragg` gspo_ess005 ·
  +ESS0.1 `8m66pubxgu` gspo_ess01 · +ESS0.2 `782xyquesk` gspo_ess02  (cut=200 drops a stray eval)

-> for_paper/figures_mains/result/q30ba3b/gspo_ablation/{curve,ess,reward}.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import COL, C, FAM, use_paper_style, save
from runlog import series

paperstyle.FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_mains")

# (label, colour, marker, runlog key)
CURVES = [
    ("GSPO (no gate)",  FAM[2], "",  "gspo_base"),
    ("GSPO + ESS 0.05", FAM[0], "o", "gspo_ess005"),
    ("GSPO + ESS 0.1",  FAM[3], "o", "gspo_ess01"),
    ("GSPO + ESS 0.2",  FAM[1], "o", "gspo_ess02"),
]

use_paper_style()


def fig_eval():
    fig, a = plt.subplots(figsize=(COL * 1.25, 2.6))
    for lbl, col, mk, key in CURVES:
        xs, ys = series(key, "eval", cut=200)
        a.plot(xs, [v * 100 for v in ys], color=col, lw=1.7, marker=mk, ms=3.5,
               label=f"{lbl}:  pk {max(ys) * 100:.1f} / fin {ys[-1] * 100:.1f}")
    a.set_ylabel("AIME-2024 mean@16 (%)")
    a.set_title("GSPO ablation — AIME", loc="left")
    a.legend(loc="lower right")
    return fig, a


def fig_ess():
    fig, a = plt.subplots(figsize=(COL * 1.25, 2.6))
    for lbl, col, mk, key in CURVES:
        try:
            xs, ys = series(key, "ess", cut=200)      # no-gate baseline logs no ESS -> skipped
        except KeyError:
            continue
        frac = 100.0 * sum(v < 0.1 for v in ys) / len(ys)
        a.plot(xs, ys, color=col, lw=1.3, label=f"{lbl}: {frac:.0f}% < 0.1")
    a.axhline(0.1, color=C["baseline"], ls=(0, (1, 2)), lw=0.9, label="gate threshold 0.1")
    a.set_ylabel("ESS (normalized)")
    a.set_ylim(0, 0.66)
    a.set_title("GSPO ablation — sequence ESS", loc="left")
    a.legend(loc="upper right")
    return fig, a


def fig_reward():
    fig, a = plt.subplots(figsize=(COL * 1.25, 2.6))
    for lbl, col, mk, key in CURVES:
        xs, ys = series(key, "reward", cut=200)
        a.plot(xs, ys, color=col, lw=1.3, label=f"{lbl}: {ys[-1]:.3f}")
    a.set_ylabel("training reward")
    a.set_title("GSPO ablation — reward", loc="left")
    a.legend(loc="lower right")
    return fig, a


def fig_trip():
    fig, a = plt.subplots(figsize=(COL * 1.25, 2.6))
    for lbl, col, mk, key in CURVES:
        try:
            xs, ys = series(key, "trip", cut=200)      # no-gate baseline logs no gate/trip -> skipped
        except KeyError:
            continue
        a.plot(xs, ys, color=col, lw=1.3, label=f"{lbl}: max {max(ys):.3f}")
    a.set_ylabel("gate trip fraction")
    a.set_ylim(-0.03, 1.03)
    a.set_title("GSPO ablation — gate trip", loc="left")
    a.legend(loc="upper right")
    return fig, a


for maker, slug in [(fig_eval, "curve"), (fig_ess, "ess"), (fig_reward, "reward"), (fig_trip, "trip")]:
    fig, a = maker()
    a.set_xlabel("training step")
    a.set_xlim(-3, 205)
    save(fig, f"result/q30ba3b/gspo_ablation/{slug}")
print("done")
