#!/usr/bin/env python3
"""Compact alternatives to manuscript Figures 3, 4, 5/6, and 7.

Run from any directory:
    python3 figures/coding/fig_compact_main.py

Only new *_compact.pdf files are written. Original PDFs, plotting scripts,
and observations are preserved. Each canvas is 6.5 inches wide, matching
the current manuscript's text width. All training curves use the tracked
CSVs, without smoothing, resampling, or additional experiments.
"""

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from paperstyle import C, format_sig, use_paper_style
from runlog import series

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))
import gaussian_quadratic_one_step as gaussian

WIDTH = 6.5
STATIC = "#4d4d4d"
CONDITIONAL = C["ours"]


def compact_style():
    use_paper_style()
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.labelpad": 2.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "legend.fontsize": 7.0,
        "legend.labelspacing": 0.2,
        "legend.handlelength": 1.6,
        "figure.constrained_layout.h_pad": 0.015,
        "figure.constrained_layout.w_pad": 0.015,
        "figure.constrained_layout.wspace": 0.035,
        "lines.linewidth": 1.4,
        "lines.markersize": 2.7,
    })


def save_compact(fig, relative_path, preview_dir=None):
    path = ROOT / relative_path
    if not path.stem.endswith("_compact"):
        raise ValueError("Only new compact variants may be written")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pdf")
    fig.savefig(temporary)
    temporary.replace(path)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / (path.stem + ".png"), dpi=200)
    plt.close(fig)
    print(path.relative_to(ROOT))


def gaussian_comparison(preview_dir=None):
    """Same stored Monte Carlo results, exact gains, and crossover markers."""
    with (ROOT / "simulation/results/gaussian_quadratic_one_step.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["rho"]))
    summary = json.loads((ROOT / "simulation/results/gaussian_quadratic_one_step_summary.json").read_text())["crossovers"]
    values = lambda key: np.array([float(row[key]) for row in rows])
    rho = values("rho")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.70))
    ax, gap, gain = axes
    actual = values("mc_actual_improvement_fraction")
    positive = values("mc_positive_certificate_fraction")
    for data, color, marker, linestyle, label, markevery in [
        (actual, gaussian.IS_COLOR, "o", "-", "Actual improvement", (0, 9)),
        (positive, gaussian.PPO_COLOR, "D", (0, (5, 2.2)), "Positive certificate", (4, 9)),
    ]:
        ax.plot(rho, data, color=color, marker=marker, linestyle=linestyle,
                markevery=markevery, markerfacecolor="white", markeredgewidth=0.9,
                label=label)
    ax.fill_between(rho, actual, positive, where=positive >= actual,
                    color=gaussian.NEUTRAL_COLOR, alpha=0.1, linewidth=0)
    ax.set(xscale="log", xlim=(gaussian.MINIMUM_RHO, gaussian.MAXIMUM_RHO),
           ylim=(0, 0.67), yticks=[0, 0.2, 0.4, 0.6], ylabel="Fraction of batches")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=6.6, handlelength=1.3,
              handletextpad=0.35)

    gap.axhline(0, color=gaussian.NEUTRAL_COLOR, linewidth=0.9)
    for key, color, marker, linestyle, label, markevery in [
        ("tis_certificate_gap_vs_is", gaussian.TIS_COLOR, "D", "-", "TIS 3 vs. IS", (0, 9)),
        ("ppo_certificate_gap_vs_is", gaussian.PPO_COLOR, "s", (0, (5, 2.2)), "PPO vs. IS", (4, 9)),
    ]:
        gap.plot(rho, values(key), color=color, marker=marker, linestyle=linestyle,
                 markevery=markevery, markerfacecolor="white", markeredgewidth=0.9,
                 label=label)
    gap.set_xscale("log")
    gap.set_yscale("symlog", linthresh=0.1, linscale=1.45)
    gap.set(xlim=(gaussian.MINIMUM_RHO, gaussian.MAXIMUM_RHO),
            yticks=[-1, -0.1, 0, 0.1, 1, 10], ylabel="Gain relative to IS")
    gap.legend(loc="lower left", frameon=False, fontsize=6.5)

    gain.axhline(0, color=gaussian.NEUTRAL_COLOR, linewidth=0.9)
    for key, color, marker, linestyle, label, markevery in [
        ("is_expected_gain", gaussian.IS_COLOR, "o", "-", "IS", (0, 8)),
        ("tis_expected_gain", gaussian.TIS_COLOR, "D", "-", "TIS 3", (3, 8)),
        ("ppo_expected_gain", gaussian.PPO_COLOR, "s", (0, (5, 2.2)), "PPO", (6, 8)),
    ]:
        gain.plot(rho, values(key), color=color, marker=marker, linestyle=linestyle,
                  markevery=markevery, markerfacecolor="white", markeredgewidth=0.9,
                  label=label)
    for rho_key, delta_key, color, label, offset in [
        ("tis_certificate_crossover_rho", "tis_certificate_crossover_delta", gaussian.TIS_COLOR, "TIS 3 = IS", 3),
        ("certificate_crossover_rho", "certificate_crossover_delta", gaussian.PPO_COLOR, "PPO = IS", -3),
    ]:
        root = summary[rho_key]
        root_gain = gaussian.estimator_moments(summary[delta_key])["is_expected_gain"]
        for axis in (gap, gain):
            axis.axvline(root, color=color, linestyle=(0, (1.2, 2)), linewidth=1.1, alpha=0.9)
        gap.annotate(f"{label}\n{format_sig(root)}", xy=(root, 0.98),
                     xycoords=("data", "axes fraction"), xytext=(offset, 0),
                     textcoords="offset points", ha="left" if offset > 0 else "right",
                     va="top", rotation=90, fontsize=6.2, color=color,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.6))
        gain.scatter([root], [root_gain], s=32, facecolor="white", edgecolor=color,
                     linewidth=1.3, zorder=6)
    gain.set(xlim=(0.025, 0.16), ylim=(-0.35, 1.12),
             xticks=[0.04, 0.08, 0.12, 0.16], yticks=[0, 0.5, 1],
             ylabel="Exact expected gain")
    gain.legend(loc="lower right", frameon=False, fontsize=6.5)
    for axis in axes:
        axis.set_xlabel(r"ESS $\rho$")
        axis.grid(True, which="major", color="#C9D1D9", alpha=0.48)
        axis.grid(False, which="minor")
    save_compact(fig, "figures/gaussian_policy_comparison_compact.pdf", preview_dir)


def reliability_diagnostics(preview_dir=None):
    """Recreate the original Figure 4's observations and annotations.

    The original PDF is the source for the displayed windows and the
    interpolated annotation at step 175. These choices and its ESS label
    are intentionally retained exactly; no onset is re-estimated here.
    """
    fig, (curve, scatter) = plt.subplots(1, 2, figsize=(WIDTH, 1.70),
                                       gridspec_kw={"width_ratios": [1.13, 1]})
    runs = [
        ("noclip_ungated", "Unclipped", "#355C7D", "o", 200),
        ("cispo3_nogate", "TIS-3", "#C06C50", "s", 80),
        ("cispo5_nogate", "TIS-5", "#5F8D7A", "D", 80),
    ]
    for run, label, color, marker, stop in runs:
        steps, scores = series(run, "eval", cut=stop)
        selected = [(x, 100 * y) for x, y in zip(steps, scores) if x >= 10]
        xs, ys = np.array(selected).T
        curve.plot(xs, ys, color=color, marker=marker, linewidth=1.4,
                   markersize=2.7, label=label)
        ess = dict(zip(*series(run, "ess")))
        evaluation = dict(zip(steps, scores))
        pairs = [(ess[x], 100 * (evaluation[x + 10] - evaluation[x]))
                 for x in steps if x >= 10 and x + 10 in evaluation and x in ess]
        x, y = np.array(pairs).T
        scatter.scatter(x, y, s=20, color=color, marker=marker, edgecolor="white",
                        linewidth=0.45, zorder=3)
    curve.set(xlim=(0, 202), ylim=(14, 47), xticks=[0, 50, 100, 150, 200],
              yticks=[15, 25, 35, 45], xlabel="training step",
              ylabel="AIME-2024 mean@16 (%)")
    for x, y, color, text, position in [
        (70, 31.25, "#C06C50", "ESS = 0.122", (83, 33.5)),
        (70, 27.9166666666667, "#5F8D7A", "ESS = 0.00374", (83, 25.7)),
        (175, 41.25, "#355C7D", "ESS = 0.037", (125, 46)),
    ]:
        curve.scatter([x], [y], s=37, facecolor="white", edgecolor=color,
                      linewidth=1.0, zorder=5)
        curve.annotate(text, xy=(x, y), xytext=position, fontsize=6.8, color=color,
                       ha="left", va="center", arrowprops=dict(arrowstyle="-",
                       color=color, linewidth=0.65, shrinkA=3, shrinkB=5))
    scatter.set(xscale="log", xlim=(0.0015, 0.8), ylim=(-9.5, 8.8),
                yticks=[-7.5, -5, -2.5, 0, 2.5, 5, 7.5], xlabel="ESS",
                ylabel="subsequent AIME\nchange (pp)")
    scatter.set_xticks([0.005, 0.01, 0.05, 0.1, 0.5], [".005", ".01", ".05", ".10", ".50"])
    scatter.axvspan(0.0015, 0.1, color="#777777", alpha=0.08, linewidth=0)
    scatter.axvspan(0.0015, 0.05, color="#777777", alpha=0.08, linewidth=0)
    scatter.axvline(0.1, color="#777777", linestyle=":", linewidth=0.8)
    scatter.axvline(0.05, color="#999999", linestyle="--", linewidth=0.7)
    scatter.axhline(0, color="#707070", linewidth=0.75)
    for axis in (curve, scatter):
        axis.grid(True, which="major", alpha=0.23)
        axis.grid(False, which="minor")
    handles, labels = curve.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False, fontsize=8, handlelength=2.1)
    save_compact(fig, "figures_mains/motivation/ess_reliability_collapse_compact.pdf", preview_dir)


def endpoint(axis, x, score, color, offset):
    """Place the final observed mean above/below the line, inside the axis."""
    axis.annotate(f"{format_sig(100 * score)}%", xy=(x, 100 * score),
                  xytext=(-2, offset), textcoords="offset points",
                  ha="right", va="bottom" if offset >= 0 else "top",
                  fontsize=7.2, color=color, clip_on=True,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.35))


def recipe_comparison(panels, relative_path, height, ylim, yticks, preview_dir=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, height), sharex=True, sharey=True)
    for i, (title, static_run, conditional_run, offsets) in enumerate(panels):
        axis = axes[i]
        for run, color, linestyle, linewidth, offset in [
            (static_run, STATIC, "--", 1.1, offsets[0]),
            (conditional_run, CONDITIONAL, "-", 1.45, offsets[1]),
        ]:
            x, y = series(run, "eval")
            axis.plot(x, np.array(y) * 100, color=color, linestyle=linestyle,
                      linewidth=linewidth, marker="o", markersize=2.4)
            endpoint(axis, x[-1], y[-1], color, offset)
        axis.set(xlim=(-3, 203), ylim=ylim, xticks=[0, 100, 200], yticks=yticks)
        axis.set_title(f"({'abcd'[i]}) {title}", loc="left", fontsize=8.0,
                       fontweight="bold", pad=3)
        axis.grid(True, which="major", alpha=0.26)
    axes[0].set_ylabel("AIME-2024 mean@16 (%)", fontsize=8)
    fig.supxlabel("training step", fontsize=8)
    handles = [
        Line2D([0], [0], color=STATIC, linestyle="--", linewidth=1.1,
               marker="o", markersize=2.4, label="Baseline" if len(panels) == 4 else "Clipped"),
        Line2D([0], [0], color=CONDITIONAL, linestyle="-", linewidth=1.45,
               marker="o", markersize=2.4, label="ESS-conditioned"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               frameon=False, fontsize=8, handlelength=2.3)
    save_compact(fig, relative_path, preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    compact_style()
    gaussian_comparison(args.preview_dir)
    reliability_diagnostics(args.preview_dir)
    recipe_comparison([
        ("GRPO", "grpo_base", "grpo_ess_clip", (-7, 6)),
        ("Clip-Higher", "dapo_base", "dapo_ess", (-7, 4)),
        ("DPPO", "dppo_base", "dppo_ess", (-7, 6)),
        ("GRPO-no clip", "noclip_ungated", "noclip_ess_clip", (-9, 9)),
    ], "figures_mains/result/q30ba3b/curves/overall_with_noclip_compact.pdf",
       1.84, (0, 53), [0, 25, 50], args.preview_dir)
    recipe_comparison([
        ("GRPO", "q8b_grpo_base", "q8b_grpo_ess", (-7, 7)),
        ("Clip-Higher", "q8b_dapo_base", "q8b_dapo_ess_nonorm", (-7, 6)),
        ("DPPO", "q8b_dppo_alwayslatch", "q8b_dppo_ess", (-7, 7)),
    ], "figures_mains/result/8b/curves/overall_with_dppo_compact.pdf",
       1.79, (10, 38), [10, 20, 30], args.preview_dir)


if __name__ == "__main__":
    main()
