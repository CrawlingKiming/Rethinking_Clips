#!/usr/bin/env python
"""Temporal diagnostics for three permissive Qwen3-30B-A3B runs.

The figure aligns AIME-2024 validation, per-step normalized sequence ESS, and
gradient norm for TIS 3, TIS 5, and GRPO without clipping. For the two TIS
runs, diagnostic traces stop at the first evaluation below the step-0 score;
the validation curves remain complete. This avoids treating post-collapse ESS
recovery as evidence of renewed support. The no-clip run never crosses that
collapse criterion over its recorded trajectory.

The output is descriptive. It does not estimate a population gradient MSE or
claim that ESS alone determines update quality.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import C, FAM, FULL, use_paper_style, save
from runlog import series


paperstyle.FIGDIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "figures_mains",
    )
)

RUNS = [
    ("TIS 3", "cispo3_nogate", FAM[3]),
    ("TIS 5", "cispo5_nogate", FAM[1]),
    ("GRPO, no clip", "noclip_ungated", FAM[0]),
]
ESS_THRESHOLD = 0.1
SMOOTHING_WINDOW = 7


def first_validation_drop(run):
    """First evaluation step below the run's step-0 value, if it occurs."""
    steps, values = series(run, "eval")
    baseline = values[0]
    for step, value in zip(steps[1:], values[1:]):
        if value < baseline:
            return step
    return None


def first_ess_crossing(run):
    """First step at which normalized sequence ESS falls below 0.1."""
    steps, values = series(run, "ess")
    for step, value in zip(steps, values):
        if value < ESS_THRESHOLD:
            return step, value
    return None, None


def truncate_at_drop(run, steps, values):
    drop = first_validation_drop(run)
    if drop is None:
        return steps, values
    kept = [(step, value) for step, value in zip(steps, values) if step <= drop]
    return [item[0] for item in kept], [item[1] for item in kept]


def trailing_mean(values, window=SMOOTHING_WINDOW):
    smoothed = []
    for index in range(len(values)):
        local = values[max(0, index - window + 1) : index + 1]
        smoothed.append(sum(local) / len(local))
    return smoothed


use_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.65))
eval_axis, ess_axis, grad_axis = axes

for label, run, color in RUNS:
    eval_steps, eval_values = series(run, "eval")
    eval_axis.plot(
        eval_steps,
        [100.0 * value for value in eval_values],
        color=color,
        marker="o",
        linewidth=1.35,
        label=label,
    )
    drop = first_validation_drop(run)
    if drop is not None:
        drop_value = dict(zip(eval_steps, eval_values))[drop]
        eval_axis.scatter(
            [drop],
            [100.0 * drop_value],
            marker="X",
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )

    ess_steps, ess_values = truncate_at_drop(run, *series(run, "ess"))
    ess_axis.plot(ess_steps, ess_values, color=color, alpha=0.22, linewidth=0.65)
    ess_axis.plot(
        ess_steps,
        trailing_mean(ess_values),
        color=color,
        linewidth=1.35,
    )

    grad_steps, grad_values = truncate_at_drop(run, *series(run, "grad_norm"))
    grad_axis.plot(grad_steps, grad_values, color=color, alpha=0.22, linewidth=0.65)
    grad_axis.plot(
        grad_steps,
        trailing_mean(grad_values),
        color=color,
        linewidth=1.35,
    )

    crossing_step, crossing_ess = first_ess_crossing(run)
    if crossing_step is not None and crossing_step in set(ess_steps):
        ess_axis.scatter(
            [crossing_step],
            [crossing_ess],
            marker="v",
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        grad_at_step = dict(zip(grad_steps, grad_values)).get(crossing_step)
        if grad_at_step is not None:
            grad_axis.scatter(
                [crossing_step],
                [grad_at_step],
                marker="v",
                s=20,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=5,
            )

eval_axis.set_title("(a) Validation performance", loc="left")
eval_axis.set_ylabel("AIME-2024 mean@16 (%)")
eval_axis.set_ylim(-1.5, 48)
eval_axis.legend(loc="lower center", fontsize=6.4)

ess_axis.set_title("(b) Normalized sequence ESS", loc="left")
ess_axis.set_ylabel("Normalized sample ESS")
ess_axis.set_ylim(0.0, 0.65)
ess_axis.axhline(
    ESS_THRESHOLD,
    color=C["baseline"],
    linestyle=(0, (1, 2)),
    linewidth=0.9,
)

grad_axis.set_title("(c) Gradient norm", loc="left")
grad_axis.set_ylabel("Gradient norm")
grad_axis.set_yscale("log")

for axis in axes:
    axis.set_xlim(-3, 203)
    axis.set_xlabel("Training step")

save(fig, "result/q30ba3b/ungated_governs/ess_governs_update")

for label, run, _ in RUNS:
    drop = first_validation_drop(run)
    crossing_step, crossing_ess = first_ess_crossing(run)
    grad_steps, grad_values = series(run, "grad_norm")
    grad_at_crossing = dict(zip(grad_steps, grad_values)).get(crossing_step)
    print(
        f"{label}: ESS<0.1 at {crossing_step} "
        f"(ESS={crossing_ess:.4g}, grad_norm={grad_at_crossing:.4g}); "
        f"validation drop={drop}"
    )
