#!/usr/bin/env python
"""ESS and validation along GRPO without clipping on Qwen3-30B-A3B.

Section 5 already refers to the two TIS failure trajectories in the motivation
figure. This companion panel supplies the complementary case: effective
support can be low while validation is still improving. Its visual grammar is
therefore copied from ``fig_motivation_ess.py``: black validation on the left
axis, red sequence ESS on the right axis, and the ESS threshold in red.

The low-support onset is defined from ESS alone as the first of five
consecutive observations below 0.1. The shaded interval runs from this onset
to the validation peak. No gradient-norm diagnostic is shown.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import C, FULL, save, use_paper_style
from runlog import series


paperstyle.FIGDIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "figures_mains",
    )
)

RUN = "noclip_ungated"
ESS_THRESHOLD = 0.1
SUSTAINED_LENGTH = 5
X_LIMIT = 203


use_paper_style()

eval_steps, eval_values = series(RUN, "eval")
ess_steps, ess_values = series(RUN, "ess")
peak_value = max(eval_values)
peak_step = eval_steps[eval_values.index(peak_value)]
eval_by_step = dict(zip(eval_steps, eval_values))

low_support_onset = next(
    ess_steps[index]
    for index in range(len(ess_steps) - SUSTAINED_LENGTH + 1)
    if all(
        value < ESS_THRESHOLD
        for value in ess_values[index : index + SUSTAINED_LENGTH]
    )
)
comparison_start = next(
    step for step in eval_steps if step >= low_support_onset
)

comparison_ess = [
    value
    for step, value in zip(ess_steps, ess_values)
    if comparison_start <= step <= peak_step
]
below_threshold = sum(value < ESS_THRESHOLD for value in comparison_ess)
first_crossing = next(
    (step for step, value in zip(ess_steps, ess_values) if value < ESS_THRESHOLD),
    None,
)

fig, validation_axis = plt.subplots(figsize=(0.72 * FULL, 2.35))

validation_line = validation_axis.plot(
    eval_steps,
    [100.0 * value for value in eval_values],
    color=C["eval"],
    linewidth=1.4,
    marker="o",
    label="AIME-2024 mean@16 (left)",
)[0]
validation_axis.axvline(
    peak_step,
    color=C["eval"],
    linestyle="--",
    linewidth=0.9,
)
validation_axis.set_xlim(-3, X_LIMIT)
validation_axis.set_ylim(10, 48)
validation_axis.set_xlabel("training step")
validation_axis.set_ylabel("AIME-2024 mean@16 (%)")
validation_axis.set_title("GRPO, no clipping", loc="left")

ess_axis = validation_axis.twinx()
ess_line = ess_axis.plot(
    ess_steps,
    ess_values,
    color=C["ours"],
    linewidth=1.1,
    label="ESS (right)",
)[0]
threshold_line = ess_axis.axhline(
    ESS_THRESHOLD,
    color=C["ours"],
    linestyle=(0, (1, 2)),
    linewidth=0.9,
    label="threshold 0.1",
)
ess_axis.axvline(
    low_support_onset,
    color=C["ours"],
    linestyle=":",
    linewidth=1.0,
)
ess_axis.axvspan(
    low_support_onset,
    peak_step,
    color=C["ours"],
    alpha=0.045,
    linewidth=0.0,
)
ess_axis.set_ylim(0, 0.66)
ess_axis.grid(False)
ess_axis.spines["right"].set_visible(True)
ess_axis.spines["right"].set_color(C["ours"])
ess_axis.tick_params(axis="y", colors=C["ours"])
ess_axis.set_ylabel("ESS (normalized)", color=C["ours"])

fig.legend(
    [validation_line, ess_line, threshold_line],
    ["AIME-2024 mean@16 (left)", "ESS (right)", "threshold 0.1"],
    loc="outside lower center",
    ncol=3,
    frameon=False,
)

save(fig, "result/q30ba3b/ungated_governs/ess_governs_update")

print(
    f"first ESS<0.1={first_crossing}; "
    f"sustained onset={low_support_onset}; "
    f"steps {comparison_start}--{peak_step}: "
    f"{below_threshold}/{len(comparison_ess)} ESS observations below 0.1; "
    f"AIME {100.0 * eval_by_step[comparison_start]:.3f}%"
    f" -> {100.0 * peak_value:.3f}%"
)
