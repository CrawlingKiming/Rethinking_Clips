#!/usr/bin/env python
"""Three support trajectories used in Section 5.

The first two panels reproduce the TIS failures from the motivation figure
and retain enough of each ESS trace to show a sustained low-support episode.
The third supplies the complementary unclipped-GRPO trajectory, where an
isolated low-ESS observation is followed by immediate recovery. All panels
use the same 0.01 ESS reference and omit gradient-norm diagnostics.
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

PANELS = [
    ("cispo3_nogate", "TIS 3, no gate", 100),
    ("cispo5_nogate", "TIS 5, no gate", 110),
    ("noclip_ungated", "GRPO, no clipping", None),
]
ESS_REFERENCE = 0.01
PERSISTENCE_LENGTH = 5
X_LIMIT = 203


def sustained_onset(steps, values):
    """First of five consecutive observations below the ESS reference."""
    for index in range(len(values) - PERSISTENCE_LENGTH + 1):
        window = values[index:index + PERSISTENCE_LENGTH]
        if all(value < ESS_REFERENCE for value in window):
            return steps[index]
    return None


use_paper_style()

fig, validation_axes = plt.subplots(
    1,
    3,
    figsize=(FULL, 2.45),
    sharex=True,
    sharey=True,
)

legend_handles = None
legend_labels = None
for index, (run, title, failure_step) in enumerate(PANELS):
    validation_axis = validation_axes[index]
    eval_steps, eval_values = series(run, "eval")
    full_ess_steps, full_ess_values = series(run, "ess")
    ess_steps, ess_values = series(run, "ess", cut=failure_step)
    first_crossing = next(
        step
        for step, value in zip(full_ess_steps, full_ess_values)
        if value < ESS_REFERENCE
    )
    onset = sustained_onset(full_ess_steps, full_ess_values)

    validation_line = validation_axis.plot(
        eval_steps,
        [100.0 * value for value in eval_values],
        color=C["eval"],
        linewidth=1.4,
        marker="o",
        label="AIME-2024 mean@16 (left)",
    )[0]
    validation_axis.set_xlim(-3, X_LIMIT)
    validation_axis.set_ylim(-1.5, 48)
    validation_axis.set_xlabel("training step")
    validation_axis.set_title(f"({'abc'[index]}) {title}", loc="left")
    if index == 0:
        validation_axis.set_ylabel("AIME-2024 mean@16 (%)")

    ess_axis = validation_axis.twinx()
    ess_line = ess_axis.plot(
        ess_steps,
        ess_values,
        color=C["ours"],
        linewidth=1.05,
        label="ESS (right)",
    )[0]
    reference_line = ess_axis.axhline(
        ESS_REFERENCE,
        color=C["ours"],
        linestyle=(0, (1, 2)),
        linewidth=0.9,
        label="ESS reference 0.01",
    )
    marker_step = onset if onset is not None else first_crossing
    ess_axis.axvline(marker_step, color=C["ours"], linestyle=":", linewidth=0.9)
    if onset is not None and failure_step is not None:
        ess_axis.axvspan(
            onset,
            failure_step,
            color=C["ours"],
            alpha=0.07,
            linewidth=0,
        )
        annotation = f"sustained low ESS: step {onset}"
    else:
        annotation = f"isolated dip: step {first_crossing}"
    ess_axis.annotate(
        annotation,
        xy=(marker_step + 3, 0.665),
        fontsize=6.0,
        color=C["ours"],
        va="top",
    )
    ess_axis.set_ylim(0, 0.68)
    ess_axis.grid(False)
    ess_axis.spines["right"].set_visible(True)
    ess_axis.spines["right"].set_color(C["ours"])
    ess_axis.tick_params(
        axis="y",
        colors=C["ours"],
        labelright=index == len(PANELS) - 1,
    )
    if index == len(PANELS) - 1:
        ess_axis.set_ylabel("ESS (normalized)", color=C["ours"])

    if legend_handles is None:
        legend_handles = [validation_line, ess_line, reference_line]
        legend_labels = [
            "AIME-2024 mean@16 (left)",
            "ESS (right)",
            "ESS reference 0.01",
        ]

fig.legend(
    legend_handles,
    legend_labels,
    loc="outside lower center",
    ncol=3,
    frameon=False,
)

save(fig, "result/q30ba3b/ungated_governs/ess_governs_update")
