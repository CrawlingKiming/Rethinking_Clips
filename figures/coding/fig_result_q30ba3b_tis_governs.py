#!/usr/bin/env python
"""Three support trajectories used in Section 5.

The first two panels reproduce the TIS failures from the motivation figure.
The third supplies the complementary unclipped-GRPO trajectory, where an
isolated low-ESS observation does not trigger immediate failure. All panels
use the same 0.01 ESS reference and omit gradient-norm diagnostics.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import paperstyle
from paperstyle import C, FULL, save, use_paper_style
from runlog import ess_cut, series


paperstyle.FIGDIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "figures_mains",
    )
)

PANELS = [
    ("cispo3_nogate", "TIS 3, no gate", True),
    ("cispo5_nogate", "TIS 5, no gate", True),
    ("noclip_ungated", "GRPO, no clipping", False),
]
ESS_REFERENCE = 0.01
X_LIMIT = 203


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
for index, (run, title, truncate_after_collapse) in enumerate(PANELS):
    validation_axis = validation_axes[index]
    eval_steps, eval_values = series(run, "eval")
    cut = ess_cut(run, ESS_REFERENCE) if truncate_after_collapse else None
    ess_steps, ess_values = series(run, "ess", cut=cut)
    full_ess_steps, full_ess_values = series(run, "ess")
    first_crossing = next(
        step
        for step, value in zip(full_ess_steps, full_ess_values)
        if value < ESS_REFERENCE
    )

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
    ess_axis.axvline(
        first_crossing,
        color=C["ours"],
        linestyle=":",
        linewidth=0.9,
    )
    if cut is not None:
        ess_axis.axvspan(cut, X_LIMIT, color=C["ours"], alpha=0.05, linewidth=0)
    ess_axis.annotate(
        f"ESS $<$ 0.01: step {first_crossing}",
        xy=(first_crossing + 3, 0.665),
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
