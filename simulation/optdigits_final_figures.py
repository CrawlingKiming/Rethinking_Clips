"""Create robust Optdigits estimator-comparison summaries and Figure 2."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_COLOR = "#355C8A"
PPO_COLOR = "#D9822B"
POPULATION_COLOR = "#2E8B78"
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def se(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def robust_bins(states: list[dict[str, str]], bins: int = 6) -> list[dict[str, float]]:
    ordered = sorted(states, key=lambda row: float(row["population_rho"]))
    groups = np.array_split(np.arange(len(ordered)), bins)
    output: list[dict[str, float]] = []
    for index, group in enumerate(groups):
        selected = [ordered[int(i)] for i in group]
        rho = np.asarray([float(row["population_rho"]) for row in selected])
        raw = np.asarray([float(row["exact_raw_risk"]) for row in selected])
        ppo = np.asarray([float(row["exact_ppo_risk"]) for row in selected])
        ratio = ppo / raw
        raw_change = np.asarray([float(row["raw_mean_change"]) for row in selected])
        ppo_change = np.asarray([float(row["ppo_mean_change"]) for row in selected])
        population_change = np.asarray(
            [float(row["oracle_change"]) for row in selected]
        )
        output.append(
            {
                "bin": float(index),
                "states": float(len(selected)),
                "rho_median": float(np.median(rho)),
                "rho_min": float(np.min(rho)),
                "rho_max": float(np.max(rho)),
                "raw_mse_median": float(np.median(raw)),
                "raw_mse_q25": float(np.quantile(raw, 0.25)),
                "raw_mse_q75": float(np.quantile(raw, 0.75)),
                "ppo_mse_median": float(np.median(ppo)),
                "ppo_mse_q25": float(np.quantile(ppo, 0.25)),
                "ppo_mse_q75": float(np.quantile(ppo, 0.75)),
                "ppo_raw_ratio_median": float(np.median(ratio)),
                "ppo_raw_ratio_q25": float(np.quantile(ratio, 0.25)),
                "ppo_raw_ratio_q75": float(np.quantile(ratio, 0.75)),
                "ppo_lower_mse_fraction": float(np.mean(ppo < raw)),
                "raw_mean_change": float(np.mean(raw_change)),
                "raw_change_se": se(raw_change),
                "ppo_mean_change": float(np.mean(ppo_change)),
                "ppo_change_se": se(ppo_change),
                "population_mean_change": float(np.mean(population_change)),
                "population_change_se": se(population_change),
                "raw_harm_rate": float(
                    np.mean([float(row["raw_harm_rate"]) for row in selected])
                ),
                "ppo_harm_rate": float(
                    np.mean([float(row["ppo_harm_rate"]) for row in selected])
                ),
            }
        )
    return output


def error_boundary_summary(redraws: list[dict[str, str]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for estimator in ("raw", "ppo"):
        selected = [
            row
            for row in redraws
            if row["estimator"] == estimator
            and np.isfinite(float(row["reward_change"]))
        ]
        below = [row for row in selected if float(row["relative_error"]) < 1.0]
        above = [row for row in selected if float(row["relative_error"]) >= 1.0]
        output[f"{estimator}_below_count"] = float(len(below))
        output[f"{estimator}_below_harm_rate"] = float(
            np.mean([float(row["reward_change"]) < 0.0 for row in below])
        )
        output[f"{estimator}_above_count"] = float(len(above))
        output[f"{estimator}_above_harm_rate"] = float(
            np.mean([float(row["reward_change"]) < 0.0 for row in above])
        )
    return output


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def asymmetric_error(
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
) -> np.ndarray:
    return np.vstack([median - q25, q75 - median])


def make_figure(
    rows: list[dict[str, float]],
    states: list[dict[str, str]],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    rho = np.asarray([row["rho_median"] for row in rows])

    ax = axes[0]
    for estimator, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        median = np.asarray([row[f"{estimator}_mse_median"] for row in rows])
        q25 = np.asarray([row[f"{estimator}_mse_q25"] for row in rows])
        q75 = np.asarray([row[f"{estimator}_mse_q75"] for row in rows])
        ax.errorbar(
            rho,
            median,
            yerr=asymmetric_error(median, q25, q75),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.2,
            capsize=3,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Median exact gradient MSE")
    ax.set_title("ESS governs estimator reliability")
    ax.legend(frameon=False)

    ax = axes[1]
    state_rho = np.asarray([float(row["population_rho"]) for row in states])
    state_ratio = np.asarray(
        [float(row["exact_ppo_risk"]) / float(row["exact_raw_risk"]) for row in states]
    )
    ax.scatter(
        state_rho,
        state_ratio,
        color=PPO_COLOR,
        alpha=0.28,
        s=19,
        linewidths=0,
    )
    ratio = np.asarray([row["ppo_raw_ratio_median"] for row in rows])
    q25 = np.asarray([row["ppo_raw_ratio_q25"] for row in rows])
    q75 = np.asarray([row["ppo_raw_ratio_q75"] for row in rows])
    ax.errorbar(
        rho,
        ratio,
        yerr=asymmetric_error(ratio, q25, q75),
        color=PPO_COLOR,
        marker="D",
        linewidth=2.0,
        markersize=5.0,
        capsize=3,
        label="Median with IQR",
    )
    ax.axhline(1.0, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel(r"PPO MSE / unmodified MSE")
    ax.set_title("Masking helps only when variance reduction wins")
    ax.legend(frameon=False)

    ax = axes[2]
    for key, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
        ("population", "Population gradient", POPULATION_COLOR, "D"),
    ):
        mean = np.asarray([row[f"{key}_mean_change"] for row in rows])
        error = np.asarray([row[f"{key}_change_se"] for row in rows])
        ax.errorbar(
            rho,
            mean,
            yerr=error,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.0,
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("Estimator reliability determines update quality")
    ax.legend(frameon=False)

    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    rows: list[dict[str, float]],
    boundary: dict[str, float],
    path: Path,
) -> None:
    low = rows[0]
    high = rows[-1]
    lines = []
    for prefix, row in (("low", low), ("high", high)):
        for key in (
            "rho_median",
            "raw_mse_median",
            "ppo_mse_median",
            "ppo_raw_ratio_median",
            "ppo_lower_mse_fraction",
            "raw_mean_change",
            "ppo_mean_change",
            "population_mean_change",
        ):
            lines.append(f"{prefix}_{key}={row[key]:.8f}")
    for key, value in boundary.items():
        lines.append(f"{key}={value:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "simulation" / "results"
    states = read_csv(result_dir / "optdigits_estimator_states.csv")
    redraws = read_csv(result_dir / "optdigits_estimator_redraws.csv")
    rows = robust_bins(states)
    boundary = error_boundary_summary(redraws)
    write_csv(result_dir / "optdigits_estimator_robust_bins.csv", rows)
    write_summary(
        rows,
        boundary,
        result_dir / "optdigits_estimator_robust_summary.txt",
    )
    make_figure(
        rows,
        states,
        root / "figures" / "optdigits_estimator_comparison",
    )


if __name__ == "__main__":
    main()
