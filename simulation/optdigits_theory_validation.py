"""Optdigits experiments used only for theoretical validation.

The task is a one-step contextual bandit with ten categorical actions. The
script validates two theoretical statements. First, population normalized ESS
controls finite-sample gradient reliability. Second, the lower-MSE estimator
between the unmodified and PPO-masked gradients gives the stronger update
certificate. No ESS-gated update rule is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import optdigits_categorical_theory as base


RAW_COLOR = "#355C8A"
PPO_COLOR = "#D9822B"
ORACLE_COLOR = "#2E8B78"
HARM_COLOR = "#B84A5A"
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"

DIAGNOSTIC_SEED_START = 20260826
CONTROL_SEED_START = 20310826


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def crossover_bin_rows(
    state_rows: list[dict[str, float]],
    bins: int = 6,
) -> list[dict[str, float]]:
    rho = np.asarray([row["population_rho"] for row in state_rows], dtype=float)
    edges = np.unique(np.quantile(rho, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(float(np.min(rho)), float(np.max(rho)) + 1e-8, 3)
    assignments = np.digitize(rho, edges[1:-1], right=True)
    output: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        selected_indices = np.where(assignments == index)[0]
        if not len(selected_indices):
            continue
        selected = [state_rows[j] for j in selected_indices]
        raw = np.asarray([row["exact_raw_risk"] for row in selected], dtype=float)
        ppo = np.asarray([row["exact_ppo_risk"] for row in selected], dtype=float)
        output.append(
            {
                "bin": float(index),
                "rho_left": float(edges[index]),
                "rho_right": float(edges[index + 1]),
                "rho_median": float(np.median(rho[selected_indices])),
                "states": float(len(selected)),
                "raw_risk": float(np.mean(raw)),
                "raw_risk_se": standard_error(raw),
                "ppo_risk": float(np.mean(ppo)),
                "ppo_risk_se": standard_error(ppo),
                "ppo_lower_risk_fraction": float(np.mean(ppo < raw)),
            }
        )
    return output


def final_value_rows(
    summaries: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    methods = ("raw", "ppo", "mse_oracle")
    output: list[dict[str, float | str]] = []
    for method in methods:
        selected = [row for row in summaries if row["method"] == method]
        values = np.asarray([row["final_value"] for row in selected], dtype=float)
        fractions = np.asarray([row["ppo_fraction"] for row in selected], dtype=float)
        output.append(
            {
                "method": method,
                "replications": float(len(values)),
                "mean_final_value": float(np.mean(values)),
                "se_final_value": standard_error(values),
                "median_final_value": float(np.median(values)),
                "mean_ppo_fraction": float(np.mean(fractions)),
                "se_ppo_fraction": standard_error(fractions),
            }
        )
    return output


def paired_rows(
    summaries: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    by_replication: dict[int, dict[str, float]] = {}
    for row in summaries:
        replication = int(row["replication"])
        by_replication.setdefault(replication, {})[str(row["method"])] = float(
            row["final_value"]
        )
    output: list[dict[str, float | str]] = []
    for comparison, left, right in (
        ("ppo_minus_raw", "ppo", "raw"),
        ("oracle_minus_raw", "mse_oracle", "raw"),
        ("oracle_minus_ppo", "mse_oracle", "ppo"),
    ):
        differences = np.asarray(
            [values[left] - values[right] for values in by_replication.values()],
            dtype=float,
        )
        output.append(
            {
                "comparison": comparison,
                "replications": float(len(differences)),
                "mean_difference": float(np.mean(differences)),
                "se_difference": standard_error(differences),
            }
        )
    return output


def iteration_curve(
    path_rows: list[dict[str, float]],
    method: str,
    minibatches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in path_rows
        if row["method"] == method and int(row["update"]) % minibatches == 0
    ]
    iterations = np.asarray(
        sorted({int(row["update"]) // minibatches for row in selected}),
        dtype=int,
    )
    means: list[float] = []
    errors: list[float] = []
    for iteration in iterations:
        values = np.asarray(
            [
                row["population_value"]
                for row in selected
                if int(row["update"]) // minibatches == iteration
            ],
            dtype=float,
        )
        means.append(float(np.mean(values)))
        errors.append(standard_error(values))
    return iterations, np.asarray(means), np.asarray(errors)


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


def make_crossover_figure(
    crossover_rows: list[dict[str, float]],
    path_rows: list[dict[str, float]],
    control_config: base.Config,
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    rho = np.asarray([row["rho_median"] for row in crossover_rows], dtype=float)
    order = np.argsort(rho)

    ax = axes[0]
    for key, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        values = np.asarray([row[f"{key}_risk"] for row in crossover_rows])
        errors = np.asarray([row[f"{key}_risk_se"] for row in crossover_rows])
        ax.errorbar(
            rho[order],
            values[order],
            yerr=errors[order],
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
    ax.set_ylabel("Exact gradient MSE")
    ax.set_title("Estimator risk across support regimes")
    ax.legend(frameon=False)

    ax = axes[1]
    fractions = np.asarray(
        [row["ppo_lower_risk_fraction"] for row in crossover_rows], dtype=float
    )
    ax.plot(
        rho[order],
        fractions[order],
        color=ORACLE_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
    )
    ax.axhline(0.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Fraction where PPO has lower MSE")
    ax.set_title("The preferred estimator changes by regime")

    ax = axes[2]
    for method, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO", PPO_COLOR, "s"),
        ("mse_oracle", "Exact MSE oracle", ORACLE_COLOR, "D"),
    ):
        iterations, means, errors = iteration_curve(
            path_rows,
            method,
            control_config.minibatches,
        )
        ax.plot(
            iterations,
            means,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            label=label,
        )
        ax.fill_between(
            iterations,
            means - 1.96 * errors,
            means + 1.96 * errors,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    ax.set_xticks(np.arange(control_config.rollout_cycles + 1))
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("Population value")
    ax.set_title("The MSE oracle combines both regimes")
    ax.legend(frameon=False)

    for panel, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def summary_text(
    initial_value: float,
    state_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
    ess_bins: list[dict[str, float]],
    crossover_rows: list[dict[str, float]],
    final_rows: list[dict[str, float | str]],
    comparisons: list[dict[str, float | str]],
) -> str:
    raw_draws = [
        row
        for row in draw_rows
        if row["estimator"] == "raw" and np.isfinite(row["reward_change"])
    ]
    below = [row for row in raw_draws if row["relative_error"] < 1.0]
    above = [row for row in raw_draws if row["relative_error"] >= 1.0]
    by_method = {str(row["method"]): row for row in final_rows}
    by_comparison = {str(row["comparison"]): row for row in comparisons}
    low_ess = min(ess_bins, key=lambda row: row["rho_median"])
    high_ess = max(ess_bins, key=lambda row: row["rho_median"])
    low_cross = min(crossover_rows, key=lambda row: row["rho_median"])
    high_cross = max(crossover_rows, key=lambda row: row["rho_median"])
    lines = [
        f"initial_value={initial_value:.8f}",
        f"diagnostic_states={len(state_rows)}",
        f"low_ess_median={low_ess['rho_median']:.8f}",
        f"low_ess_raw_mse={low_ess['raw_mse']:.8f}",
        f"high_ess_median={high_ess['rho_median']:.8f}",
        f"high_ess_raw_mse={high_ess['raw_mse']:.8f}",
        f"relative_error_below_one_count={len(below)}",
        f"relative_error_below_one_harm_rate={np.mean([row['reward_change'] < 0 for row in below]) if below else float('nan'):.8f}",
        f"relative_error_above_one_count={len(above)}",
        f"relative_error_above_one_harm_rate={np.mean([row['reward_change'] < 0 for row in above]) if above else float('nan'):.8f}",
        f"low_ess_ppo_lower_risk_fraction={low_cross['ppo_lower_risk_fraction']:.8f}",
        f"high_ess_ppo_lower_risk_fraction={high_cross['ppo_lower_risk_fraction']:.8f}",
        f"final_raw={float(by_method['raw']['mean_final_value']):.8f}",
        f"final_raw_se={float(by_method['raw']['se_final_value']):.8f}",
        f"final_ppo={float(by_method['ppo']['mean_final_value']):.8f}",
        f"final_ppo_se={float(by_method['ppo']['se_final_value']):.8f}",
        f"final_oracle={float(by_method['mse_oracle']['mean_final_value']):.8f}",
        f"final_oracle_se={float(by_method['mse_oracle']['se_final_value']):.8f}",
        f"oracle_ppo_fraction={float(by_method['mse_oracle']['mean_ppo_fraction']):.8f}",
        f"ppo_minus_raw={float(by_comparison['ppo_minus_raw']['mean_difference']):.8f}",
        f"ppo_minus_raw_se={float(by_comparison['ppo_minus_raw']['se_difference']):.8f}",
        f"oracle_minus_raw={float(by_comparison['oracle_minus_raw']['mean_difference']):.8f}",
        f"oracle_minus_raw_se={float(by_comparison['oracle_minus_raw']['se_difference']):.8f}",
        f"oracle_minus_ppo={float(by_comparison['oracle_minus_ppo']['mean_difference']):.8f}",
        f"oracle_minus_ppo_se={float(by_comparison['oracle_minus_ppo']['se_difference']):.8f}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--diagnostic-replications", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    template = base.Config(
        rollout_size=320,
        minibatches=8,
        training_learning_rate=2.0,
    )
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    initial_weights = base.fit_initial_policy(features, labels, template)
    initial_value = base.population_value(initial_weights, features, labels)

    diagnostic_config = replace(
        template,
        replications=args.diagnostic_replications,
        rollout_cycles=6,
    )
    all_states: list[base.FrozenState] = []
    next_state_id = 0
    for replication in range(diagnostic_config.replications):
        rng = np.random.default_rng(DIAGNOSTIC_SEED_START + replication)
        draws = base.common_randomness(rng, len(features), diagnostic_config)
        for method in ("raw", "ppo"):
            states, _, _ = base.run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                diagnostic_config,
                replication,
                next_state_id,
                collect_states=True,
            )
            next_state_id += len(states)
            all_states.extend(states)

    selected_states = base.choose_states(all_states, diagnostic_config.checkpoints)
    diagnostic_rng = np.random.default_rng(DIAGNOSTIC_SEED_START + 100000)
    state_rows, draw_rows = base.evaluate_frozen_states(
        selected_states,
        features,
        labels,
        diagnostic_config,
        diagnostic_rng,
    )
    ess_bins = base.quantile_bin_rows(state_rows)
    error_bins = base.relative_error_bin_rows(draw_rows)
    crossover_rows = crossover_bin_rows(state_rows)

    control_config = replace(
        template,
        replications=args.replications,
        rollout_cycles=2,
    )
    path_rows: list[dict[str, float]] = []
    summaries: list[dict[str, float]] = []
    for replication in range(control_config.replications):
        rng = np.random.default_rng(CONTROL_SEED_START + replication)
        draws = base.common_randomness(rng, len(features), control_config)
        for method in ("raw", "ppo", "mse_oracle"):
            _, rows, summary = base.run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                control_config,
                replication,
                0,
                collect_states=False,
            )
            path_rows.extend(rows)
            summaries.append(summary)

    final_rows = final_value_rows(summaries)
    comparisons = paired_rows(summaries)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_theory_frozen_states.csv", state_rows)
    write_csv(result_dir / "optdigits_theory_redraws.csv", draw_rows)
    write_csv(result_dir / "optdigits_theory_ess_bins.csv", ess_bins)
    write_csv(result_dir / "optdigits_theory_error_bins.csv", error_bins)
    write_csv(result_dir / "optdigits_crossover_bins.csv", crossover_rows)
    write_csv(result_dir / "optdigits_crossover_paths.csv", path_rows)
    write_csv(result_dir / "optdigits_crossover_runs.csv", summaries)
    write_csv(result_dir / "optdigits_crossover_final.csv", final_rows)
    write_csv(result_dir / "optdigits_crossover_pairwise.csv", comparisons)

    base.make_theory_figure(
        ess_bins,
        error_bins,
        root / "figures" / "optdigits_theory_validation",
    )
    make_crossover_figure(
        crossover_rows,
        path_rows,
        control_config,
        root / "figures" / "optdigits_mse_crossover",
    )

    summary = summary_text(
        initial_value,
        state_rows,
        draw_rows,
        ess_bins,
        crossover_rows,
        final_rows,
        comparisons,
    )
    (result_dir / "optdigits_theory_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
