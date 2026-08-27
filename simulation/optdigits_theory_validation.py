"""Optdigits experiments used only for theoretical validation.

The task is a one-step contextual bandit with ten categorical actions. The
script compares the unmodified and PPO-masked gradient estimators at the same
frozen policy states. It validates the relation from normalized ESS to gradient
MSE and from gradient error to one-step population change. It also reports a
40-update learning curve using a step size certified by a global smoothness
bound. No ESS-gated update rule is evaluated.
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
POPULATION_COLOR = "#2E8B78"
HARM_COLOR = "#B84A5A"
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"

STATE_SEED_START = 20260826
LEARNING_SEED_START = 20360826
ROLLOUT_SIZE = 320
MINIBATCHES = 8
MINIBATCH_SIZE = 40
STATE_GENERATION_ITERATIONS = 6
LEARNING_ITERATIONS = 5
LEARNING_RATE = 0.17
PPO_EPSILON = 0.2


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, float | str]],
) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def global_smoothness_bound(features: np.ndarray) -> tuple[float, float, float]:
    covariance = features.T @ features / len(features)
    lambda_max = float(np.linalg.eigvalsh(covariance)[-1])
    smoothness_bound = 0.5 * lambda_max
    return lambda_max, smoothness_bound, 1.0 / smoothness_bound


def estimator_error_bin_rows(
    draw_rows: list[dict[str, float]],
    estimator: str,
) -> list[dict[str, float | str]]:
    selected = [
        row
        for row in draw_rows
        if row["estimator"] == estimator and np.isfinite(row["reward_change"])
    ]
    boundaries = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])
    ratios = np.asarray([row["relative_error"] for row in selected], dtype=float)
    assignments = np.digitize(ratios, boundaries[1:-1], right=False)
    output: list[dict[str, float | str]] = []
    for index in range(len(boundaries) - 1):
        subset = [selected[j] for j in np.where(assignments == index)[0]]
        if not subset:
            continue
        changes = np.asarray([row["reward_change"] for row in subset], dtype=float)
        right = boundaries[index + 1]
        label = (
            f"{boundaries[index]:g}+"
            if np.isinf(right)
            else f"{boundaries[index]:g}-{right:g}"
        )
        output.append(
            {
                "estimator": estimator,
                "bin": float(index),
                "label": label,
                "relative_error_left": float(boundaries[index]),
                "relative_error_right": float(right),
                "relative_error_median": float(
                    np.median([row["relative_error"] for row in subset])
                ),
                "count": float(len(subset)),
                "mean_change": float(np.mean(changes)),
                "change_se": standard_error(changes),
                "harm_rate": float(np.mean(changes < 0.0)),
            }
        )
    return output


def ess_bin_rows(
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
        row: dict[str, float] = {
            "bin": float(index),
            "rho_left": float(edges[index]),
            "rho_right": float(edges[index + 1]),
            "rho_median": float(np.median(rho[selected_indices])),
            "states": float(len(selected)),
        }
        oracle_changes = np.asarray(
            [item["oracle_change"] for item in selected], dtype=float
        )
        row["population_gradient_change"] = float(np.mean(oracle_changes))
        row["population_gradient_change_se"] = standard_error(oracle_changes)
        for estimator in ("raw", "ppo"):
            exact_risks = np.asarray(
                [item[f"exact_{estimator}_risk"] for item in selected], dtype=float
            )
            redraw_mse = np.asarray(
                [item[f"{estimator}_mse"] for item in selected], dtype=float
            )
            changes = np.asarray(
                [item[f"{estimator}_mean_change"] for item in selected], dtype=float
            )
            harms = np.asarray(
                [item[f"{estimator}_harm_rate"] for item in selected], dtype=float
            )
            row[f"{estimator}_exact_mse"] = float(np.mean(exact_risks))
            row[f"{estimator}_exact_mse_se"] = standard_error(exact_risks)
            row[f"{estimator}_redraw_mse"] = float(np.mean(redraw_mse))
            row[f"{estimator}_redraw_mse_se"] = standard_error(redraw_mse)
            row[f"{estimator}_mean_change"] = float(np.mean(changes))
            row[f"{estimator}_change_se"] = standard_error(changes)
            row[f"{estimator}_harm_rate"] = float(np.mean(harms))
        row["ppo_lower_mse_fraction"] = float(
            np.mean(
                [
                    item["exact_ppo_risk"] < item["exact_raw_risk"]
                    for item in selected
                ]
            )
        )
        output.append(row)
    return output


def final_value_rows(
    summaries: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for method in ("raw", "ppo"):
        selected = [row for row in summaries if row["method"] == method]
        values = np.asarray([row["final_value"] for row in selected], dtype=float)
        output.append(
            {
                "method": method,
                "replications": float(len(values)),
                "mean_final_value": float(np.mean(values)),
                "se_final_value": standard_error(values),
                "median_final_value": float(np.median(values)),
            }
        )
    return output


def paired_learning_summary(
    summaries: list[dict[str, float]],
) -> dict[str, float]:
    by_replication: dict[int, dict[str, float]] = {}
    for row in summaries:
        replication = int(row["replication"])
        by_replication.setdefault(replication, {})[str(row["method"])] = float(
            row["final_value"]
        )
    differences = np.asarray(
        [values["raw"] - values["ppo"] for values in by_replication.values()],
        dtype=float,
    )
    return {
        "replications": float(len(differences)),
        "raw_minus_ppo": float(np.mean(differences)),
        "raw_minus_ppo_se": standard_error(differences),
    }


def curve_statistics(
    path_rows: list[dict[str, float]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in path_rows if row["method"] == method]
    updates = np.asarray(sorted({int(row["update"]) for row in selected}), dtype=int)
    means: list[float] = []
    errors: list[float] = []
    for update in updates:
        values = np.asarray(
            [
                row["population_value"]
                for row in selected
                if int(row["update"]) == update
            ],
            dtype=float,
        )
        means.append(float(np.mean(values)))
        errors.append(standard_error(values))
    return updates, np.asarray(means), np.asarray(errors)


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


def make_estimator_figure(
    bin_rows: list[dict[str, float]],
    error_rows: list[dict[str, float | str]],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.65))

    rho = np.asarray([row["rho_median"] for row in bin_rows], dtype=float)
    order = np.argsort(rho)

    ax = axes[0]
    for estimator, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        values = np.asarray(
            [row[f"{estimator}_exact_mse"] for row in bin_rows], dtype=float
        )
        errors = np.asarray(
            [row[f"{estimator}_exact_mse_se"] for row in bin_rows], dtype=float
        )
        ax.errorbar(
            rho[order],
            values[order],
            yerr=errors[order],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.4,
            capsize=3,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Exact gradient MSE")
    ax.set_title("PPO changes estimator reliability")
    ax.legend(frameon=False)

    ax = axes[1]
    raw_rows = [row for row in error_rows if row["estimator"] == "raw"]
    ppo_rows = [row for row in error_rows if row["estimator"] == "ppo"]
    labels = [str(row["label"]) for row in raw_rows]
    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    raw_harm = np.asarray([row["harm_rate"] for row in raw_rows], dtype=float)
    ppo_by_label = {str(row["label"]): row for row in ppo_rows}
    ppo_harm = np.asarray(
        [float(ppo_by_label[label]["harm_rate"]) if label in ppo_by_label else 0.0 for label in labels],
        dtype=float,
    )
    ax.bar(
        positions - width / 2,
        raw_harm,
        width=width,
        color=RAW_COLOR,
        alpha=0.9,
        label="Unmodified",
    )
    ax.bar(
        positions + width / 2,
        ppo_harm,
        width=width,
        color=PPO_COLOR,
        alpha=0.9,
        label="PPO masking",
    )
    ax.axvline(2.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.1)
    ax.set_xticks(positions, labels, rotation=28, ha="right")
    ax.set_xlabel(r"Realized squared error / $\|g\|_2^2$")
    ax.set_ylabel("Harmful-update rate")
    ax.set_title("Failure begins after error reaches signal scale")
    ax.legend(frameon=False)

    ax = axes[2]
    for estimator, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        values = np.asarray(
            [row[f"{estimator}_mean_change"] for row in bin_rows], dtype=float
        )
        errors = np.asarray(
            [row[f"{estimator}_change_se"] for row in bin_rows], dtype=float
        )
        ax.errorbar(
            rho[order],
            values[order],
            yerr=errors[order],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.4,
            capsize=3,
            label=label,
        )
    population = np.asarray(
        [row["population_gradient_change"] for row in bin_rows], dtype=float
    )
    population_se = np.asarray(
        [row["population_gradient_change_se"] for row in bin_rows], dtype=float
    )
    ax.errorbar(
        rho[order],
        population[order],
        yerr=population_se[order],
        color=POPULATION_COLOR,
        marker="D",
        linewidth=2.0,
        markersize=5.0,
        capsize=3,
        label="Population gradient",
    )
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("PPO trades learning signal for stability")
    ax.legend(frameon=False)

    for panel, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def make_learning_curve(
    path_rows: list[dict[str, float]],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(7.2, 4.15))
    for method, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        updates, means, errors = curve_statistics(path_rows, method)
        ax.plot(
            updates,
            means,
            color=color,
            linewidth=2.1,
            marker=marker,
            markevery=4,
            markersize=4.5,
            label=label,
        )
        ax.fill_between(
            updates,
            means - 1.96 * errors,
            means + 1.96 * errors,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    for boundary in range(MINIBATCHES, LEARNING_ITERATIONS * MINIBATCHES, MINIBATCHES):
        ax.axvline(boundary, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, LEARNING_ITERATIONS * MINIBATCHES)
    ax.set_xticks(np.arange(0, LEARNING_ITERATIONS * MINIBATCHES + 1, 5))
    ax.set_xlabel("Minibatch update")
    ax.set_ylabel("Population value")
    ax.set_title("Certified-step learning across all 40 updates")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def summary_text(
    initial_value: float,
    lambda_max: float,
    smoothness_bound: float,
    eta_max: float,
    state_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
    bin_rows: list[dict[str, float]],
    final_rows: list[dict[str, float | str]],
    paired: dict[str, float],
) -> str:
    by_method = {str(row["method"]): row for row in final_rows}
    low = min(bin_rows, key=lambda row: row["rho_median"])
    high = max(bin_rows, key=lambda row: row["rho_median"])
    lines = [
        f"initial_value={initial_value:.8f}",
        f"feature_cov_lambda_max={lambda_max:.8f}",
        f"global_smoothness_bound={smoothness_bound:.8f}",
        f"certified_eta_max={eta_max:.8f}",
        f"used_learning_rate={LEARNING_RATE:.8f}",
        f"learning_rate_condition_holds={float(LEARNING_RATE <= eta_max):.0f}",
        f"diagnostic_states={len(state_rows)}",
        f"low_ess_median={low['rho_median']:.8f}",
        f"low_ess_raw_mse={low['raw_exact_mse']:.8f}",
        f"low_ess_ppo_mse={low['ppo_exact_mse']:.8f}",
        f"low_ess_ppo_lower_mse_fraction={low['ppo_lower_mse_fraction']:.8f}",
        f"high_ess_median={high['rho_median']:.8f}",
        f"high_ess_raw_mse={high['raw_exact_mse']:.8f}",
        f"high_ess_ppo_mse={high['ppo_exact_mse']:.8f}",
        f"high_ess_ppo_lower_mse_fraction={high['ppo_lower_mse_fraction']:.8f}",
        f"final_raw={float(by_method['raw']['mean_final_value']):.8f}",
        f"final_raw_se={float(by_method['raw']['se_final_value']):.8f}",
        f"final_ppo={float(by_method['ppo']['mean_final_value']):.8f}",
        f"final_ppo_se={float(by_method['ppo']['se_final_value']):.8f}",
        f"raw_minus_ppo={paired['raw_minus_ppo']:.8f}",
        f"raw_minus_ppo_se={paired['raw_minus_ppo_se']:.8f}",
    ]
    for estimator in ("raw", "ppo"):
        finite = [
            row
            for row in draw_rows
            if row["estimator"] == estimator and np.isfinite(row["reward_change"])
        ]
        below = [row for row in finite if row["relative_error"] < 1.0]
        above = [row for row in finite if row["relative_error"] >= 1.0]
        lines.extend(
            [
                f"{estimator}_relative_error_below_one_count={len(below)}",
                f"{estimator}_relative_error_below_one_harm_rate={np.mean([row['reward_change'] < 0 for row in below]) if below else float('nan'):.8f}",
                f"{estimator}_relative_error_above_one_count={len(above)}",
                f"{estimator}_relative_error_above_one_harm_rate={np.mean([row['reward_change'] < 0 for row in above]) if above else float('nan'):.8f}",
            ]
        )
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
        rollout_size=ROLLOUT_SIZE,
        minibatches=MINIBATCHES,
        ppo_epsilon=PPO_EPSILON,
    )
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    initial_weights = base.fit_initial_policy(features, labels, template)
    initial_value = base.population_value(initial_weights, features, labels)
    lambda_max, smoothness_bound, eta_max = global_smoothness_bound(features)
    if LEARNING_RATE > eta_max:
        raise RuntimeError(
            f"learning rate {LEARNING_RATE} exceeds certified maximum {eta_max}"
        )

    state_config = replace(
        template,
        replications=args.diagnostic_replications,
        rollout_cycles=STATE_GENERATION_ITERATIONS,
        training_learning_rate=2.0,
        diagnostic_step_size=LEARNING_RATE,
        redraw_batch_size=MINIBATCH_SIZE,
        redraws=80,
        improvement_redraws=20,
        checkpoints=30,
    )
    all_states: list[base.FrozenState] = []
    next_state_id = 0
    for replication in range(state_config.replications):
        rng = np.random.default_rng(STATE_SEED_START + replication)
        draws = base.common_randomness(rng, len(features), state_config)
        for method in ("raw", "ppo"):
            states, _, _ = base.run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                state_config,
                replication,
                next_state_id,
                collect_states=True,
            )
            next_state_id += len(states)
            all_states.extend(states)

    selected_states = base.choose_states(all_states, state_config.checkpoints)
    diagnostic_rng = np.random.default_rng(STATE_SEED_START + 100000)
    state_rows, draw_rows = base.evaluate_frozen_states(
        selected_states,
        features,
        labels,
        state_config,
        diagnostic_rng,
    )
    bin_rows = ess_bin_rows(state_rows)
    error_rows = estimator_error_bin_rows(draw_rows, "raw")
    error_rows.extend(estimator_error_bin_rows(draw_rows, "ppo"))

    learning_config = replace(
        template,
        replications=args.replications,
        rollout_cycles=LEARNING_ITERATIONS,
        training_learning_rate=LEARNING_RATE,
    )
    path_rows: list[dict[str, float]] = []
    summaries: list[dict[str, float]] = []
    for replication in range(learning_config.replications):
        rng = np.random.default_rng(LEARNING_SEED_START + replication)
        draws = base.common_randomness(rng, len(features), learning_config)
        for method in ("raw", "ppo"):
            _, rows, summary = base.run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                learning_config,
                replication,
                0,
                collect_states=False,
            )
            path_rows.extend(rows)
            summaries.append(summary)

    final_rows = final_value_rows(summaries)
    paired = paired_learning_summary(summaries)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_estimator_states.csv", state_rows)
    write_csv(result_dir / "optdigits_estimator_redraws.csv", draw_rows)
    write_csv(result_dir / "optdigits_estimator_ess_bins.csv", bin_rows)
    write_csv(result_dir / "optdigits_estimator_error_bins.csv", error_rows)
    write_csv(result_dir / "optdigits_certified_learning_paths.csv", path_rows)
    write_csv(result_dir / "optdigits_certified_learning_runs.csv", summaries)
    write_csv(result_dir / "optdigits_certified_learning_final.csv", final_rows)
    write_csv(result_dir / "optdigits_certified_learning_pairwise.csv", [paired])

    make_estimator_figure(
        bin_rows,
        error_rows,
        root / "figures" / "optdigits_estimator_comparison",
    )
    make_learning_curve(
        path_rows,
        root / "figures" / "optdigits_certified_learning",
    )

    summary = summary_text(
        initial_value,
        lambda_max,
        smoothness_bound,
        eta_max,
        state_rows,
        draw_rows,
        bin_rows,
        final_rows,
        paired,
    )
    (result_dir / "optdigits_estimator_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
