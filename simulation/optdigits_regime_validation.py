"""Controlled Optdigits validation of the estimator-reliability theory.

The reported setting was selected in a pilot grid and is evaluated here on
independent seeds. The experiment compares the unmodified and PPO-masked
estimators under a globally certified step size. It produces two figures:

1. a cross-sectional ESS, MSE, and one-step-improvement comparison;
2. a long policy-iteration endpoint curve showing the cumulative regime change.

No ESS threshold, estimator oracle, or adaptive update rule is used.
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


RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
POPULATION_COLOR = "#2F8F78"
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"

ROLLOUT_SIZE = 160
MINIBATCHES = 40
MINIBATCH_SIZE = 4
POLICY_ITERATIONS = 25
LEARNING_RATE = 0.17
PPO_EPSILON = 0.20
INITIALIZATION_SCALE = 0.20

CURVE_SEED_START = 20400826
STATE_SEED_START = 20410826
STATE_REPLICATIONS = 12
STATE_ITERATIONS = (1, 3, 5, 8, 11, 15, 20, 25)
STATE_MINIBATCHES = (1, 5, 10, 20, 30, 40)
SELECTED_STATES = 48
REDRAWS = 80
IMPROVEMENT_REDRAWS = 40


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


def global_smoothness_bound(features: np.ndarray) -> tuple[float, float, float]:
    covariance = features.T @ features / len(features)
    lambda_max = float(np.linalg.eigvalsh(covariance)[-1])
    smoothness = 0.5 * lambda_max
    return lambda_max, smoothness, 1.0 / smoothness


def common_draws(
    rng: np.random.Generator,
    population_size: int,
    config: base.Config,
) -> list[dict[str, np.ndarray]]:
    return base.common_randomness(rng, population_size, config)


def run_endpoint_trajectory(
    method: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: base.Config,
) -> np.ndarray:
    weights = initial_weights.copy()
    values = [base.population_value(weights, features, labels)]
    for draw in draws:
        rollout_weights = weights.copy()
        rollout = base.collect_rollout(
            rollout_weights,
            features,
            labels,
            draw["contexts"],
            draw["uniforms"],
        )
        for indices in np.split(draw["order"], config.minibatches):
            gradients, _, _ = base.estimate_gradients(weights, rollout, indices, config)
            weights = weights + config.training_learning_rate * gradients[method]
        values.append(base.population_value(weights, features, labels))
    return np.asarray(values, dtype=float)


def collect_state_library(
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    config: base.Config,
) -> list[base.FrozenState]:
    states: list[base.FrozenState] = []
    state_id = 0
    for replication in range(STATE_REPLICATIONS):
        rng = np.random.default_rng(STATE_SEED_START + replication)
        draws = common_draws(rng, len(features), config)
        for method in ("raw", "ppo"):
            weights = initial_weights.copy()
            for iteration, draw in enumerate(draws, start=1):
                rollout_weights = weights.copy()
                rollout = base.collect_rollout(
                    rollout_weights,
                    features,
                    labels,
                    draw["contexts"],
                    draw["uniforms"],
                )
                for minibatch, indices in enumerate(
                    np.split(draw["order"], config.minibatches),
                    start=1,
                ):
                    if iteration in STATE_ITERATIONS and minibatch in STATE_MINIBATCHES:
                        rho = base.population_rho(weights, rollout_weights, features)
                        states.append(
                            base.FrozenState(
                                state_id=state_id,
                                trajectory=method,
                                replication=replication,
                                rollout_cycle=iteration,
                                minibatch=minibatch,
                                approximate_rho=rho,
                                weights=weights.copy(),
                                rollout_weights=rollout_weights.copy(),
                            )
                        )
                        state_id += 1
                    gradients, _, _ = base.estimate_gradients(
                        weights,
                        rollout,
                        indices,
                        config,
                    )
                    weights = weights + config.training_learning_rate * gradients[method]
    return states


def median_bin_rows(
    state_rows: list[dict[str, float]],
    bins: int = 8,
) -> list[dict[str, float]]:
    rho = np.asarray([row["population_rho"] for row in state_rows], dtype=float)
    edges = np.unique(np.quantile(rho, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(float(np.min(rho)), float(np.max(rho)) + 1e-8, 3)
    assignments = np.digitize(rho, edges[1:-1], right=True)
    output: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        indices = np.where(assignments == index)[0]
        if not len(indices):
            continue
        selected = [state_rows[j] for j in indices]
        row: dict[str, float] = {
            "bin": float(index),
            "rho_left": float(edges[index]),
            "rho_right": float(edges[index + 1]),
            "rho_median": float(np.median(rho[indices])),
            "states": float(len(indices)),
        }
        raw_risk = np.asarray([item["exact_raw_risk"] for item in selected])
        ppo_risk = np.asarray([item["exact_ppo_risk"] for item in selected])
        ratio = ppo_risk / np.maximum(raw_risk, 1e-16)
        for name, values in (("raw", raw_risk), ("ppo", ppo_risk)):
            row[f"{name}_mse_median"] = float(np.median(values))
            row[f"{name}_mse_q25"] = float(np.quantile(values, 0.25))
            row[f"{name}_mse_q75"] = float(np.quantile(values, 0.75))
            changes = np.asarray([item[f"{name}_mean_change"] for item in selected])
            harms = np.asarray([item[f"{name}_harm_rate"] for item in selected])
            row[f"{name}_mean_change"] = float(np.mean(changes))
            row[f"{name}_change_se"] = standard_error(changes)
            row[f"{name}_harm_rate"] = float(np.mean(harms))
        oracle_change = np.asarray([item["oracle_change"] for item in selected])
        row["population_mean_change"] = float(np.mean(oracle_change))
        row["population_change_se"] = standard_error(oracle_change)
        row["risk_ratio_median"] = float(np.median(ratio))
        row["risk_ratio_q25"] = float(np.quantile(ratio, 0.25))
        row["risk_ratio_q75"] = float(np.quantile(ratio, 0.75))
        row["ppo_lower_mse_fraction"] = float(np.mean(ppo_risk < raw_risk))
        output.append(row)
    return output


def curve_rows_and_summary(
    raw_curves: np.ndarray,
    ppo_curves: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    raw_mean = np.mean(raw_curves, axis=0)
    ppo_mean = np.mean(ppo_curves, axis=0)
    raw_se = np.std(raw_curves, axis=0, ddof=1) / math.sqrt(len(raw_curves))
    ppo_se = np.std(ppo_curves, axis=0, ddof=1) / math.sqrt(len(ppo_curves))
    gaps = raw_mean - ppo_mean
    persistent_crossover = -1
    for iteration in range(1, POLICY_ITERATIONS + 1):
        if np.all(gaps[iteration:] < 0.0):
            persistent_crossover = iteration
            break
    for iteration in range(POLICY_ITERATIONS + 1):
        rows.append(
            {
                "iteration": float(iteration),
                "raw_mean": float(raw_mean[iteration]),
                "raw_se": float(raw_se[iteration]),
                "ppo_mean": float(ppo_mean[iteration]),
                "ppo_se": float(ppo_se[iteration]),
                "raw_minus_ppo": float(gaps[iteration]),
            }
        )
    paired_final = raw_curves[:, -1] - ppo_curves[:, -1]
    early_slice = gaps[1:11]
    summary = {
        "replications": float(len(raw_curves)),
        "persistent_crossover_iteration": float(persistent_crossover),
        "max_early_raw_advantage": float(np.max(early_slice)),
        "max_early_raw_advantage_iteration": float(np.argmax(early_slice) + 1),
        "raw_final": float(raw_mean[-1]),
        "raw_final_se": float(raw_se[-1]),
        "ppo_final": float(ppo_mean[-1]),
        "ppo_final_se": float(ppo_se[-1]),
        "final_raw_minus_ppo": float(np.mean(paired_final)),
        "final_gap_se": standard_error(paired_final),
    }
    return rows, summary


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.6,
            "axes.titlesize": 10.7,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def make_mechanism_figure(
    bins: list[dict[str, float]],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.65))
    rho = np.asarray([row["rho_median"] for row in bins])
    order = np.argsort(rho)

    ax = axes[0]
    for name, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        median = np.asarray([row[f"{name}_mse_median"] for row in bins])
        lower = median - np.asarray([row[f"{name}_mse_q25"] for row in bins])
        upper = np.asarray([row[f"{name}_mse_q75"] for row in bins]) - median
        ax.errorbar(
            rho[order],
            median[order],
            yerr=np.vstack([lower[order], upper[order]]),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.4,
            capsize=3,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Exact gradient MSE")
    ax.set_title("Effective support controls estimator risk")
    ax.legend(frameon=False)

    ax = axes[1]
    ratio = np.asarray([row["risk_ratio_median"] for row in bins])
    q25 = np.asarray([row["risk_ratio_q25"] for row in bins])
    q75 = np.asarray([row["risk_ratio_q75"] for row in bins])
    ax.plot(rho[order], ratio[order], color=NEUTRAL_COLOR, marker="o", linewidth=2.0)
    ax.fill_between(
        rho[order], q25[order], q75[order], color=NEUTRAL_COLOR, alpha=0.16
    )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel(r"PPO MSE / unmodified MSE")
    ax.set_title("The lower-risk estimator changes by regime")

    ax = axes[2]
    for name, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        change = np.asarray([row[f"{name}_mean_change"] for row in bins])
        error = np.asarray([row[f"{name}_change_se"] for row in bins])
        ax.errorbar(
            rho[order],
            change[order],
            yerr=error[order],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.4,
            capsize=3,
            label=label,
        )
    population = np.asarray([row["population_mean_change"] for row in bins])
    population_se = np.asarray([row["population_change_se"] for row in bins])
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
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("Estimator reliability governs improvement")
    ax.legend(frameon=False)

    for panel, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def make_endpoint_figure(
    curve_rows: list[dict[str, float]],
    summary: dict[str, float],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    iteration = np.asarray([row["iteration"] for row in curve_rows])
    figure, ax = plt.subplots(figsize=(7.4, 4.25))
    for name, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        mean = np.asarray([row[f"{name}_mean"] for row in curve_rows])
        se = np.asarray([row[f"{name}_se"] for row in curve_rows])
        ax.plot(
            iteration,
            mean,
            color=color,
            marker=marker,
            markersize=4.3,
            linewidth=2.1,
            label=label,
        )
        ax.fill_between(
            iteration,
            mean - 1.96 * se,
            mean + 1.96 * se,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    crossover = int(summary["persistent_crossover_iteration"])
    if crossover > 0:
        ax.axvline(crossover, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.1)
        ax.text(
            crossover + 0.3,
            ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"persistent crossover: {crossover}",
            color=NEUTRAL_COLOR,
            fontsize=8.2,
            rotation=90,
            va="bottom",
        )
    ax.set_xlim(0, POLICY_ITERATIONS)
    ax.set_xticks(np.arange(0, POLICY_ITERATIONS + 1, 2))
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("Population value")
    ax.set_title("Signal preservation helps early; variance control helps later")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def summary_text(
    lambda_max: float,
    smoothness: float,
    eta_max: float,
    curve_summary: dict[str, float],
    bins: list[dict[str, float]],
) -> str:
    low = min(bins, key=lambda row: row["rho_median"])
    high = max(bins, key=lambda row: row["rho_median"])
    lines = [
        f"rollout_size={ROLLOUT_SIZE}",
        f"minibatches={MINIBATCHES}",
        f"minibatch_size={MINIBATCH_SIZE}",
        f"policy_iterations={POLICY_ITERATIONS}",
        f"learning_rate={LEARNING_RATE:.8f}",
        f"ppo_epsilon={PPO_EPSILON:.8f}",
        f"initialization_scale={INITIALIZATION_SCALE:.8f}",
        f"feature_cov_lambda_max={lambda_max:.8f}",
        f"global_smoothness_bound={smoothness:.8f}",
        f"certified_eta_max={eta_max:.8f}",
        f"learning_rate_condition_holds={float(LEARNING_RATE <= eta_max):.0f}",
    ]
    lines.extend(f"{key}={value:.8f}" for key, value in curve_summary.items())
    lines.extend(
        [
            f"low_ess_median={low['rho_median']:.8f}",
            f"low_raw_mse_median={low['raw_mse_median']:.8f}",
            f"low_ppo_mse_median={low['ppo_mse_median']:.8f}",
            f"low_risk_ratio_median={low['risk_ratio_median']:.8f}",
            f"low_ppo_lower_mse_fraction={low['ppo_lower_mse_fraction']:.8f}",
            f"low_raw_mean_change={low['raw_mean_change']:.8f}",
            f"low_ppo_mean_change={low['ppo_mean_change']:.8f}",
            f"high_ess_median={high['rho_median']:.8f}",
            f"high_raw_mse_median={high['raw_mse_median']:.8f}",
            f"high_ppo_mse_median={high['ppo_mse_median']:.8f}",
            f"high_risk_ratio_median={high['risk_ratio_median']:.8f}",
            f"high_ppo_lower_mse_fraction={high['ppo_lower_mse_fraction']:.8f}",
            f"high_raw_mean_change={high['raw_mean_change']:.8f}",
            f"high_ppo_mean_change={high['ppo_mean_change']:.8f}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = base.Config(
        replications=args.replications,
        rollout_cycles=POLICY_ITERATIONS,
        rollout_size=ROLLOUT_SIZE,
        minibatches=MINIBATCHES,
        training_learning_rate=LEARNING_RATE,
        diagnostic_step_size=LEARNING_RATE,
        ppo_epsilon=PPO_EPSILON,
        initialization_scale=INITIALIZATION_SCALE,
        redraw_batch_size=MINIBATCH_SIZE,
        redraws=REDRAWS,
        improvement_redraws=IMPROVEMENT_REDRAWS,
    )
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    initial_weights = base.fit_initial_policy(features, labels, config)
    lambda_max, smoothness, eta_max = global_smoothness_bound(features)
    if LEARNING_RATE > eta_max:
        raise RuntimeError("chosen learning rate violates the global smoothness bound")

    raw_curves: list[np.ndarray] = []
    ppo_curves: list[np.ndarray] = []
    for replication in range(args.replications):
        rng = np.random.default_rng(CURVE_SEED_START + replication)
        draws = common_draws(rng, len(features), config)
        raw_curves.append(
            run_endpoint_trajectory(
                "raw", initial_weights, features, labels, draws, config
            )
        )
        ppo_curves.append(
            run_endpoint_trajectory(
                "ppo", initial_weights, features, labels, draws, config
            )
        )
    raw_array = np.asarray(raw_curves)
    ppo_array = np.asarray(ppo_curves)
    curve_rows, curve_summary = curve_rows_and_summary(raw_array, ppo_array)

    library = collect_state_library(
        initial_weights,
        features,
        labels,
        config,
    )
    selected_states = base.choose_states(library, SELECTED_STATES)
    redraw_config = replace(
        config,
        redraw_batch_size=MINIBATCH_SIZE,
        redraws=REDRAWS,
        improvement_redraws=IMPROVEMENT_REDRAWS,
        diagnostic_step_size=LEARNING_RATE,
    )
    rng = np.random.default_rng(STATE_SEED_START + 100000)
    state_rows, draw_rows = base.evaluate_frozen_states(
        selected_states,
        features,
        labels,
        redraw_config,
        rng,
    )
    bins = median_bin_rows(state_rows)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_regime_curve.csv", curve_rows)
    write_csv(result_dir / "optdigits_regime_states.csv", state_rows)
    write_csv(result_dir / "optdigits_regime_redraws.csv", draw_rows)
    write_csv(result_dir / "optdigits_regime_bins.csv", bins)

    make_mechanism_figure(
        bins,
        root / "figures" / "optdigits_regime_mechanism",
    )
    make_endpoint_figure(
        curve_rows,
        curve_summary,
        root / "figures" / "optdigits_policy_iterations",
    )

    summary = summary_text(
        lambda_max,
        smoothness,
        eta_max,
        curve_summary,
        bins,
    )
    (result_dir / "optdigits_regime_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
