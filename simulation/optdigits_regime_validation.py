"""Optdigits validation used in the paper.

The experiment has two parts. First, a controlled path moves a current policy
away from a fixed rollout policy and computes exact finite-population MSEs and
policy-improvement certificates. Second, a longer on-policy experiment reports
population value at the end of every policy iteration. The longitudinal setting
was selected in a pilot grid and is evaluated on a disjoint set of 100 seeds.
No ESS threshold, oracle update, or adaptive switching rule is used.
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
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"
BOTH_COLOR = "#E8F1EA"
PPO_ONLY_COLOR = "#FBEEDB"
NEITHER_COLOR = "#F2E7E8"

LEARNING_RATE = 0.17
PPO_EPSILON = 0.20

CONTROL_BATCH_SIZE = 320
CONTROL_INITIALIZATION_SCALE = 0.20
CONTROL_TARGET_SCALE = 1.00
CONTROL_PATH_MAX = 2.50
CONTROL_PATH_POINTS = 201
CONTROL_PLOT_RHO_MIN = 0.45

ROLLOUT_SIZE = 160
MINIBATCHES = 40
MINIBATCH_SIZE = 4
POLICY_ITERATIONS = 25
INITIALIZATION_SCALE = 0.20
CURVE_SEED_START = 20400826


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


def controlled_path_rows(
    features: np.ndarray,
    labels: np.ndarray,
    config: base.Config,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rollout_config = replace(
        config,
        initialization_scale=CONTROL_INITIALIZATION_SCALE,
    )
    target_config = replace(
        config,
        initialization_scale=CONTROL_TARGET_SCALE,
    )
    rollout_weights = base.fit_initial_policy(features, labels, rollout_config)
    target_weights = base.fit_initial_policy(features, labels, target_config)
    direction = target_weights - rollout_weights

    rows: list[dict[str, float]] = []
    for scale in np.linspace(0.0, CONTROL_PATH_MAX, CONTROL_PATH_POINTS):
        weights = rollout_weights + scale * direction
        rho = base.population_rho(weights, rollout_weights, features)
        value, gradient = base.population_value_and_gradient(weights, features, labels)
        signal_sq = float(np.sum(gradient**2))
        risks = base.exact_estimator_risks(
            weights,
            rollout_weights,
            features,
            labels,
            CONTROL_BATCH_SIZE,
            config,
        )
        raw_relative = risks["raw_risk"] / max(signal_sq, 1e-16)
        ppo_relative = risks["ppo_risk"] / max(signal_sq, 1e-16)
        raw_certificate = 0.5 * LEARNING_RATE * (signal_sq - risks["raw_risk"])
        ppo_certificate = 0.5 * LEARNING_RATE * (signal_sq - risks["ppo_risk"])
        rows.append(
            {
                "path_scale": float(scale),
                "population_rho": rho,
                "population_value": value,
                "signal_sq": signal_sq,
                "raw_mse": risks["raw_risk"],
                "ppo_mse": risks["ppo_risk"],
                "raw_relative_risk": raw_relative,
                "ppo_relative_risk": ppo_relative,
                "raw_certificate": raw_certificate,
                "ppo_certificate": ppo_certificate,
                "raw_certified": float(raw_certificate > 0.0),
                "ppo_certified": float(ppo_certificate > 0.0),
                "ppo_only_certified": float(
                    raw_certificate <= 0.0 and ppo_certificate > 0.0
                ),
            }
        )

    raw_loss = next(
        (row for row in rows if row["raw_certificate"] <= 0.0),
        rows[-1],
    )
    ppo_loss = next(
        (row for row in rows if row["ppo_certificate"] <= 0.0),
        rows[-1],
    )
    ppo_only = [row for row in rows if row["ppo_only_certified"] == 1.0]
    summary = {
        "control_batch_size": float(CONTROL_BATCH_SIZE),
        "high_rho_raw_relative_risk": rows[0]["raw_relative_risk"],
        "high_rho_ppo_relative_risk": rows[0]["ppo_relative_risk"],
        "raw_certificate_loss_rho": raw_loss["population_rho"],
        "ppo_certificate_loss_rho": ppo_loss["population_rho"],
        "ppo_only_rho_min": min(
            (row["population_rho"] for row in ppo_only),
            default=float("nan"),
        ),
        "ppo_only_rho_max": max(
            (row["population_rho"] for row in ppo_only),
            default=float("nan"),
        ),
        "ppo_only_points": float(len(ppo_only)),
    }
    return rows, summary


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
            gradients, _, _ = base.estimate_gradients(
                weights,
                rollout,
                indices,
                config,
            )
            weights = weights + config.training_learning_rate * gradients[method]
        values.append(base.population_value(weights, features, labels))
    return np.asarray(values, dtype=float)


def curve_rows_and_summary(
    raw_curves: np.ndarray,
    ppo_curves: np.ndarray,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    raw_mean = np.mean(raw_curves, axis=0)
    ppo_mean = np.mean(ppo_curves, axis=0)
    raw_se = np.std(raw_curves, axis=0, ddof=1) / math.sqrt(len(raw_curves))
    ppo_se = np.std(ppo_curves, axis=0, ddof=1) / math.sqrt(len(ppo_curves))
    gap = raw_mean - ppo_mean

    persistent_crossover = -1
    for iteration in range(1, POLICY_ITERATIONS + 1):
        if np.all(gap[iteration:] < 0.0):
            persistent_crossover = iteration
            break

    aggregate_rows: list[dict[str, float]] = []
    for iteration in range(POLICY_ITERATIONS + 1):
        aggregate_rows.append(
            {
                "iteration": float(iteration),
                "raw_mean": float(raw_mean[iteration]),
                "raw_se": float(raw_se[iteration]),
                "ppo_mean": float(ppo_mean[iteration]),
                "ppo_se": float(ppo_se[iteration]),
                "raw_minus_ppo": float(gap[iteration]),
            }
        )

    replication_rows: list[dict[str, float]] = []
    for replication in range(len(raw_curves)):
        for iteration in range(POLICY_ITERATIONS + 1):
            replication_rows.append(
                {
                    "replication": float(replication),
                    "iteration": float(iteration),
                    "raw_value": float(raw_curves[replication, iteration]),
                    "ppo_value": float(ppo_curves[replication, iteration]),
                    "raw_minus_ppo": float(
                        raw_curves[replication, iteration]
                        - ppo_curves[replication, iteration]
                    ),
                }
            )

    paired_final = raw_curves[:, -1] - ppo_curves[:, -1]
    early = gap[1:11]
    summary = {
        "replications": float(len(raw_curves)),
        "persistent_crossover_iteration": float(persistent_crossover),
        "max_early_raw_advantage": float(np.max(early)),
        "max_early_raw_advantage_iteration": float(np.argmax(early) + 1),
        "raw_final": float(raw_mean[-1]),
        "raw_final_se": float(raw_se[-1]),
        "ppo_final": float(ppo_mean[-1]),
        "ppo_final_se": float(ppo_se[-1]),
        "final_raw_minus_ppo": float(np.mean(paired_final)),
        "final_gap_se": standard_error(paired_final),
    }
    return aggregate_rows, replication_rows, summary


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.7,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.5,
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


def shade_regimes(
    ax: plt.Axes,
    raw_loss_rho: float,
    ppo_loss_rho: float,
) -> None:
    upper = 1.0
    lower = CONTROL_PLOT_RHO_MIN
    ax.axvspan(raw_loss_rho, upper, color=BOTH_COLOR, alpha=0.45, linewidth=0)
    ax.axvspan(
        ppo_loss_rho,
        raw_loss_rho,
        color=PPO_ONLY_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    ax.axvspan(lower, ppo_loss_rho, color=NEITHER_COLOR, alpha=0.42, linewidth=0)


def make_mechanism_figure(
    rows: list[dict[str, float]],
    summary: dict[str, float],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    visible = [row for row in rows if row["population_rho"] >= CONTROL_PLOT_RHO_MIN]
    rho = np.asarray([row["population_rho"] for row in visible])
    raw_loss = summary["raw_certificate_loss_rho"]
    ppo_loss = summary["ppo_certificate_loss_rho"]

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.72))

    ax = axes[0]
    shade_regimes(ax, raw_loss, ppo_loss)
    ax.plot(
        rho,
        [row["raw_mse"] for row in visible],
        color=RAW_COLOR,
        linewidth=2.1,
        label="Unmodified",
    )
    ax.plot(
        rho,
        [row["ppo_mse"] for row in visible],
        color=PPO_COLOR,
        linewidth=2.1,
        label="PPO masking",
    )
    ax.set_yscale("log")
    ax.set_xlim(1.0, CONTROL_PLOT_RHO_MIN)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Exact gradient MSE")
    ax.set_title("Effective support controls gradient error")
    ax.legend(frameon=False)

    ax = axes[1]
    shade_regimes(ax, raw_loss, ppo_loss)
    ax.plot(
        rho,
        [row["raw_relative_risk"] for row in visible],
        color=RAW_COLOR,
        linewidth=2.1,
        label="Unmodified",
    )
    ax.plot(
        rho,
        [row["ppo_relative_risk"] for row in visible],
        color=PPO_COLOR,
        linewidth=2.1,
        label="PPO masking",
    )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlim(1.0, CONTROL_PLOT_RHO_MIN)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel(r"Gradient MSE / $\|g\|_2^2$")
    ax.set_title("Reliability is lost when error reaches signal")

    ax = axes[2]
    shade_regimes(ax, raw_loss, ppo_loss)
    ax.plot(
        rho,
        [row["raw_certificate"] for row in visible],
        color=RAW_COLOR,
        linewidth=2.1,
        label="Unmodified",
    )
    ax.plot(
        rho,
        [row["ppo_certificate"] for row in visible],
        color=PPO_COLOR,
        linewidth=2.1,
        label="PPO masking",
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.1)
    ax.set_xlim(1.0, CONTROL_PLOT_RHO_MIN)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Expected-improvement certificate")
    ax.set_title("PPO briefly recovers a positive certificate")

    axes[1].text(
        0.97,
        0.08,
        "both certified",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=8.0,
        color=NEUTRAL_COLOR,
    )
    axes[1].text(
        0.31,
        0.08,
        "PPO only",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8.0,
        color=NEUTRAL_COLOR,
    )
    axes[1].text(
        0.05,
        0.08,
        "neither",
        transform=axes[1].transAxes,
        ha="left",
        fontsize=8.0,
        color=NEUTRAL_COLOR,
    )

    for panel, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def make_endpoint_figure(
    rows: list[dict[str, float]],
    summary: dict[str, float],
    output: Path,
) -> None:
    set_plot_defaults()
    output.parent.mkdir(parents=True, exist_ok=True)
    iteration = np.asarray([row["iteration"] for row in rows])
    figure, ax = plt.subplots(figsize=(7.6, 4.35))
    for name, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        mean = np.asarray([row[f"{name}_mean"] for row in rows])
        se = np.asarray([row[f"{name}_se"] for row in rows])
        ax.plot(
            iteration,
            mean,
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=4.2,
            markevery=2,
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
            crossover + 0.35,
            0.04,
            f"persistent crossover at iteration {crossover}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            fontsize=8.1,
            color=NEUTRAL_COLOR,
        )
    ax.set_xlim(0, POLICY_ITERATIONS)
    ax.set_xticks(np.arange(0, POLICY_ITERATIONS + 1, 2))
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("Population value")
    ax.set_title("Unmodified updates learn faster early; PPO is better later")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def summary_text(
    lambda_max: float,
    smoothness: float,
    eta_max: float,
    control_summary: dict[str, float],
    curve_summary: dict[str, float],
) -> str:
    values = {
        "control_batch_size": float(CONTROL_BATCH_SIZE),
        "rollout_size": float(ROLLOUT_SIZE),
        "minibatches": float(MINIBATCHES),
        "minibatch_size": float(MINIBATCH_SIZE),
        "policy_iterations": float(POLICY_ITERATIONS),
        "learning_rate": LEARNING_RATE,
        "ppo_epsilon": PPO_EPSILON,
        "initialization_scale": INITIALIZATION_SCALE,
        "feature_cov_lambda_max": lambda_max,
        "global_smoothness_bound": smoothness,
        "certified_eta_max": eta_max,
        "learning_rate_condition_holds": float(LEARNING_RATE <= eta_max),
    }
    values.update(control_summary)
    values.update(curve_summary)
    return "\n".join(f"{key}={value:.8f}" for key, value in values.items()) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    lambda_max, smoothness, eta_max = global_smoothness_bound(features)
    if LEARNING_RATE > eta_max:
        raise RuntimeError("learning rate violates the global smoothness bound")

    base_config = base.Config(
        training_learning_rate=LEARNING_RATE,
        ppo_epsilon=PPO_EPSILON,
    )
    control_rows, control_summary = controlled_path_rows(
        features,
        labels,
        base_config,
    )

    curve_config = replace(
        base_config,
        replications=args.replications,
        rollout_cycles=POLICY_ITERATIONS,
        rollout_size=ROLLOUT_SIZE,
        minibatches=MINIBATCHES,
        initialization_scale=INITIALIZATION_SCALE,
    )
    initial_weights = base.fit_initial_policy(features, labels, curve_config)
    raw_curves: list[np.ndarray] = []
    ppo_curves: list[np.ndarray] = []
    for replication in range(args.replications):
        rng = np.random.default_rng(CURVE_SEED_START + replication)
        draws = base.common_randomness(rng, len(features), curve_config)
        raw_curves.append(
            run_endpoint_trajectory(
                "raw",
                initial_weights,
                features,
                labels,
                draws,
                curve_config,
            )
        )
        ppo_curves.append(
            run_endpoint_trajectory(
                "ppo",
                initial_weights,
                features,
                labels,
                draws,
                curve_config,
            )
        )
    aggregate_rows, replication_rows, curve_summary = curve_rows_and_summary(
        np.asarray(raw_curves),
        np.asarray(ppo_curves),
    )

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_controlled_certificate_path.csv", control_rows)
    write_csv(result_dir / "optdigits_policy_iteration_curve.csv", aggregate_rows)
    write_csv(result_dir / "optdigits_policy_iteration_runs.csv", replication_rows)

    make_mechanism_figure(
        control_rows,
        control_summary,
        root / "figures" / "optdigits_reliability_transition",
    )
    make_endpoint_figure(
        aggregate_rows,
        curve_summary,
        root / "figures" / "optdigits_policy_iterations",
    )

    summary = summary_text(
        lambda_max,
        smoothness,
        eta_max,
        control_summary,
        curve_summary,
    )
    (result_dir / "optdigits_regime_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
