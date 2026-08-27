"""Add the exact-MSE oracle to the long Optdigits policy-iteration curve.

The oracle uses the same 100 paired rollout draws as the reported raw and PPO
trajectories. At each minibatch update it computes the exact finite-population
MSE of the unmodified and PPO-masked gradient estimators and applies the lower-
MSE estimator. If the two sampled gradients are identical, the update is shared
and the expensive exact-risk calculation is skipped.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import optdigits_categorical_theory as base


RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
ORACLE_COLOR = "#2F8F78"
LIGHT_GRID = "#D9DEE8"

ROLLOUT_SIZE = 160
MINIBATCHES = 40
POLICY_ITERATIONS = 25
LEARNING_RATE = 0.17
PPO_EPSILON = 0.20
INITIALIZATION_SCALE = 0.20
SEED_START = 20400826
REPLICATIONS = 100


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def read_existing_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: float(value) for key, value in row.items()} for row in rows]


def run_oracle(
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: base.Config,
) -> tuple[np.ndarray, float, float]:
    weights = initial_weights.copy()
    values = [base.population_value(weights, features, labels)]
    ppo_updates = 0
    distinct_updates = 0
    total_updates = 0

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
            total_updates += 1
            if np.array_equal(gradients["raw"], gradients["ppo"]):
                selected = "raw"
            else:
                distinct_updates += 1
                risks = base.exact_estimator_risks(
                    weights,
                    rollout_weights,
                    features,
                    labels,
                    len(indices),
                    config,
                )
                selected = "ppo" if risks["ppo_risk"] < risks["raw_risk"] else "raw"
            ppo_updates += int(selected == "ppo")
            weights = weights + config.training_learning_rate * gradients[selected]
        values.append(base.population_value(weights, features, labels))

    return (
        np.asarray(values, dtype=float),
        ppo_updates / max(total_updates, 1),
        distinct_updates / max(total_updates, 1),
    )


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.8,
            "axes.titlesize": 10.9,
            "axes.labelsize": 9.9,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "simulation" / "results"
    existing = read_existing_curve(result_dir / "optdigits_regime_curve.csv")

    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    config = base.Config(
        replications=REPLICATIONS,
        rollout_cycles=POLICY_ITERATIONS,
        rollout_size=ROLLOUT_SIZE,
        minibatches=MINIBATCHES,
        training_learning_rate=LEARNING_RATE,
        ppo_epsilon=PPO_EPSILON,
        initialization_scale=INITIALIZATION_SCALE,
    )
    initial_weights = base.fit_initial_policy(features, labels, config)
    initial_value = base.population_value(initial_weights, features, labels)

    oracle_curves: list[np.ndarray] = []
    ppo_fractions: list[float] = []
    distinct_fractions: list[float] = []
    for replication in range(REPLICATIONS):
        rng = np.random.default_rng(SEED_START + replication)
        draws = base.common_randomness(rng, len(features), config)
        curve, ppo_fraction, distinct_fraction = run_oracle(
            initial_weights,
            features,
            labels,
            draws,
            config,
        )
        oracle_curves.append(curve)
        ppo_fractions.append(ppo_fraction)
        distinct_fractions.append(distinct_fraction)

    oracle = np.asarray(oracle_curves)
    oracle_mean = np.mean(oracle, axis=0)
    oracle_se = np.std(oracle, axis=0, ddof=1) / math.sqrt(REPLICATIONS)

    rows: list[dict[str, float]] = []
    for iteration, prior in enumerate(existing):
        rows.append(
            {
                "iteration": float(iteration),
                "raw_mean": prior["raw_mean"],
                "raw_se": prior["raw_se"],
                "ppo_mean": prior["ppo_mean"],
                "ppo_se": prior["ppo_se"],
                "oracle_mean": float(oracle_mean[iteration]),
                "oracle_se": float(oracle_se[iteration]),
                "raw_relative_improvement": 100.0 * (prior["raw_mean"] - initial_value) / initial_value,
                "ppo_relative_improvement": 100.0 * (prior["ppo_mean"] - initial_value) / initial_value,
                "oracle_relative_improvement": 100.0 * (oracle_mean[iteration] - initial_value) / initial_value,
            }
        )

    output_csv = result_dir / "optdigits_regime_curve_with_oracle.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_raw = rows[-1]["raw_mean"]
    final_ppo = rows[-1]["ppo_mean"]
    final_oracle = rows[-1]["oracle_mean"]
    paired_oracle_raw = oracle[:, -1] - np.asarray([final_raw] * REPLICATIONS)
    paired_oracle_ppo = oracle[:, -1] - np.asarray([final_ppo] * REPLICATIONS)
    summary_lines = [
        f"initial_value={initial_value:.8f}",
        f"replications={REPLICATIONS}",
        f"raw_final={final_raw:.8f}",
        f"ppo_final={final_ppo:.8f}",
        f"oracle_final={final_oracle:.8f}",
        f"oracle_final_se={float(oracle_se[-1]):.8f}",
        f"raw_final_relative_improvement={rows[-1]['raw_relative_improvement']:.8f}",
        f"ppo_final_relative_improvement={rows[-1]['ppo_relative_improvement']:.8f}",
        f"oracle_final_relative_improvement={rows[-1]['oracle_relative_improvement']:.8f}",
        f"oracle_mean_ppo_fraction={float(np.mean(ppo_fractions)):.8f}",
        f"oracle_mean_distinct_fraction={float(np.mean(distinct_fractions)):.8f}",
        f"oracle_minus_raw_mean={final_oracle-final_raw:.8f}",
        f"oracle_minus_ppo_mean={final_oracle-final_ppo:.8f}",
    ]
    (result_dir / "optdigits_oracle_curve_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    set_plot_defaults()
    figure, ax = plt.subplots(figsize=(7.6, 4.45))
    iterations = np.asarray([row["iteration"] for row in rows])
    for name, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
        ("oracle", "Exact MSE oracle", ORACLE_COLOR, "D"),
    ):
        mean = np.asarray([row[f"{name}_relative_improvement"] for row in rows])
        if name == "oracle":
            value_se = np.asarray([row["oracle_se"] for row in rows])
        else:
            value_se = np.asarray([row[f"{name}_se"] for row in rows])
        relative_se = 100.0 * value_se / initial_value
        ax.plot(
            iterations,
            mean,
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=4.3,
            markevery=2,
            label=label,
        )
        ax.fill_between(
            iterations,
            mean - 1.96 * relative_se,
            mean + 1.96 * relative_se,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlim(0, POLICY_ITERATIONS)
    ax.set_xticks(np.arange(0, POLICY_ITERATIONS + 1, 2))
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("Relative improvement from initialization (%)")
    ax.set_title("Estimator choice changes learning across policy iterations")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figures" / "optdigits_policy_iterations.pdf", bbox_inches="tight")
    figure.savefig(root / "figures" / "optdigits_policy_iterations.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
