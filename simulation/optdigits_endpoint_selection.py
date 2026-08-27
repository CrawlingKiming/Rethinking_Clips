"""Select and validate a long Optdigits policy-iteration endpoint experiment.

The search is restricted to step sizes that satisfy the global smoothness
certificate used in the paper. Exploratory seeds rank candidate designs by a
prespecified regime-transition score, validation seeds select among the top
candidates, and a disjoint final seed set is used for the reported curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import optdigits_categorical_theory as base


RAW_COLOR = "#355C8A"
PPO_COLOR = "#D9822B"
LIGHT_GRID = "#D9DEE8"
NEUTRAL_COLOR = "#667085"

EXPLORATION_SEED_START = 20400826
VALIDATION_SEED_START = 20500826
FINAL_SEED_START = 20600826

CANDIDATE_MINIBATCHES = (16, 20, 32, 40)
CANDIDATE_ITERATIONS = (10, 15, 20, 25)
CANDIDATE_LEARNING_RATES = (0.15, 0.17)


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_data(
    path_rows: list[dict[str, float]],
    minibatches: int,
    iterations: int,
    initial_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    by_update = {int(row["update"]): row for row in path_rows}
    values = [initial_value]
    minimum_ess = [1.0]
    for iteration in range(1, iterations + 1):
        endpoint = iteration * minibatches
        if endpoint not in by_update:
            raise RuntimeError(f"missing endpoint update {endpoint}")
        values.append(float(by_update[endpoint]["population_value"]))
        block = [
            float(row["population_rho"])
            for row in path_rows
            if (iteration - 1) * minibatches < int(row["update"]) <= endpoint
        ]
        minimum_ess.append(float(np.min(block)))
    return np.asarray(values), np.asarray(minimum_ess)


def run_design(
    config: base.Config,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    replications: int,
    seed_start: int,
    keep_paths: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, float]]]:
    initial_value = base.population_value(initial_weights, features, labels)
    values = {method: [] for method in ("raw", "ppo")}
    ess = {method: [] for method in ("raw", "ppo")}
    retained_rows: list[dict[str, float]] = []
    for replication in range(replications):
        rng = np.random.default_rng(seed_start + replication)
        draws = base.common_randomness(rng, len(features), config)
        for method in ("raw", "ppo"):
            _, path_rows, _ = base.run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                config,
                replication,
                0,
                collect_states=False,
            )
            endpoint_values, endpoint_ess = endpoint_data(
                path_rows,
                config.minibatches,
                config.rollout_cycles,
                initial_value,
            )
            values[method].append(endpoint_values)
            ess[method].append(endpoint_ess)
            if keep_paths:
                for iteration, (value, rho) in enumerate(
                    zip(endpoint_values, endpoint_ess)
                ):
                    retained_rows.append(
                        {
                            "replication": float(replication),
                            "method": method,
                            "policy_iteration": float(iteration),
                            "population_value": float(value),
                            "minimum_within_iteration_ess": float(rho),
                        }
                    )
    return (
        {key: np.asarray(value) for key, value in values.items()},
        {key: np.asarray(value) for key, value in ess.items()},
        retained_rows,
    )


def summarize_design(
    values: dict[str, np.ndarray],
    ess: dict[str, np.ndarray],
    config: base.Config,
    stage: str,
) -> dict[str, float | str]:
    raw = values["raw"]
    ppo = values["ppo"]
    gap = raw - ppo
    iterations = config.rollout_cycles
    window = max(2, min(5, iterations // 5))
    early_rep = np.mean(gap[:, 1 : window + 1], axis=1)
    late_rep = np.mean(gap[:, -window:], axis=1)
    final_rep = gap[:, -1]
    mean_gap = np.mean(gap, axis=0)
    crossover = next(
        (
            iteration
            for iteration in range(window + 1, iterations + 1)
            if mean_gap[iteration] <= 0.0
        ),
        999,
    )
    raw_mean = np.mean(raw, axis=0)
    ppo_mean = np.mean(ppo, axis=0)
    raw_decline = float(np.max(raw_mean) - raw_mean[-1])
    ppo_decline = float(np.max(ppo_mean) - ppo_mean[-1])
    early_ess = float(np.mean(ess["raw"][:, 1 : window + 1]))
    late_ess = float(np.mean(ess["raw"][:, -window:]))

    early = float(np.mean(early_rep))
    late = float(np.mean(late_rep))
    final = float(np.mean(final_rep))
    early_se = standard_error(early_rep)
    late_se = standard_error(late_rep)
    final_se = standard_error(final_rep)

    transition_score = (
        early / (early_se + 1e-4)
        - late / (late_se + 1e-4)
        - final / (final_se + 1e-4)
        + 15.0 * max(0.0, raw_decline - ppo_decline)
        + 2.0 * max(0.0, early_ess - late_ess)
    )
    transition_visible = float(
        early > 0.0
        and late < 0.0
        and final < 0.0
        and crossover < 999
        and early_ess > late_ess
    )

    return {
        "stage": stage,
        "rollout_size": float(config.rollout_size),
        "minibatches": float(config.minibatches),
        "minibatch_size": float(config.minibatch_size),
        "policy_iterations": float(config.rollout_cycles),
        "learning_rate": float(config.training_learning_rate),
        "replications": float(raw.shape[0]),
        "early_window": float(window),
        "raw_final": float(raw_mean[-1]),
        "ppo_final": float(ppo_mean[-1]),
        "early_raw_minus_ppo": early,
        "early_gap_se": early_se,
        "late_raw_minus_ppo": late,
        "late_gap_se": late_se,
        "final_raw_minus_ppo": final,
        "final_gap_se": final_se,
        "crossover_iteration": float(crossover),
        "raw_peak_drop": raw_decline,
        "ppo_peak_drop": ppo_decline,
        "early_minimum_ess": early_ess,
        "late_minimum_ess": late_ess,
        "transition_visible": transition_visible,
        "transition_score": float(transition_score),
    }


def search_designs(
    template: base.Config,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    exploration_replications: int,
    validation_replications: int,
) -> tuple[base.Config, list[dict[str, float | str]]]:
    exploration: list[dict[str, float | str]] = []
    for minibatches in CANDIDATE_MINIBATCHES:
        for iterations in CANDIDATE_ITERATIONS:
            for learning_rate in CANDIDATE_LEARNING_RATES:
                config = replace(
                    template,
                    minibatches=minibatches,
                    rollout_cycles=iterations,
                    training_learning_rate=learning_rate,
                )
                values, ess, _ = run_design(
                    config,
                    initial_weights,
                    features,
                    labels,
                    exploration_replications,
                    EXPLORATION_SEED_START,
                    keep_paths=False,
                )
                exploration.append(
                    summarize_design(values, ess, config, stage="exploration")
                )

    visible = [row for row in exploration if row["transition_visible"] == 1.0]
    ranked = sorted(
        visible if visible else exploration,
        key=lambda row: float(row["transition_score"]),
        reverse=True,
    )
    top = ranked[:5]

    validation: list[dict[str, float | str]] = []
    for row in top:
        config = replace(
            template,
            minibatches=int(float(row["minibatches"])),
            rollout_cycles=int(float(row["policy_iterations"])),
            training_learning_rate=float(row["learning_rate"]),
        )
        values, ess, _ = run_design(
            config,
            initial_weights,
            features,
            labels,
            validation_replications,
            VALIDATION_SEED_START,
            keep_paths=False,
        )
        validation.append(
            summarize_design(values, ess, config, stage="validation")
        )

    validation_visible = [
        row for row in validation if row["transition_visible"] == 1.0
    ]
    selected = max(
        validation_visible if validation_visible else validation,
        key=lambda row: float(row["transition_score"]),
    )
    selected_config = replace(
        template,
        minibatches=int(float(selected["minibatches"])),
        rollout_cycles=int(float(selected["policy_iterations"])),
        training_learning_rate=float(selected["learning_rate"]),
    )
    return selected_config, exploration + validation


def aggregate_endpoint_rows(
    rows: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    methods = ("raw", "ppo")
    iterations = sorted({int(row["policy_iteration"]) for row in rows})
    for method in methods:
        for iteration in iterations:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and int(row["policy_iteration"]) == iteration
            ]
            values = np.asarray(
                [row["population_value"] for row in selected], dtype=float
            )
            rho = np.asarray(
                [row["minimum_within_iteration_ess"] for row in selected],
                dtype=float,
            )
            output.append(
                {
                    "method": method,
                    "policy_iteration": float(iteration),
                    "mean_population_value": float(np.mean(values)),
                    "se_population_value": standard_error(values),
                    "mean_minimum_ess": float(np.mean(rho)),
                    "se_minimum_ess": standard_error(rho),
                }
            )
    return output


def make_endpoint_figure(
    aggregate_rows: list[dict[str, float | str]],
    summary: dict[str, float | str],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(7.4, 4.25))
    for method, label, color, marker in (
        ("raw", "Unmodified", RAW_COLOR, "o"),
        ("ppo", "PPO masking", PPO_COLOR, "s"),
    ):
        selected = sorted(
            [row for row in aggregate_rows if row["method"] == method],
            key=lambda row: float(row["policy_iteration"]),
        )
        x = np.asarray([row["policy_iteration"] for row in selected], dtype=float)
        mean = np.asarray(
            [row["mean_population_value"] for row in selected], dtype=float
        )
        error = np.asarray(
            [row["se_population_value"] for row in selected], dtype=float
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=4.4,
            label=label,
        )
        ax.fill_between(
            x,
            mean - 1.96 * error,
            mean + 1.96 * error,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    crossover = int(float(summary["crossover_iteration"]))
    if crossover < 999:
        ax.axvline(
            crossover,
            color=NEUTRAL_COLOR,
            linestyle=":",
            linewidth=1.1,
        )
        ax.text(
            crossover + 0.25,
            0.02,
            "mean crossover",
            transform=ax.get_xaxis_transform(),
            color=NEUTRAL_COLOR,
            fontsize=8.2,
            rotation=90,
            va="bottom",
        )
    max_iteration = int(float(summary["policy_iterations"]))
    ax.set_xlim(0, max_iteration)
    tick_step = 2 if max_iteration <= 20 else 5
    ax.set_xticks(np.arange(0, max_iteration + 1, tick_step))
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("Population value")
    ax.set_title("Long-run endpoint comparison")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def summary_text(
    summary: dict[str, float | str],
    smoothness_limit: float,
) -> str:
    keys = [
        "rollout_size",
        "minibatches",
        "minibatch_size",
        "policy_iterations",
        "learning_rate",
        "replications",
        "early_window",
        "raw_final",
        "ppo_final",
        "early_raw_minus_ppo",
        "early_gap_se",
        "late_raw_minus_ppo",
        "late_gap_se",
        "final_raw_minus_ppo",
        "final_gap_se",
        "crossover_iteration",
        "raw_peak_drop",
        "ppo_peak_drop",
        "early_minimum_ess",
        "late_minimum_ess",
        "transition_visible",
        "transition_score",
    ]
    lines = [f"certified_eta_max={smoothness_limit:.8f}"]
    for key in keys:
        value = summary[key]
        if isinstance(value, str):
            lines.append(f"{key}={value}")
        else:
            lines.append(f"{key}={float(value):.8f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-replications", type=int, default=8)
    parser.add_argument("--validation-replications", type=int, default=25)
    parser.add_argument("--final-replications", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    template = base.Config(
        rollout_size=320,
        minibatches=16,
        rollout_cycles=10,
        training_learning_rate=0.15,
        ppo_epsilon=0.2,
    )
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    initial_weights = base.fit_initial_policy(features, labels, template)
    covariance = features.T @ features / len(features)
    smoothness = 0.5 * float(np.linalg.eigvalsh(covariance)[-1])
    eta_max = 1.0 / smoothness
    if max(CANDIDATE_LEARNING_RATES) > eta_max:
        raise RuntimeError("candidate learning rate exceeds the certified maximum")

    selected_config, search_rows = search_designs(
        template,
        initial_weights,
        features,
        labels,
        args.exploration_replications,
        args.validation_replications,
    )
    final_values, final_ess, final_paths = run_design(
        selected_config,
        initial_weights,
        features,
        labels,
        args.final_replications,
        FINAL_SEED_START,
        keep_paths=True,
    )
    final_summary = summarize_design(
        final_values,
        final_ess,
        selected_config,
        stage="final_holdout",
    )
    if final_summary["transition_visible"] != 1.0:
        raise RuntimeError("the held-out curve does not show the selected transition")
    aggregate_rows = aggregate_endpoint_rows(final_paths)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_endpoint_design_search.csv", search_rows)
    write_csv(result_dir / "optdigits_endpoint_holdout_paths.csv", final_paths)
    write_csv(result_dir / "optdigits_endpoint_holdout_curve.csv", aggregate_rows)
    (result_dir / "optdigits_endpoint_summary.txt").write_text(
        summary_text(final_summary, eta_max),
        encoding="utf-8",
    )
    (result_dir / "optdigits_endpoint_selected_config.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    make_endpoint_figure(
        aggregate_rows,
        final_summary,
        root / "figures" / "optdigits_policy_iteration_endpoints",
    )
    print(summary_text(final_summary, eta_max))


if __name__ == "__main__":
    main()
