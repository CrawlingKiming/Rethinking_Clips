"""Validate the chain from ESS to gradient MSE to policy optimization.

The environment uses the classification-to-contextual-bandit reduction from
prior off-policy evaluation work.  Optdigits images are contexts, digit labels
are actions, and a correct action receives reward one.  Because the population
is finite and fully labeled, every policy value and gradient is enumerable.

The script runs two connected experiments with 100 independent batches:

1. A controlled logger sweep keeps one evaluation policy fixed and perturbs
   its logging policy independently of the gradient contributions.  This
   isolates the ESS effect on raw-gradient MSE and one-step policy improvement.
2. Fixed-rollout optimization compares an unclipped importance-weighted update,
   the actual PPO advantage-sign gradient mask, and an ESS-gated choice between
   them.  Forty runs select the ESS threshold and sixty runs evaluate it.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextual_bandit_ess import Config as ClassifierConfig
from contextual_bandit_ess import fit_softmax_classifier, load_optdigits, softmax


@dataclass(frozen=True)
class Config:
    seed: int = 37
    repetitions: int = 100
    batch_size: int = 256
    classifier_steps: int = 400
    base_policy_scale: float = 0.30
    behavior_exploration: float = 0.10
    coverage_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    one_step_learning_rate: float = 0.8
    optimization_learning_rate: float = 5.0
    optimization_steps: int = 50
    ppo_epsilon: float = 1.0
    validation_fraction: float = 0.4
    threshold_grid: tuple[float, ...] = (
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.90,
        0.95,
        0.97,
        0.98,
        0.99,
        0.995,
        0.999,
    )


def policy_value(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(probabilities[np.arange(len(labels)), labels]))


def make_rewards(labels: np.ndarray, actions: int) -> np.ndarray:
    return np.eye(actions)[labels]


def population_gradient_and_moments(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    behavior: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float | np.ndarray]:
    probabilities = softmax(features @ weights.T)
    samples, actions = probabilities.shape
    rewards = make_rewards(labels, actions)
    advantage = rewards - baseline[:, None]
    ratios = probabilities / behavior
    joint = behavior / samples

    factors = joint * ratios * advantage
    residual_factors = factors - np.sum(factors, axis=1, keepdims=True) * probabilities
    gradient = residual_factors.T @ features

    feature_norm_squared = np.sum(features**2, axis=1)
    probability_norm_squared = np.sum(probabilities**2, axis=1, keepdims=True)
    score_norm_squared = (
        1.0 - 2.0 * probabilities + probability_norm_squared
    ) * feature_norm_squared[:, None]
    contribution_norm_squared = advantage**2 * score_norm_squared
    weighted_second = float(
        np.sum(joint * ratios**2 * contribution_norm_squared)
    )
    ratio_second = float(np.sum(joint * ratios**2))
    rho = 1.0 / ratio_second
    g2 = weighted_second / ratio_second
    variance_trace = weighted_second - float(np.sum(gradient**2))
    return {
        "probabilities": probabilities,
        "gradient": gradient,
        "rho": rho,
        "g2": g2,
        "variance_trace": variance_trace,
        "value": policy_value(probabilities, labels),
    }


def sample_logged_batch(
    rng: np.random.Generator,
    behavior: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts = rng.integers(0, behavior.shape[0], size=batch_size)
    uniforms = rng.random(batch_size)
    cumulative = np.cumsum(behavior[contexts], axis=1)
    actions = np.sum(uniforms[:, None] > cumulative, axis=1)
    rewards = (actions == labels[contexts]).astype(float)
    return contexts, actions, rewards


def batch_gradient(
    features: np.ndarray,
    weights: np.ndarray,
    behavior: np.ndarray,
    baseline: np.ndarray,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    contexts, actions, rewards = batch
    probabilities = softmax(features[contexts] @ weights.T)
    row = np.arange(len(contexts))
    ratios = probabilities[row, actions] / behavior[contexts, actions]
    advantage = rewards - baseline[contexts]
    residual = -probabilities
    residual[row, actions] += 1.0
    score = residual[:, :, None] * features[contexts, None, :]
    contribution = advantage[:, None, None] * score

    if method == "raw":
        coefficients = ratios
    elif method == "ppo":
        mask = ((advantage >= 0.0) & (ratios <= 1.0 + epsilon)) | (
            (advantage < 0.0) & (ratios >= 1.0 - epsilon)
        )
        coefficients = ratios * mask
    else:
        raise ValueError(f"Unknown method: {method}")

    gradient = np.mean(coefficients[:, None, None] * contribution, axis=0)
    ess = float(np.sum(ratios) ** 2 / (len(ratios) * np.sum(ratios**2)))
    return gradient, ess


def controlled_loggers(
    target: np.ndarray,
    levels: tuple[float, ...],
    rng: np.random.Generator,
) -> list[tuple[float, np.ndarray]]:
    noise = rng.normal(size=target.shape)
    noise -= np.mean(noise, axis=1, keepdims=True)
    log_target = np.log(np.maximum(target, 1e-14))
    return [(level, softmax(log_target + level * noise)) for level in levels]


def coverage_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Config,
) -> list[dict[str, float]]:
    target = softmax(features @ weights.T)
    baseline = target[np.arange(len(labels)), labels]
    logger_rng = np.random.default_rng(config.seed + 1)
    batch_rng = np.random.default_rng(config.seed + 2)
    rows: list[dict[str, float]] = []
    for level, behavior in controlled_loggers(
        target, config.coverage_levels, logger_rng
    ):
        population = population_gradient_and_moments(
            features, labels, weights, behavior, baseline
        )
        true_gradient = np.asarray(population["gradient"])
        exact_weights = weights + config.one_step_learning_rate * true_gradient
        exact_value = policy_value(softmax(features @ exact_weights.T), labels)
        gradient_errors: list[float] = []
        improvements: list[float] = []
        alignments: list[float] = []
        for _ in range(config.repetitions):
            batch = sample_logged_batch(
                batch_rng, behavior, labels, config.batch_size
            )
            estimate, _ = batch_gradient(
                features,
                weights,
                behavior,
                baseline,
                batch,
                method="raw",
                epsilon=config.ppo_epsilon,
            )
            gradient_errors.append(float(np.sum((estimate - true_gradient) ** 2)))
            updated = weights + config.one_step_learning_rate * estimate
            updated_value = policy_value(softmax(features @ updated.T), labels)
            improvements.append(updated_value - float(population["value"]))
            denominator = float(np.linalg.norm(estimate) * np.linalg.norm(true_gradient))
            alignments.append(
                float(np.sum(estimate * true_gradient) / denominator)
                if denominator > 0.0
                else 0.0
            )

        theory_mse = float(population["variance_trace"]) / config.batch_size
        rows.append(
            {
                "logger_perturbation": level,
                "rho": float(population["rho"]),
                "effective_count": config.batch_size * float(population["rho"]),
                "g2": float(population["g2"]),
                "theory_mse": theory_mse,
                "empirical_mse": float(np.mean(gradient_errors)),
                "empirical_mse_se": float(
                    np.std(gradient_errors, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "one_step_improvement": float(np.mean(improvements)),
                "one_step_improvement_se": float(
                    np.std(improvements, ddof=1) / math.sqrt(config.repetitions)
                ),
                "negative_update_rate": float(np.mean(np.array(improvements) < 0.0)),
                "gradient_alignment": float(np.mean(alignments)),
                "exact_gradient_improvement": exact_value
                - float(population["value"]),
                "repetitions": config.repetitions,
                "batch_size": config.batch_size,
            }
        )
    return rows


def optimize_on_fixed_batch(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    behavior: np.ndarray,
    baseline: np.ndarray,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: Config,
    method: str,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = initial_weights.copy()
    values = [policy_value(softmax(features @ weights.T), labels)]
    effective_sizes: list[float] = []
    used_raw: list[float] = []
    for _ in range(config.optimization_steps):
        raw_gradient, ess = batch_gradient(
            features,
            weights,
            behavior,
            baseline,
            batch,
            method="raw",
            epsilon=config.ppo_epsilon,
        )
        if method == "raw":
            gradient = raw_gradient
            raw_flag = 1.0
        elif method == "ppo":
            gradient, _ = batch_gradient(
                features,
                weights,
                behavior,
                baseline,
                batch,
                method="ppo",
                epsilon=config.ppo_epsilon,
            )
            raw_flag = 0.0
        elif method == "gate":
            if threshold is None:
                raise ValueError("ESS gate requires a threshold")
            if ess >= threshold:
                gradient = raw_gradient
                raw_flag = 1.0
            else:
                gradient, _ = batch_gradient(
                    features,
                    weights,
                    behavior,
                    baseline,
                    batch,
                    method="ppo",
                    epsilon=config.ppo_epsilon,
                )
                raw_flag = 0.0
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        weights += config.optimization_learning_rate * gradient
        values.append(policy_value(softmax(features @ weights.T), labels))
        effective_sizes.append(ess)
        used_raw.append(raw_flag)
    return np.array(values), np.array(effective_sizes), np.array(used_raw)


def optimization_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    config: Config,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], float]:
    on_policy = softmax(features @ initial_weights.T)
    behavior = (
        (1.0 - config.behavior_exploration) * on_policy
        + config.behavior_exploration / on_policy.shape[1]
    )
    baseline = behavior[np.arange(len(labels)), labels]
    rng = np.random.default_rng(config.seed + 3)
    batches = [
        sample_logged_batch(rng, behavior, labels, config.batch_size)
        for _ in range(config.repetitions)
    ]

    raw_paths: list[np.ndarray] = []
    ppo_paths: list[np.ndarray] = []
    gate_paths: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in config.threshold_grid
    }
    gate_ess: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in config.threshold_grid
    }
    gate_raw: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in config.threshold_grid
    }
    for batch in batches:
        raw_path, _, _ = optimize_on_fixed_batch(
            features,
            labels,
            initial_weights,
            behavior,
            baseline,
            batch,
            config,
            method="raw",
        )
        ppo_path, _, _ = optimize_on_fixed_batch(
            features,
            labels,
            initial_weights,
            behavior,
            baseline,
            batch,
            config,
            method="ppo",
        )
        raw_paths.append(raw_path)
        ppo_paths.append(ppo_path)
        for threshold in config.threshold_grid:
            path, ess, raw_use = optimize_on_fixed_batch(
                features,
                labels,
                initial_weights,
                behavior,
                baseline,
                batch,
                config,
                method="gate",
                threshold=threshold,
            )
            gate_paths[threshold].append(path)
            gate_ess[threshold].append(ess)
            gate_raw[threshold].append(raw_use)

    validation_repetitions = int(
        config.repetitions * config.validation_fraction
    )
    selected_threshold = max(
        config.threshold_grid,
        key=lambda threshold: float(
            np.mean(
                np.stack(gate_paths[threshold])[:validation_repetitions, -1]
            )
        ),
    )
    method_paths = {
        "Unclipped": np.stack(raw_paths),
        "PPO clipped": np.stack(ppo_paths),
        "ESS gated": np.stack(gate_paths[selected_threshold]),
    }
    test_slice = slice(validation_repetitions, config.repetitions)
    summary_rows: list[dict[str, float | str]] = []
    path_rows: list[dict[str, float | str]] = []
    for method, paths in method_paths.items():
        test_paths = paths[test_slice]
        final = test_paths[:, -1]
        improvement = final - test_paths[:, 0]
        summary_rows.append(
            {
                "method": method,
                "final_value": float(np.mean(final)),
                "final_value_se": float(
                    np.std(final, ddof=1) / math.sqrt(len(final))
                ),
                "improvement": float(np.mean(improvement)),
                "improvement_se": float(
                    np.std(improvement, ddof=1) / math.sqrt(len(improvement))
                ),
                "selected_threshold": selected_threshold
                if method == "ESS gated"
                else math.nan,
                "ppo_epsilon": config.ppo_epsilon,
                "validation_repetitions": validation_repetitions,
                "test_repetitions": config.repetitions - validation_repetitions,
            }
        )
        for step in range(config.optimization_steps + 1):
            values = test_paths[:, step]
            path_rows.append(
                {
                    "method": method,
                    "step": step,
                    "mean_value": float(np.mean(values)),
                    "value_se": float(
                        np.std(values, ddof=1) / math.sqrt(len(values))
                    ),
                }
            )

    selected_ess = np.stack(gate_ess[selected_threshold])[test_slice]
    selected_raw_use = np.stack(gate_raw[selected_threshold])[test_slice]
    for step in range(config.optimization_steps):
        path_rows.append(
            {
                "method": "ESS gate diagnostics",
                "step": step,
                "mean_value": float(np.mean(selected_ess[:, step])),
                "value_se": float(np.mean(selected_raw_use[:, step])),
            }
        )
    return summary_rows, path_rows, selected_threshold


def write_csv(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    coverage_rows: list[dict[str, float]],
    path_rows: list[dict[str, float | str]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    coverage = sorted(coverage_rows, key=lambda row: row["rho"])
    rho = np.array([row["rho"] for row in coverage])
    empirical_mse = np.array([row["empirical_mse"] for row in coverage])
    empirical_mse_se = np.array([row["empirical_mse_se"] for row in coverage])
    theory_mse = np.array([row["theory_mse"] for row in coverage])
    improvement = np.array([row["one_step_improvement"] for row in coverage])
    improvement_se = np.array(
        [row["one_step_improvement_se"] for row in coverage]
    )
    exact_improvement = np.array(
        [row["exact_gradient_improvement"] for row in coverage]
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.15), constrained_layout=True)
    axes[0].errorbar(
        rho,
        empirical_mse,
        yerr=1.96 * empirical_mse_se,
        color="#0072B2",
        marker="o",
        label="Observed",
    )
    axes[0].plot(rho, theory_mse, color="#222222", linestyle="--", label="Exact")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) ESS predicts gradient error", fontsize=10)
    axes[0].set_ylabel("Raw-gradient MSE")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].errorbar(
        rho,
        improvement,
        yerr=1.96 * improvement_se,
        color="#D55E00",
        marker="s",
        label="Estimated gradient",
    )
    axes[1].plot(
        rho,
        exact_improvement,
        color="#222222",
        linestyle="--",
        label="Exact gradient",
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_title("(b) Error reduces update quality", fontsize=10)
    axes[1].set_ylabel("One-step population gain")
    axes[1].legend(frameon=False, fontsize=8)

    styles = {
        "Unclipped": ("#0072B2", "--"),
        "PPO clipped": ("#D55E00", ":"),
        "ESS gated": ("#009E73", "-"),
    }
    for method, (color, linestyle) in styles.items():
        selected = [row for row in path_rows if row["method"] == method]
        selected.sort(key=lambda row: int(row["step"]))
        steps = np.array([int(row["step"]) for row in selected])
        values = np.array([float(row["mean_value"]) for row in selected])
        standard_errors = np.array([float(row["value_se"]) for row in selected])
        axes[2].plot(steps, values, color=color, linestyle=linestyle, label=method)
        axes[2].fill_between(
            steps,
            values - 1.96 * standard_errors,
            values + 1.96 * standard_errors,
            color=color,
            alpha=0.12,
        )
    axes[2].set_title("(c) Fixed-rollout optimization", fontsize=10)
    axes[2].set_xlabel("Update step")
    axes[2].set_ylabel("Population reward")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes[:2]:
        axis.set_xlabel("Normalized ESS, $\\rho$")
    for axis in axes:
        axis.grid(True, color="#e6e6e6", linewidth=0.5)
        axis.tick_params(labelsize=8)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("simulation/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--repetitions", type=int, default=Config.repetitions)
    parser.add_argument("--optimization-steps", type=int, default=Config.optimization_steps)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 100:
        raise ValueError("repetitions must be between 2 and 100")

    config = Config(
        repetitions=args.repetitions,
        optimization_steps=args.optimization_steps,
    )
    features, labels = load_optdigits(args.data_dir)
    classifier_config = ClassifierConfig(training_steps=config.classifier_steps)
    fitted_weights = fit_softmax_classifier(features, labels, classifier_config)
    initial_weights = config.base_policy_scale * fitted_weights

    coverage_rows = coverage_experiment(features, labels, initial_weights, config)
    summary_rows, path_rows, threshold = optimization_experiment(
        features, labels, initial_weights, config
    )
    coverage_csv = args.output_dir / "ess_coverage_results.csv"
    summary_csv = args.output_dir / "policy_optimization_summary.csv"
    paths_csv = args.output_dir / "policy_optimization_paths.csv"
    write_csv(coverage_rows, coverage_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(path_rows, paths_csv)
    if not args.skip_figure:
        make_figure(
            coverage_rows,
            path_rows,
            args.figure_dir / "ess_policy_validation",
        )

    rho_mse_correlation = float(
        np.corrcoef(
            [row["rho"] for row in coverage_rows],
            np.log([row["empirical_mse"] for row in coverage_rows]),
        )[0, 1]
    )
    mse_gain_correlation = float(
        np.corrcoef(
            np.log([row["empirical_mse"] for row in coverage_rows]),
            [row["one_step_improvement"] for row in coverage_rows],
        )[0, 1]
    )
    print(f"Wrote {coverage_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {paths_csv}")
    print(f"Selected ESS threshold on validation runs: {threshold:.2f}")
    print(f"Correlation rho, log(MSE): {rho_mse_correlation:.4f}")
    print(f"Correlation log(MSE), one-step gain: {mse_gain_correlation:.4f}")
    for row in summary_rows:
        print(
            f"{row['method']}: final value {float(row['final_value']):.5f} "
            f"+/- {float(row['final_value_se']):.5f}"
        )


if __name__ == "__main__":
    main()
