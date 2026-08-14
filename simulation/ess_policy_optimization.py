"""Validate the chain from ESS to gradient MSE to policy optimization.

The environment uses the classification-to-contextual-bandit reduction from
prior off-policy evaluation work.  Optdigits images are contexts, digit labels
are actions, and a correct action receives reward one.  Because the population
is finite and fully labeled, every policy value and gradient is enumerable.

The script runs three connected experiments with at most 100 independent
batches per condition:

1. A controlled rollout-policy sweep keeps one current policy fixed and
   perturbs its rollout policy independently of the gradient contributions.  This
   isolates the ESS effect on raw-gradient MSE and one-step policy improvement.
2. A prespecified ESS gate uses the unclipped update when sample ESS is at least
   0.1 and the actual PPO advantage-sign gradient mask otherwise.  The gate is
   compared with always-unclipped and always-clipped decisions across the same
   coverage sweep.
3. Sixteen-update fixed-rollout experiments cover both a fresh rollout, for
   which the rollout and current policies initially coincide, and a stale
   rollout with population ESS 0.0025.  The first checks that the gate preserves
   unmasked learning under adequate coverage; the second checks whether masking
   helps once the fixed batch has become unreliable.
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
    batch_size: int = 2048
    classifier_steps: int = 400
    base_policy_scale: float = 0.30
    coverage_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    one_step_learning_rate: float = 5.0
    ppo_epsilon: float = 0.2
    ess_threshold: float = 0.1
    optimization_rollout_levels: tuple[float, ...] = (0.0, 2.5)
    optimization_learning_rates: tuple[float, ...] = (2.0, 5.0)
    optimization_steps: int = 16


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


def population_masked_moments(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    behavior: np.ndarray,
    baseline: np.ndarray,
    true_gradient: np.ndarray,
    epsilon: float,
) -> dict[str, float | np.ndarray]:
    """Enumerate the bias, variance, and MSE of the PPO-masked estimator."""
    probabilities = softmax(features @ weights.T)
    samples, actions = probabilities.shape
    rewards = make_rewards(labels, actions)
    advantage = rewards - baseline[:, None]
    ratios = probabilities / behavior
    mask = ((advantage >= 0.0) & (ratios <= 1.0 + epsilon)) | (
        (advantage < 0.0) & (ratios >= 1.0 - epsilon)
    )
    coefficients = ratios * mask
    joint = behavior / samples

    factors = joint * coefficients * advantage
    residual_factors = factors - np.sum(factors, axis=1, keepdims=True) * probabilities
    mean_gradient = residual_factors.T @ features

    feature_norm_squared = np.sum(features**2, axis=1)
    probability_norm_squared = np.sum(probabilities**2, axis=1, keepdims=True)
    score_norm_squared = (
        1.0 - 2.0 * probabilities + probability_norm_squared
    ) * feature_norm_squared[:, None]
    contribution_norm_squared = advantage**2 * score_norm_squared
    second_moment = float(
        np.sum(joint * coefficients**2 * contribution_norm_squared)
    )
    variance_trace = second_moment - float(np.sum(mean_gradient**2))
    squared_bias = float(np.sum((mean_gradient - true_gradient) ** 2))
    return {
        "gradient": mean_gradient,
        "variance_trace": variance_trace,
        "squared_bias": squared_bias,
    }


def sample_rollout_batch(
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
    if method == "raw":
        coefficients = ratios
    elif method == "ppo":
        mask = ((advantage >= 0.0) & (ratios <= 1.0 + epsilon)) | (
            (advantage < 0.0) & (ratios >= 1.0 - epsilon)
        )
        coefficients = ratios * mask
    else:
        raise ValueError(f"Unknown method: {method}")

    weighted_residual = residual * (coefficients * advantage)[:, None]
    gradient = weighted_residual.T @ features[contexts] / len(contexts)
    ess = float(np.sum(ratios) ** 2 / (len(ratios) * np.sum(ratios**2)))
    return gradient, ess


def controlled_rollout_policies(
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
    rollout_rng = np.random.default_rng(config.seed + 1)
    batch_rng = np.random.default_rng(config.seed + 2)
    rows: list[dict[str, float]] = []
    for level, behavior in controlled_rollout_policies(
        target, config.coverage_levels, rollout_rng
    ):
        population = population_gradient_and_moments(
            features, labels, weights, behavior, baseline
        )
        true_gradient = np.asarray(population["gradient"])
        masked_population = population_masked_moments(
            features,
            labels,
            weights,
            behavior,
            baseline,
            true_gradient,
            config.ppo_epsilon,
        )
        exact_weights = weights + config.one_step_learning_rate * true_gradient
        exact_value = policy_value(softmax(features @ exact_weights.T), labels)
        gradient_errors: list[float] = []
        ppo_gradient_errors: list[float] = []
        gate_gradient_errors: list[float] = []
        improvements: dict[str, list[float]] = {
            "raw": [],
            "ppo": [],
            "gate": [],
        }
        alignments: list[float] = []
        sample_effective_sizes: list[float] = []
        gate_uses_raw: list[float] = []
        for _ in range(config.repetitions):
            batch = sample_rollout_batch(
                batch_rng, behavior, labels, config.batch_size
            )
            estimate, sample_ess = batch_gradient(
                features,
                weights,
                behavior,
                baseline,
                batch,
                method="raw",
                epsilon=config.ppo_epsilon,
            )
            ppo_estimate, _ = batch_gradient(
                features,
                weights,
                behavior,
                baseline,
                batch,
                method="ppo",
                epsilon=config.ppo_epsilon,
            )
            gate_uses_unclipped = sample_ess >= config.ess_threshold
            gated_estimate = estimate if gate_uses_unclipped else ppo_estimate
            gradient_errors.append(float(np.sum((estimate - true_gradient) ** 2)))
            ppo_gradient_errors.append(
                float(np.sum((ppo_estimate - true_gradient) ** 2))
            )
            gate_gradient_errors.append(
                float(np.sum((gated_estimate - true_gradient) ** 2))
            )
            for name, candidate in (
                ("raw", estimate),
                ("ppo", ppo_estimate),
                ("gate", gated_estimate),
            ):
                updated = weights + config.one_step_learning_rate * candidate
                updated_value = policy_value(softmax(features @ updated.T), labels)
                improvements[name].append(
                    updated_value - float(population["value"])
                )
            sample_effective_sizes.append(sample_ess)
            gate_uses_raw.append(float(gate_uses_unclipped))
            denominator = float(np.linalg.norm(estimate) * np.linalg.norm(true_gradient))
            alignments.append(
                float(np.sum(estimate * true_gradient) / denominator)
                if denominator > 0.0
                else 0.0
            )

        theory_mse = float(population["variance_trace"]) / config.batch_size
        ppo_theory_mse = float(masked_population["squared_bias"]) + (
            float(masked_population["variance_trace"]) / config.batch_size
        )
        gate_minus_raw = np.array(improvements["gate"]) - np.array(
            improvements["raw"]
        )
        gate_minus_ppo = np.array(improvements["gate"]) - np.array(
            improvements["ppo"]
        )
        rows.append(
            {
                "rollout_perturbation": level,
                "rho": float(population["rho"]),
                "effective_count": config.batch_size * float(population["rho"]),
                "g2": float(population["g2"]),
                "theory_mse": theory_mse,
                "ppo_theory_mse": ppo_theory_mse,
                "ppo_squared_bias": float(masked_population["squared_bias"]),
                "empirical_mse": float(np.mean(gradient_errors)),
                "empirical_mse_se": float(
                    np.std(gradient_errors, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "ppo_empirical_mse": float(np.mean(ppo_gradient_errors)),
                "ppo_empirical_mse_se": float(
                    np.std(ppo_gradient_errors, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "gate_empirical_mse": float(np.mean(gate_gradient_errors)),
                "gate_empirical_mse_se": float(
                    np.std(gate_gradient_errors, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "one_step_improvement": float(np.mean(improvements["raw"])),
                "one_step_improvement_se": float(
                    np.std(improvements["raw"], ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "ppo_improvement": float(np.mean(improvements["ppo"])),
                "ppo_improvement_se": float(
                    np.std(improvements["ppo"], ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "gate_improvement": float(np.mean(improvements["gate"])),
                "gate_improvement_se": float(
                    np.std(improvements["gate"], ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "gate_minus_raw": float(np.mean(gate_minus_raw)),
                "gate_minus_raw_se": float(
                    np.std(gate_minus_raw, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "gate_minus_ppo": float(np.mean(gate_minus_ppo)),
                "gate_minus_ppo_se": float(
                    np.std(gate_minus_ppo, ddof=1)
                    / math.sqrt(config.repetitions)
                ),
                "mean_sample_ess": float(np.mean(sample_effective_sizes)),
                "gate_unclipped_fraction": float(np.mean(gate_uses_raw)),
                "negative_update_rate": float(
                    np.mean(np.array(improvements["raw"]) < 0.0)
                ),
                "gradient_alignment": float(np.mean(alignments)),
                "exact_gradient_improvement": exact_value
                - float(population["value"]),
                "repetitions": config.repetitions,
                "batch_size": config.batch_size,
            }
        )
    return rows


def optimize_fixed_rollout(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    behavior: np.ndarray,
    baseline: np.ndarray,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: Config,
    method: str,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = initial_weights.copy()
    values = [policy_value(softmax(features @ weights.T), labels)]
    effective_sizes: list[float] = []
    raw_use: list[float] = []
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
        ppo_gradient, _ = batch_gradient(
            features,
            weights,
            behavior,
            baseline,
            batch,
            method="ppo",
            epsilon=config.ppo_epsilon,
        )
        if method == "Unclipped":
            gradient = raw_gradient
            use_raw = 1.0
        elif method == "PPO masked":
            gradient = ppo_gradient
            use_raw = 0.0
        elif method == "ESS gated":
            use_raw = float(ess >= config.ess_threshold)
            gradient = raw_gradient if use_raw else ppo_gradient
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        weights += learning_rate * gradient
        values.append(policy_value(softmax(features @ weights.T), labels))
        effective_sizes.append(ess)
        raw_use.append(use_raw)
    return np.array(values), np.array(effective_sizes), np.array(raw_use)


def optimization_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    config: Config,
    learning_rate: float,
    rollout_level: float,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    target = softmax(features @ initial_weights.T)
    baseline = target[np.arange(len(labels)), labels]
    rollout_policies = dict(
        controlled_rollout_policies(
            target,
            config.coverage_levels,
            np.random.default_rng(config.seed + 1),
        )
    )
    behavior = rollout_policies[rollout_level]
    population = population_gradient_and_moments(
        features, labels, initial_weights, behavior, baseline
    )
    rng = np.random.default_rng(config.seed + 3)
    methods = ("Unclipped", "PPO masked", "ESS gated")
    paths: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    gate_effective_sizes: list[np.ndarray] = []
    gate_raw_use: list[np.ndarray] = []
    for _ in range(config.repetitions):
        batch = sample_rollout_batch(rng, behavior, labels, config.batch_size)
        for method in methods:
            values, effective_sizes, raw_use = optimize_fixed_rollout(
                features,
                labels,
                initial_weights,
                behavior,
                baseline,
                batch,
                config,
                method,
                learning_rate,
            )
            paths[method].append(values)
            if method == "ESS gated":
                gate_effective_sizes.append(effective_sizes)
                gate_raw_use.append(raw_use)

    stacked = {method: np.stack(method_paths) for method, method_paths in paths.items()}
    initial_value = float(next(iter(stacked.values()))[0, 0])
    path_rows: list[dict[str, float | str]] = []
    for method, method_paths in stacked.items():
        gains = method_paths - method_paths[:, [0]]
        for step in range(config.optimization_steps + 1):
            path_rows.append(
                {
                    "method": method,
                    "step": step,
                    "mean_population_gain": float(np.mean(gains[:, step])),
                    "population_gain_se": float(
                        np.std(gains[:, step], ddof=1)
                        / math.sqrt(config.repetitions)
                    ),
                    "mean_population_value": float(np.mean(method_paths[:, step])),
                    "population_value_se": float(
                        np.std(method_paths[:, step], ddof=1)
                        / math.sqrt(config.repetitions)
                    ),
                    "mean_relative_improvement_pct": float(
                        100.0 * np.mean(gains[:, step]) / initial_value
                    ),
                    "relative_improvement_pct_se": float(
                        100.0
                        * np.std(gains[:, step], ddof=1)
                        / math.sqrt(config.repetitions)
                        / initial_value
                    ),
                    "learning_rate": learning_rate,
                    "rollout_perturbation": rollout_level,
                    "population_ess": float(population["rho"]),
                }
            )

    final_gains = {
        method: method_paths[:, -1] - method_paths[:, 0]
        for method, method_paths in stacked.items()
    }
    gate_minus_raw = final_gains["ESS gated"] - final_gains["Unclipped"]
    gate_minus_ppo = final_gains["ESS gated"] - final_gains["PPO masked"]
    gate_ess = np.stack(gate_effective_sizes)
    gate_raw = np.stack(gate_raw_use)
    summary_rows: list[dict[str, float | str]] = []
    for method in methods:
        gains = final_gains[method]
        summary_rows.append(
            {
                "method": method,
                "final_population_gain": float(np.mean(gains)),
                "final_population_gain_se": float(
                    np.std(gains, ddof=1) / math.sqrt(config.repetitions)
                ),
                "final_relative_improvement_pct": float(
                    100.0 * np.mean(gains) / initial_value
                ),
                "final_relative_improvement_pct_se": float(
                    100.0
                    * np.std(gains, ddof=1)
                    / math.sqrt(config.repetitions)
                    / initial_value
                ),
                "difference_from_unclipped": float(np.mean(gate_minus_raw))
                if method == "ESS gated"
                else math.nan,
                "difference_from_unclipped_se": float(
                    np.std(gate_minus_raw, ddof=1)
                    / math.sqrt(config.repetitions)
                )
                if method == "ESS gated"
                else math.nan,
                "relative_difference_from_unclipped_pct": float(
                    100.0 * np.mean(gate_minus_raw) / initial_value
                )
                if method == "ESS gated"
                else math.nan,
                "relative_difference_from_unclipped_pct_se": float(
                    100.0
                    * np.std(gate_minus_raw, ddof=1)
                    / math.sqrt(config.repetitions)
                    / initial_value
                )
                if method == "ESS gated"
                else math.nan,
                "difference_from_ppo": float(np.mean(gate_minus_ppo))
                if method == "ESS gated"
                else math.nan,
                "difference_from_ppo_se": float(
                    np.std(gate_minus_ppo, ddof=1)
                    / math.sqrt(config.repetitions)
                )
                if method == "ESS gated"
                else math.nan,
                "relative_difference_from_ppo_pct": float(
                    100.0 * np.mean(gate_minus_ppo) / initial_value
                )
                if method == "ESS gated"
                else math.nan,
                "relative_difference_from_ppo_pct_se": float(
                    100.0
                    * np.std(gate_minus_ppo, ddof=1)
                    / math.sqrt(config.repetitions)
                    / initial_value
                )
                if method == "ESS gated"
                else math.nan,
                "mean_gate_raw_updates": float(np.mean(np.sum(gate_raw, axis=1)))
                if method == "ESS gated"
                else math.nan,
                "mean_gate_clipped_updates": float(
                    config.optimization_steps - np.mean(np.sum(gate_raw, axis=1))
                )
                if method == "ESS gated"
                else math.nan,
                "mean_gate_ess": float(np.mean(gate_ess))
                if method == "ESS gated"
                else math.nan,
                "rollout_perturbation": rollout_level,
                "population_ess": float(population["rho"]),
                "learning_rate": learning_rate,
                "optimization_steps": config.optimization_steps,
                "repetitions": config.repetitions,
            }
        )
    return summary_rows, path_rows


def write_csv(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    coverage_rows: list[dict[str, float]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    coverage = sorted(coverage_rows, key=lambda row: row["rho"])
    rho = np.array([row["rho"] for row in coverage])
    theory_mse = np.array([row["theory_mse"] for row in coverage])
    ppo_theory_mse = np.array([row["ppo_theory_mse"] for row in coverage])
    improvement = 1e3 * np.array(
        [row["one_step_improvement"] for row in coverage]
    )
    improvement_se = np.array(
        [row["one_step_improvement_se"] for row in coverage]
    ) * 1e3
    exact_improvement = 1e3 * np.array(
        [row["exact_gradient_improvement"] for row in coverage]
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.15), constrained_layout=True)
    axes[0].plot(
        rho,
        theory_mse,
        color="#0072B2",
        linestyle="--",
        marker="o",
        label="Raw, oracle",
    )
    axes[0].plot(
        rho,
        ppo_theory_mse,
        color="#D55E00",
        linestyle="-.",
        marker="s",
        label="PPO mask, oracle",
    )
    axes[0].axvline(0.1, color="#888888", linestyle=":", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_title("(a) ESS predicts estimator error", fontsize=10)
    axes[0].set_ylabel("Gradient MSE")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].errorbar(
        rho,
        improvement,
        yerr=1.96 * improvement_se,
        color="#D55E00",
        marker="s",
        label="Raw estimate",
    )
    axes[1].plot(
        rho,
        exact_improvement,
        color="#222222",
        linestyle="--",
        label="Oracle gradient",
    )
    axes[1].axvline(0.1, color="#888888", linestyle=":", linewidth=0.9)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_title("(b) Error reduces update quality", fontsize=10)
    axes[1].set_ylabel("One-step reward gain ($\\times 10^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8)

    decision_styles = {
        "Raw": ("one_step_improvement", "one_step_improvement_se", "#0072B2", "--"),
        "PPO mask": ("ppo_improvement", "ppo_improvement_se", "#D55E00", ":"),
        "ESS gate": ("gate_improvement", "gate_improvement_se", "#009E73", "-"),
    }
    for method, (mean_key, se_key, color, linestyle) in decision_styles.items():
        means = 1e3 * np.array([row[mean_key] for row in coverage])
        standard_errors = 1e3 * np.array([row[se_key] for row in coverage])
        axes[2].errorbar(
            rho,
            means,
            yerr=1.96 * standard_errors,
            color=color,
            linestyle=linestyle,
            marker="o" if method == "ESS gate" else None,
            label=method,
        )
    axes[2].axvline(0.1, color="#888888", linestyle=":", linewidth=0.9)
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    axes[2].set_title("(c) ESS selects the update", fontsize=10)
    axes[2].set_xlabel("Normalized ESS, $\\rho$")
    axes[2].set_ylabel("One-step reward gain ($\\times 10^{-3}$)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(1e-3, 1.05)
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.set_xlabel("Normalized ESS, $\\rho$")
        axis.set_xscale("log")
        axis.set_xlim(1e-3, 1.05)
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
    parser.add_argument(
        "--optimization-rollout-levels",
        type=float,
        nargs="+",
        default=Config.optimization_rollout_levels,
    )
    parser.add_argument(
        "--optimization-learning-rates",
        type=float,
        nargs="+",
        default=Config.optimization_learning_rates,
    )
    parser.add_argument(
        "--optimization-steps",
        type=int,
        default=Config.optimization_steps,
    )
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 100:
        raise ValueError("repetitions must be between 2 and 100")

    config = Config(
        repetitions=args.repetitions,
        optimization_rollout_levels=tuple(args.optimization_rollout_levels),
        optimization_learning_rates=tuple(args.optimization_learning_rates),
        optimization_steps=args.optimization_steps,
    )
    features, labels = load_optdigits(args.data_dir)
    classifier_config = ClassifierConfig(training_steps=config.classifier_steps)
    fitted_weights = fit_softmax_classifier(features, labels, classifier_config)
    initial_weights = config.base_policy_scale * fitted_weights

    coverage_rows = coverage_experiment(features, labels, initial_weights, config)
    optimization_summary: list[dict[str, float | str]] = []
    optimization_paths: list[dict[str, float | str]] = []
    for rollout_level in config.optimization_rollout_levels:
        for learning_rate in config.optimization_learning_rates:
            rate_summary, rate_paths = optimization_experiment(
                features,
                labels,
                initial_weights,
                config,
                learning_rate,
                rollout_level,
            )
            optimization_summary.extend(rate_summary)
            optimization_paths.extend(rate_paths)
    coverage_csv = args.output_dir / "ess_coverage_results.csv"
    optimization_summary_csv = args.output_dir / "ess_optimization_summary.csv"
    optimization_paths_csv = args.output_dir / "ess_optimization_paths.csv"
    write_csv(coverage_rows, coverage_csv)
    write_csv(optimization_summary, optimization_summary_csv)
    write_csv(optimization_paths, optimization_paths_csv)
    if not args.skip_figure:
        make_figure(
            coverage_rows,
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
    print(f"Wrote {optimization_summary_csv}")
    print(f"Wrote {optimization_paths_csv}")
    print(f"Prespecified ESS threshold: {config.ess_threshold:.2f}")
    print(f"Correlation rho, log(MSE): {rho_mse_correlation:.4f}")
    print(f"Correlation log(MSE), one-step gain: {mse_gain_correlation:.4f}")
    for row in optimization_summary:
        print(
            f"{row['method']} (rollout perturbation "
            f"{float(row['rollout_perturbation']):g}, step size "
            f"{float(row['learning_rate']):g}): "
            f"{int(row['optimization_steps'])}-update relative improvement "
            f"{float(row['final_relative_improvement_pct']):.2f}% +/- "
            f"{float(row['final_relative_improvement_pct_se']):.2f}% "
            f"(absolute gain "
            f"{float(row['final_population_gain']):.5f} +/- "
            f"{float(row['final_population_gain_se']):.5f})"
        )


if __name__ == "__main__":
    main()
