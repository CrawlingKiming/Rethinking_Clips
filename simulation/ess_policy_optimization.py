"""Validate the chain from ESS to gradient MSE to policy optimization.

The environment uses the classification-to-contextual-bandit reduction from
prior off-policy evaluation work.  Optdigits images are contexts, digit labels
are actions, and a correct action receives reward one.  Because the population
is finite and fully labeled, every policy value and gradient is enumerable.

The script runs two connected experiments with 100 independent batches per
coverage condition:

1. A controlled logger sweep keeps one evaluation policy fixed and perturbs
   its logging policy independently of the gradient contributions.  This
   isolates the ESS effect on raw-gradient MSE and one-step policy improvement.
2. A prespecified ESS gate uses the unclipped update when sample ESS is at least
   0.1 and the actual PPO advantage-sign gradient mask otherwise.  The gate is
   compared with always-unclipped and always-clipped decisions across the same
   coverage sweep.
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
    coverage_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    one_step_learning_rate: float = 0.8
    ppo_epsilon: float = 1.0
    ess_threshold: float = 0.1


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
        improvements: dict[str, list[float]] = {
            "raw": [],
            "ppo": [],
            "gate": [],
        }
        alignments: list[float] = []
        sample_effective_sizes: list[float] = []
        gate_uses_raw: list[float] = []
        for _ in range(config.repetitions):
            batch = sample_logged_batch(
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
        gate_minus_raw = np.array(improvements["gate"]) - np.array(
            improvements["raw"]
        )
        gate_minus_ppo = np.array(improvements["gate"]) - np.array(
            improvements["ppo"]
        )
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


def decision_summary(
    coverage_rows: list[dict[str, float]], config: Config
) -> list[dict[str, float | str]]:
    methods = {
        "Unclipped": ("one_step_improvement", "one_step_improvement_se"),
        "PPO clipped": ("ppo_improvement", "ppo_improvement_se"),
        "ESS gated": ("gate_improvement", "gate_improvement_se"),
    }
    total = len(coverage_rows) * config.repetitions
    summary: list[dict[str, float | str]] = []
    for method, (mean_key, se_key) in methods.items():
        means = np.array([float(row[mean_key]) for row in coverage_rows])
        variances = np.array(
            [
                (float(row[se_key]) * math.sqrt(config.repetitions)) ** 2
                for row in coverage_rows
            ]
        )
        pooled_mean = float(np.mean(means))
        sum_squares = float(
            np.sum(
                (config.repetitions - 1) * variances
                + config.repetitions * (means - pooled_mean) ** 2
            )
        )
        pooled_se = math.sqrt(sum_squares / (total - 1)) / math.sqrt(total)
        row: dict[str, float | str] = {
                "method": method,
                "mean_one_step_gain": pooled_mean,
                "gain_se": pooled_se,
                "ess_threshold": config.ess_threshold
                if method == "ESS gated"
                else math.nan,
                "ppo_epsilon": config.ppo_epsilon,
                "coverage_conditions": len(coverage_rows),
                "repetitions_per_condition": config.repetitions,
                "difference_from_unclipped": math.nan,
                "difference_from_unclipped_se": math.nan,
                "difference_from_ppo": math.nan,
                "difference_from_ppo_se": math.nan,
            }
        if method == "ESS gated":
            for key, output_key in (
                ("gate_minus_raw", "difference_from_unclipped"),
                ("gate_minus_ppo", "difference_from_ppo"),
            ):
                differences = np.array(
                    [float(coverage_row[key]) for coverage_row in coverage_rows]
                )
                difference_variances = np.array(
                    [
                        (
                            float(coverage_row[f"{key}_se"])
                            * math.sqrt(config.repetitions)
                        )
                        ** 2
                        for coverage_row in coverage_rows
                    ]
                )
                pooled_difference = float(np.mean(differences))
                difference_sum_squares = float(
                    np.sum(
                        (config.repetitions - 1) * difference_variances
                        + config.repetitions
                        * (differences - pooled_difference) ** 2
                    )
                )
                row[output_key] = pooled_difference
                row[f"{output_key}_se"] = (
                    math.sqrt(difference_sum_squares / (total - 1))
                    / math.sqrt(total)
                )
        summary.append(row)
    return summary


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
    sample_ess = np.array([row["mean_sample_ess"] for row in coverage])

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

    decision_styles = {
        "Unclipped": (
            "one_step_improvement",
            "one_step_improvement_se",
            "#0072B2",
            "--",
        ),
        "PPO clipped": (
            "ppo_improvement",
            "ppo_improvement_se",
            "#D55E00",
            ":",
        ),
        "ESS gated": (
            "gate_improvement",
            "gate_improvement_se",
            "#009E73",
            "-",
        ),
    }
    for method, (mean_key, se_key, color, linestyle) in decision_styles.items():
        means = np.array([float(row[mean_key]) for row in coverage])
        standard_errors = np.array([float(row[se_key]) for row in coverage])
        axes[2].errorbar(
            sample_ess,
            means,
            yerr=1.96 * standard_errors,
            color=color,
            linestyle=linestyle,
            marker="o" if method == "ESS gated" else None,
            label=method,
        )
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    axes[2].axvline(0.1, color="#009E73", linewidth=0.8, alpha=0.7)
    axes[2].set_title("(c) ESS selects the update", fontsize=10)
    axes[2].set_xlabel("Mean sample ESS, $\\widehat\\rho$")
    axes[2].set_ylabel("One-step population gain")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes[:2]:
        axis.set_xlabel("Normalized ESS, $\\rho$")
    for axis in (axes[0], axes[2]):
        axis.set_xscale("log")
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
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 100:
        raise ValueError("repetitions must be between 2 and 100")

    config = Config(
        repetitions=args.repetitions,
    )
    features, labels = load_optdigits(args.data_dir)
    classifier_config = ClassifierConfig(training_steps=config.classifier_steps)
    fitted_weights = fit_softmax_classifier(features, labels, classifier_config)
    initial_weights = config.base_policy_scale * fitted_weights

    coverage_rows = coverage_experiment(features, labels, initial_weights, config)
    summary_rows = decision_summary(coverage_rows, config)
    coverage_csv = args.output_dir / "ess_coverage_results.csv"
    summary_csv = args.output_dir / "ess_gate_summary.csv"
    write_csv(coverage_rows, coverage_csv)
    write_csv(summary_rows, summary_csv)
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
    print(f"Wrote {summary_csv}")
    print(f"Prespecified ESS threshold: {config.ess_threshold:.2f}")
    print(f"Correlation rho, log(MSE): {rho_mse_correlation:.4f}")
    print(f"Correlation log(MSE), one-step gain: {mse_gain_correlation:.4f}")
    for row in summary_rows:
        print(
            f"{row['method']}: mean gain {float(row['mean_one_step_gain']):.5f} "
            f"+/- {float(row['gain_se']):.5f}"
        )


if __name__ == "__main__":
    main()
