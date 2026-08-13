"""Exact contextual-bandit validation of the ESS reliability theory.

The target policy, reward model, and weighted gradient scale are fixed.  Only
the behavior policy changes, which changes sequence ESS.  Every population MSE
and every expected one-step policy value is computed by finite enumeration.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    batch_size: int = 32
    step_size: float = 20.0
    coverage_points: int = 31
    minimum_behavior_coverage: float = 0.005
    positive_contribution_probability: float = 0.75
    target_action_probability: float = 0.5
    contribution_magnitude: float = 0.25


METHODS: tuple[tuple[str, float | None], ...] = (
    ("raw", None),
    ("cap_3", 3.0),
    ("cap_5", 5.0),
)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def multinomial_probability(counts: tuple[int, ...], probabilities: tuple[float, ...]) -> float:
    sample_count = sum(counts)
    log_probability = math.lgamma(sample_count + 1.0)
    for count, probability in zip(counts, probabilities):
        log_probability -= math.lgamma(count + 1.0)
        if count:
            log_probability += count * math.log(probability)
    return math.exp(log_probability)


def coefficients(
    behavior_coverage: float,
    target_probability: float,
    cap: float | None,
) -> tuple[float, float]:
    correct_ratio = target_probability / behavior_coverage
    incorrect_ratio = (1.0 - target_probability) / (1.0 - behavior_coverage)
    if cap is None:
        return correct_ratio, incorrect_ratio
    return min(correct_ratio, cap), min(incorrect_ratio, cap)


def estimator_metrics(
    behavior_coverage: float,
    config: Config,
    cap: float | None,
) -> dict[str, float]:
    target_probability = config.target_action_probability
    positive_probability = config.positive_contribution_probability
    magnitude = config.contribution_magnitude
    correct_coefficient, incorrect_coefficient = coefficients(
        behavior_coverage, target_probability, cap
    )

    coefficient_mean = (
        behavior_coverage * correct_coefficient
        + (1.0 - behavior_coverage) * incorrect_coefficient
    )
    coefficient_second_moment = (
        behavior_coverage * correct_coefficient**2
        + (1.0 - behavior_coverage) * incorrect_coefficient**2
    )
    contribution_mean = magnitude * (2.0 * positive_probability - 1.0)
    true_gradient = contribution_mean
    estimator_mean = contribution_mean * coefficient_mean
    second_moment = magnitude**2 * coefficient_second_moment
    bias_squared = (estimator_mean - true_gradient) ** 2
    variance_over_n = (
        second_moment - estimator_mean**2
    ) / config.batch_size
    mse = bias_squared + variance_over_n

    return {
        "estimator_mean": estimator_mean,
        "bias_squared": bias_squared,
        "variance_over_n": variance_over_n,
        "mse": mse,
        "correct_coefficient": correct_coefficient,
        "incorrect_coefficient": incorrect_coefficient,
    }


def expected_policy_value(
    behavior_coverage: float,
    config: Config,
    cap: float | None,
) -> dict[str, float]:
    positive_probability = config.positive_contribution_probability
    magnitude = config.contribution_magnitude
    sample_count = config.batch_size
    correct_coefficient, incorrect_coefficient = coefficients(
        behavior_coverage, config.target_action_probability, cap
    )

    category_probabilities = (
        behavior_coverage * positive_probability,
        behavior_coverage * (1.0 - positive_probability),
        (1.0 - behavior_coverage) * positive_probability,
        (1.0 - behavior_coverage) * (1.0 - positive_probability),
    )
    expected_value = 0.0
    non_improvement_probability = 0.0
    probability_total = 0.0
    for correct_positive in range(sample_count + 1):
        remaining_after_correct_positive = sample_count - correct_positive
        for correct_negative in range(remaining_after_correct_positive + 1):
            remaining_after_correct = (
                remaining_after_correct_positive - correct_negative
            )
            for incorrect_positive in range(remaining_after_correct + 1):
                incorrect_negative = remaining_after_correct - incorrect_positive
                counts = (
                    correct_positive,
                    correct_negative,
                    incorrect_positive,
                    incorrect_negative,
                )
                probability = multinomial_probability(counts, category_probabilities)
                gradient = magnitude * (
                    correct_coefficient * (correct_positive - correct_negative)
                    + incorrect_coefficient * (incorrect_positive - incorrect_negative)
                ) / sample_count
                target_correct_probability = sigmoid(config.step_size * gradient)
                policy_value = 0.25 + 0.5 * target_correct_probability
                expected_value += probability * policy_value
                non_improvement_probability += probability * (gradient <= 0.0)
                probability_total += probability

    if abs(probability_total - 1.0) > 1e-10:
        raise AssertionError("Multinomial probabilities do not sum to one.")
    return {
        "expected_policy_value": expected_value,
        "non_improvement_probability": non_improvement_probability,
    }


def evaluate(config: Config) -> list[dict[str, float]]:
    behavior_coverages = np.unique(
        np.round(
            np.concatenate(
            (
                np.geomspace(
                    config.minimum_behavior_coverage,
                    config.target_action_probability,
                    config.coverage_points,
                ),
                np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]),
            )
            ),
            12,
        )
    )
    rows: list[dict[str, float]] = []
    for behavior_coverage in behavior_coverages:
        rho = 4.0 * behavior_coverage * (1.0 - behavior_coverage)
        row: dict[str, float] = {
            "behavior_coverage": float(behavior_coverage),
            "rho": float(rho),
            "effective_count": float(config.batch_size * rho),
            "g2": float(config.contribution_magnitude**2),
            "true_gradient": float(
                config.contribution_magnitude
                * (2.0 * config.positive_contribution_probability - 1.0)
            ),
            "batch_size": float(config.batch_size),
            "step_size": float(config.step_size),
        }
        for method, cap in METHODS:
            metrics = estimator_metrics(float(behavior_coverage), config, cap)
            optimization = expected_policy_value(
                float(behavior_coverage), config, cap
            )
            for key, value in metrics.items():
                row[f"{method}_{key}"] = float(value)
            for key, value in optimization.items():
                row[f"{method}_{key}"] = float(value)
        row["raw_mse_identity"] = (
            row["g2"] / row["rho"] - row["true_gradient"] ** 2
        ) / config.batch_size
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_main_figure(rows: list[dict[str, float]], output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rho = np.array([row["rho"] for row in rows])
    colors = {"raw": "#20242b", "cap_3": "#0072B2", "cap_5": "#D55E00"}
    labels = {"raw": "Unclipped", "cap_3": "Upper cap 3", "cap_5": "Upper cap 5"}
    markers = {"raw": "o", "cap_3": "s", "cap_5": "^"}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    for method in ("raw", "cap_3", "cap_5"):
        axes[0].plot(
            rho,
            [row[f"{method}_mse"] for row in rows],
            color=colors[method],
            marker=markers[method],
            markevery=3,
            markersize=3.5,
            linewidth=1.7,
            label=labels[method],
        )
        axes[1].plot(
            rho,
            [row[f"{method}_expected_policy_value"] for row in rows],
            color=colors[method],
            marker=markers[method],
            markevery=3,
            markersize=3.5,
            linewidth=1.7,
            label=labels[method],
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Gradient MSE")
    axes[0].set_xlabel("Normalized sequence ESS, $\\rho$")
    axes[0].set_ylabel("Exact MSE")
    axes[0].legend(frameon=False)

    axes[1].set_xscale("log")
    axes[1].set_title("(b) One-step policy value")
    axes[1].set_xlabel("Normalized sequence ESS, $\\rho$")
    axes[1].set_ylabel("Exact expected reward")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(True, which="major", color="#dddddd", linewidth=0.6)
        axis.grid(True, which="minor", color="#eeeeee", linewidth=0.4)
        axis.set_xlim(float(np.min(rho)) * 0.9, 1.05)

    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    config = Config()
    rows = evaluate(config)
    csv_path = args.output_dir / "contextual_bandit_results.csv"
    figure_stem = args.figure_dir / "ess_theory_validation"
    write_csv(rows, csv_path)
    if not args.skip_figure:
        make_main_figure(rows, figure_stem)

    identity_error = max(
        abs(row["raw_mse"] - row["raw_mse_identity"]) for row in rows
    )
    g2_range = max(row["g2"] for row in rows) - min(row["g2"] for row in rows)
    if identity_error > 1e-12:
        raise AssertionError("The exact MSE identity failed its numerical check.")
    if g2_range > 1e-15:
        raise AssertionError("The weighted gradient scale is not held fixed.")

    print(f"Wrote {csv_path}")
    if not args.skip_figure:
        print(f"Wrote {figure_stem.with_suffix('.pdf')}")
        print(f"Wrote {figure_stem.with_suffix('.png')}")
    print(f"Maximum raw-MSE identity error: {identity_error:.3e}")
    print(f"Range of G2 across ESS conditions: {g2_range:.3e}")


if __name__ == "__main__":
    main()
