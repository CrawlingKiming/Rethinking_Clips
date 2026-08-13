"""Literature-grounded contextual-bandit test of ESS and gradient MSE.

The testbed follows the classification-to-bandit reduction used by Wang,
Agarwal, and Dudik (ICML 2017): a labeled example is a context, class labels
are actions, and the reward is one if the selected label is correct.  A softmax
classifier defines the target policy.  We construct logging policies by
temperature-softening either the target policy or a deliberately shifted
classifier.  This varies coverage and the association between large weights and
large gradient contributions.

The empirical targets are the paper's actual theoretical quantities:

1. Does the raw-gradient MSE follow (G2 / rho - ||g||^2) / N?
2. Does ESS alone fail when G2 changes?
3. Does lower ESS make a data-calibrated clipping estimator more likely to beat
   the raw estimator in MSE?

All reported Monte Carlo summaries use at most 100 independent batches.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/80/"
    "optical+recognition+of+handwritten+digits.zip"
)
CAP_GRID = np.array([1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, np.inf])


@dataclass(frozen=True)
class Config:
    seed: int = 19
    repetitions: int = 100
    batch_size: int = 256
    training_steps: int = 500
    learning_rate: float = 0.8
    target_temperature: float = 0.75
    behavior_temperatures: tuple[float, ...] = (0.75, 1.0, 1.5, 2.5, 4.0)
    exploration_mass: float = 0.05
    validation_fraction: float = 0.4


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def load_optdigits(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = data_dir / "optdigits.zip"
    extracted = data_dir / "optdigits"
    training = extracted / "optdigits.tra"
    test = extracted / "optdigits.tes"
    if not training.exists() or not test.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            urllib.request.urlretrieve(DATA_URL, archive)
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)

    arrays = [np.loadtxt(path, delimiter=",") for path in (training, test)]
    combined = np.vstack(arrays)
    features = combined[:, :-1] / 16.0
    labels = combined[:, -1].astype(int)
    features = np.column_stack([features, np.ones(features.shape[0])])
    return features, labels


def fit_softmax_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    classes = int(np.max(labels)) + 1
    weights = np.zeros((classes, features.shape[1]))
    one_hot = np.eye(classes)[labels]
    if sample_weights is None:
        sample_weights = np.ones(features.shape[0])
    sample_weights = sample_weights / np.mean(sample_weights)
    for _ in range(config.training_steps):
        probabilities = softmax(features @ weights.T)
        gradient = (
            (sample_weights[:, None] * (probabilities - one_hot)).T @ features
            / features.shape[0]
        )
        weights -= config.learning_rate * gradient
    return weights


def population_quantities(
    features: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    behavior: np.ndarray,
    cap: float = np.inf,
) -> dict[str, float | np.ndarray]:
    samples, actions = target.shape
    rewards = np.eye(actions)[labels]
    baseline = np.sum(target * rewards, axis=1)
    advantage = rewards - baseline[:, None]
    ratios = target / behavior
    coefficients = np.minimum(ratios, cap)
    joint = behavior / samples

    factors = joint * coefficients * advantage
    class_factors = factors - np.sum(factors, axis=1, keepdims=True) * target
    mean = class_factors.T @ features

    feature_norm_squared = np.sum(features**2, axis=1)
    probability_norm_squared = np.sum(target**2, axis=1, keepdims=True)
    score_norm_squared = (
        1.0 - 2.0 * target + probability_norm_squared
    ) * feature_norm_squared[:, None]
    contribution_norm_squared = advantage**2 * score_norm_squared
    second_moment = float(
        np.sum(joint * coefficients**2 * contribution_norm_squared)
    )
    variance_trace = second_moment - float(np.sum(mean**2))

    raw_second = float(np.sum(joint * ratios**2 * contribution_norm_squared))
    ew2 = float(np.sum(joint * ratios**2))
    return {
        "mean": mean,
        "variance_trace": variance_trace,
        "rho": 1.0 / ew2,
        "g2": raw_second / ew2,
        "raw_second": raw_second,
    }


def sample_batch_errors(
    rng: np.random.Generator,
    features: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    behavior: np.ndarray,
    true_gradient: np.ndarray,
    batch_size: int,
    caps: np.ndarray,
) -> np.ndarray:
    samples, actions = target.shape
    contexts = rng.integers(0, samples, size=batch_size)
    uniforms = rng.random(batch_size)
    cumulative = np.cumsum(behavior[contexts], axis=1)
    selected = np.sum(uniforms[:, None] > cumulative, axis=1)

    selected_target = target[contexts, selected]
    selected_behavior = behavior[contexts, selected]
    ratios = selected_target / selected_behavior
    rewards = (selected == labels[contexts]).astype(float)
    baseline = target[contexts, labels[contexts]]
    advantage = rewards - baseline

    action_residual = -target[contexts].copy()
    action_residual[np.arange(batch_size), selected] += 1.0
    score = action_residual[:, :, None] * features[contexts, None, :]
    contribution = advantage[:, None, None] * score

    errors = np.empty(len(caps))
    for index, cap in enumerate(caps):
        estimate = np.mean(
            np.minimum(ratios, cap)[:, None, None] * contribution,
            axis=0,
        )
        errors[index] = float(np.sum((estimate - true_gradient) ** 2))
    return errors


def make_behavior_policies(
    features: np.ndarray,
    target_weights: np.ndarray,
    shifted_weights: np.ndarray,
    target: np.ndarray,
    config: Config,
) -> list[tuple[str, float, np.ndarray]]:
    actions = target.shape[1]
    uniform = np.full_like(target, 1.0 / actions)
    policies: list[tuple[str, float, np.ndarray]] = []
    for temperature in config.behavior_temperatures:
        aligned = softmax(features @ target_weights.T, temperature)
        shifted = softmax(features @ shifted_weights.T, temperature)
        policies.append(
            (
                "aligned",
                temperature,
                (1.0 - config.exploration_mass) * aligned
                + config.exploration_mass * uniform,
            )
        )
        policies.append(
            (
                "shifted",
                temperature,
                (1.0 - config.exploration_mass) * shifted
                + config.exploration_mass * uniform,
            )
        )
    policies.append(("target", config.target_temperature, target.copy()))
    return policies


def evaluate(config: Config, data_dir: Path) -> list[dict[str, float | str]]:
    features, labels = load_optdigits(data_dir)
    classifier = fit_softmax_classifier(features, labels, config)
    shift_rng = np.random.default_rng(config.seed + 1)
    projection = features[:, :-1] @ shift_rng.normal(size=features.shape[1] - 1)
    projection = (projection - np.mean(projection)) / np.std(projection)
    shift_weights = 0.1 + 0.9 / (1.0 + np.exp(-2.0 * projection))
    shifted_classifier = fit_softmax_classifier(
        features, labels, config, sample_weights=shift_weights
    )
    target = softmax(features @ classifier.T, config.target_temperature)
    target_reference = population_quantities(features, labels, target, target)
    true_gradient = target_reference["mean"]

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | str]] = []
    validation_repetitions = int(config.repetitions * config.validation_fraction)
    for family, temperature, behavior in make_behavior_policies(
        features, classifier, shifted_classifier, target, config
    ):
        raw = population_quantities(features, labels, target, behavior)
        errors = np.stack(
            [
                sample_batch_errors(
                    rng,
                    features,
                    labels,
                    target,
                    behavior,
                    true_gradient,
                    config.batch_size,
                    CAP_GRID,
                )
                for _ in range(config.repetitions)
            ]
        )
        validation_mse = np.mean(errors[:validation_repetitions], axis=0)
        selected_index = int(np.argmin(validation_mse))
        selected_cap = CAP_GRID[selected_index]
        test_errors = errors[validation_repetitions:]
        raw_test = test_errors[:, -1]
        selected_test = test_errors[:, selected_index]

        raw_theory_mse = (
            float(raw["g2"]) / float(raw["rho"])
            - float(np.sum(true_gradient**2))
        ) / config.batch_size
        selected_population = population_quantities(
            features, labels, target, behavior, selected_cap
        )
        selected_bias_squared = float(
            np.sum((selected_population["mean"] - true_gradient) ** 2)
        )
        selected_theory_mse = (
            selected_bias_squared
            + float(selected_population["variance_trace"]) / config.batch_size
        )

        rows.append(
            {
                "family": family,
                "temperature": temperature,
                "rho": float(raw["rho"]),
                "effective_count": config.batch_size * float(raw["rho"]),
                "g2": float(raw["g2"]),
                "raw_theory_mse": raw_theory_mse,
                "raw_empirical_mse": float(np.mean(raw_test)),
                "raw_empirical_se": float(
                    np.std(raw_test, ddof=1) / math.sqrt(len(raw_test))
                ),
                "selected_cap": float(selected_cap),
                "selected_theory_mse": selected_theory_mse,
                "selected_empirical_mse": float(np.mean(selected_test)),
                "selected_empirical_se": float(
                    np.std(selected_test, ddof=1) / math.sqrt(len(selected_test))
                ),
                "clipping_mse_ratio": float(
                    np.mean(selected_test) / np.mean(raw_test)
                ),
                "repetitions": config.repetitions,
                "validation_repetitions": validation_repetitions,
                "test_repetitions": config.repetitions - validation_repetitions,
                "batch_size": config.batch_size,
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, float | str]], output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(9.1, 2.9), constrained_layout=True)
    styles = {
        "aligned": ("#0072B2", "o", "Aligned logger"),
        "shifted": ("#D55E00", "s", "Shifted logger"),
        "target": ("#20242b", "^", "Target logger"),
    }
    for family, (color, marker, label) in styles.items():
        selected = [row for row in rows if row["family"] == family]
        selected.sort(key=lambda row: float(row["rho"]))
        if not selected:
            continue
        rho = np.array([float(row["rho"]) for row in selected])
        g2 = np.array([float(row["g2"]) for row in selected])
        empirical = np.array([float(row["raw_empirical_mse"]) for row in selected])
        predicted = np.array([float(row["raw_theory_mse"]) for row in selected])
        ratio = np.array([float(row["clipping_mse_ratio"]) for row in selected])
        axes[0].scatter(predicted, empirical, color=color, marker=marker, label=label)
        axes[1].plot(rho, g2, color=color, marker=marker, label=label)
        axes[2].plot(rho, ratio, color=color, marker=marker, label=label)

    all_predicted = np.array([float(row["raw_theory_mse"]) for row in rows])
    lower = float(np.min(all_predicted)) * 0.85
    upper = float(np.max(all_predicted)) * 1.15
    axes[0].plot([lower, upper], [lower, upper], color="#777777", linestyle="--")
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_title("(a) Theorem prediction")
    axes[0].set_xlabel("Predicted raw MSE")
    axes[0].set_ylabel("Empirical raw MSE")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_title("(b) Why ESS is not enough")
    axes[1].set_ylabel("$G_2$")
    axes[1].set_yscale("log")
    axes[2].set_title("(c) Clipping crossover")
    axes[2].set_ylabel("Selected / raw MSE")
    axes[2].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    for axis in axes[1:]:
        axis.set_xlabel("Normalized ESS, $\\rho$")
        axis.set_xscale("log")
    for axis in axes:
        axis.grid(True, which="both", color="#e6e6e6", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
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

    config = Config(repetitions=args.repetitions)
    rows = evaluate(config, args.data_dir)
    csv_path = args.output_dir / "contextual_bandit_results.csv"
    figure_stem = args.figure_dir / "ess_theory_validation"
    write_csv(rows, csv_path)
    if not args.skip_figure:
        make_figure(rows, figure_stem)

    correlation = np.corrcoef(
        np.log([float(row["raw_empirical_mse"]) for row in rows]),
        np.log(
            [
                float(row["g2"]) / float(row["rho"])
                for row in rows
            ]
        ),
    )[0, 1]
    relative_identity_error = max(
        abs(float(row["raw_empirical_mse"]) - float(row["raw_theory_mse"]))
        / float(row["raw_theory_mse"])
        for row in rows
    )
    print(f"Wrote {csv_path}")
    if not args.skip_figure:
        print(f"Wrote {figure_stem.with_suffix('.pdf')}")
        print(f"Wrote {figure_stem.with_suffix('.png')}")
    print(f"Repetitions per condition: {config.repetitions}")
    print(f"Correlation log(MSE), log(G2/rho): {correlation:.4f}")
    print(f"Maximum empirical/theory relative gap: {relative_identity_error:.3f}")


if __name__ == "__main__":
    main()
