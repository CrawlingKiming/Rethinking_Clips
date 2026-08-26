"""Theory-focused Optdigits contextual-bandit experiment.

The action space is the ten digit classes. Each policy iteration draws a fresh
on-policy rollout, performs one epoch of sequential minibatch updates, and then
discards the rollout. No ESS threshold is used for training or estimator
selection. The experiment freezes policy states spanning the observed ESS range
and uses independent redraws to measure gradient-estimator risk and one-step
population improvement.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/80/"
    "optical+recognition+of+handwritten+digits.zip"
)


@dataclass(frozen=True)
class Config:
    seed: int = 20260826
    replications: int = 12
    rollout_cycles: int = 6
    rollout_size: int = 600
    minibatches: int = 12
    training_learning_rate: float = 3.0
    diagnostic_step_size: float = 0.25
    ppo_epsilon: float = 0.2
    truncation_cap: float = 2.0
    classifier_steps: int = 400
    classifier_learning_rate: float = 0.5
    initialization_scale: float = 0.35
    checkpoints: int = 30
    redraws: int = 80
    improvement_redraws: int = 12
    redraw_batch_size: int = 128


@dataclass
class FrozenState:
    state_id: int
    trajectory: str
    replication: int
    rollout_cycle: int
    minibatch: int
    approximate_rho: float
    weights: np.ndarray
    rollout_weights: np.ndarray


def standard_error(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -50.0, 50.0))
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def load_optdigits(data_dir: Path, synthetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if synthetic:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(1200, 65))
        labels = rng.integers(0, 10, size=len(features))
        features[:, -1] = 1.0
        return features, labels

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


def fit_initial_policy(
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> np.ndarray:
    classes = 10
    weights = np.zeros((classes, features.shape[1]), dtype=float)
    one_hot = np.eye(classes)[labels]
    for _ in range(config.classifier_steps):
        probabilities = softmax(features @ weights.T)
        gradient = (probabilities - one_hot).T @ features / len(features)
        weights -= config.classifier_learning_rate * gradient
    return config.initialization_scale * weights


def population_value_and_gradient(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    probabilities = softmax(features @ weights.T)
    correct = probabilities[np.arange(len(features)), labels]
    coefficients = -correct[:, None] * probabilities
    coefficients[np.arange(len(features)), labels] += correct
    gradient = coefficients.T @ features / len(features)
    return float(np.mean(correct)), gradient


def population_value(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    probabilities = softmax(features @ weights.T)
    return float(np.mean(probabilities[np.arange(len(features)), labels]))


def population_rho(
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
) -> float:
    current = np.clip(softmax(features @ weights.T), 1e-12, 1.0)
    rollout = np.clip(softmax(features @ rollout_weights.T), 1e-12, 1.0)
    second_moment = np.mean(np.sum(current**2 / rollout, axis=1))
    return float(1.0 / second_moment)


def sample_actions(
    probabilities: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    actions = np.sum(uniforms[:, None] > cumulative, axis=1)
    return np.minimum(actions, probabilities.shape[1] - 1).astype(int)


def collect_rollout(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    context_indices: np.ndarray,
    uniforms: np.ndarray,
) -> dict[str, np.ndarray]:
    batch_features = features[context_indices]
    batch_labels = labels[context_indices]
    probabilities = softmax(batch_features @ weights.T)
    actions = sample_actions(probabilities, uniforms)
    rewards = (actions == batch_labels).astype(float)
    baseline = probabilities[np.arange(len(actions)), batch_labels]
    advantages = rewards - baseline
    action_probabilities = probabilities[np.arange(len(actions)), actions]
    return {
        "features": batch_features,
        "labels": batch_labels,
        "actions": actions,
        "advantages": advantages,
        "old_action_probabilities": action_probabilities,
    }


def score_rows(probabilities: np.ndarray, actions: np.ndarray) -> np.ndarray:
    scores = -probabilities.copy()
    scores[np.arange(len(actions)), actions] += 1.0
    return scores


def gradient_from_coefficients(
    coefficients: np.ndarray,
    scores: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return (coefficients[:, None] * scores).T @ features / len(features)


def estimate_gradients(
    weights: np.ndarray,
    rollout: dict[str, np.ndarray],
    indices: np.ndarray,
    config: Config,
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    features = rollout["features"][indices]
    actions = rollout["actions"][indices]
    advantages = rollout["advantages"][indices]
    old_action_probabilities = rollout["old_action_probabilities"][indices]
    probabilities = softmax(features @ weights.T)
    current_action_probabilities = probabilities[np.arange(len(actions)), actions]
    ratios = current_action_probabilities / np.clip(old_action_probabilities, 1e-12, None)
    scores = score_rows(probabilities, actions)

    raw_coefficients = ratios * advantages
    truncation_coefficients = np.minimum(ratios, config.truncation_cap) * advantages
    positive = advantages >= 0.0
    ppo_mask = np.where(
        positive,
        ratios <= 1.0 + config.ppo_epsilon,
        ratios >= 1.0 - config.ppo_epsilon,
    )
    ppo_coefficients = raw_coefficients * ppo_mask

    gradients = {
        "raw": gradient_from_coefficients(raw_coefficients, scores, features),
        "truncation": gradient_from_coefficients(
            truncation_coefficients, scores, features
        ),
        "ppo": gradient_from_coefficients(ppo_coefficients, scores, features),
    }
    sample_rho = float(
        np.sum(ratios) ** 2 / (len(ratios) * np.sum(ratios**2))
    )
    return gradients, sample_rho, ratios


def common_randomness(
    rng: np.random.Generator,
    population_size: int,
    config: Config,
) -> list[dict[str, np.ndarray]]:
    draws: list[dict[str, np.ndarray]] = []
    for _ in range(config.rollout_cycles):
        contexts = rng.choice(
            population_size,
            size=config.rollout_size,
            replace=False,
        )
        uniforms = rng.random(config.rollout_size)
        order = rng.permutation(config.rollout_size)
        draws.append({"contexts": contexts, "uniforms": uniforms, "order": order})
    return draws


def run_trajectory(
    trajectory: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: Config,
    replication: int,
    state_offset: int,
) -> tuple[list[FrozenState], list[dict[str, float]], float]:
    if trajectory not in {"raw", "ppo"}:
        raise ValueError(f"unsupported trajectory {trajectory}")
    weights = initial_weights.copy()
    states: list[FrozenState] = []
    path_rows: list[dict[str, float]] = []
    state_id = state_offset

    for cycle, draw in enumerate(draws, start=1):
        rollout_weights = weights.copy()
        rollout = collect_rollout(
            rollout_weights,
            features,
            labels,
            draw["contexts"],
            draw["uniforms"],
        )
        minibatches = np.split(draw["order"], config.minibatches)
        for minibatch, indices in enumerate(minibatches, start=1):
            rho = population_rho(weights, rollout_weights, features)
            states.append(
                FrozenState(
                    state_id=state_id,
                    trajectory=trajectory,
                    replication=replication,
                    rollout_cycle=cycle,
                    minibatch=minibatch,
                    approximate_rho=rho,
                    weights=weights.copy(),
                    rollout_weights=rollout_weights.copy(),
                )
            )
            state_id += 1
            gradients, sample_rho, _ = estimate_gradients(
                weights, rollout, indices, config
            )
            weights = weights + config.training_learning_rate * gradients[trajectory]
            path_rows.append(
                {
                    "replication": float(replication),
                    "trajectory": trajectory,
                    "rollout_cycle": float(cycle),
                    "minibatch": float(minibatch),
                    "population_rho": rho,
                    "sample_rho": sample_rho,
                }
            )

    final_value = population_value(weights, features, labels)
    return states, path_rows, final_value


def choose_states(states: list[FrozenState], count: int) -> list[FrozenState]:
    ordered = sorted(states, key=lambda item: item.approximate_rho)
    if count >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count)
    selected: list[FrozenState] = []
    used: set[int] = set()
    for position in positions:
        index = int(round(float(position)))
        while index in used and index + 1 < len(ordered):
            index += 1
        used.add(index)
        selected.append(ordered[index])
    return selected


def draw_fresh_batch(
    rng: np.random.Generator,
    state: FrozenState,
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    indices = rng.integers(0, len(features), size=config.redraw_batch_size)
    old_probabilities = softmax(features[indices] @ state.rollout_weights.T)
    uniforms = rng.random(config.redraw_batch_size)
    actions = sample_actions(old_probabilities, uniforms)
    rewards = (actions == labels[indices]).astype(float)
    baseline = old_probabilities[np.arange(len(indices)), labels[indices]]
    return {
        "features": features[indices],
        "labels": labels[indices],
        "actions": actions,
        "advantages": rewards - baseline,
        "old_action_probabilities": old_probabilities[
            np.arange(len(indices)), actions
        ],
    }


def evaluate_frozen_states(
    states: list[FrozenState],
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    state_rows: list[dict[str, float]] = []
    draw_rows: list[dict[str, float]] = []
    for ordinal, state in enumerate(states, start=1):
        value, true_gradient = population_value_and_gradient(
            state.weights, features, labels
        )
        rho = population_rho(state.weights, state.rollout_weights, features)
        signal_sq = float(np.sum(true_gradient**2))
        errors = {name: [] for name in ("raw", "truncation", "ppo")}
        changes = {name: [] for name in ("raw", "truncation", "ppo")}
        harm = {name: [] for name in ("raw", "truncation", "ppo")}
        sample_rhos: list[float] = []

        for redraw in range(config.redraws):
            rollout = draw_fresh_batch(rng, state, features, labels, config)
            indices = np.arange(config.redraw_batch_size)
            gradients, sample_rho, _ = estimate_gradients(
                state.weights, rollout, indices, config
            )
            sample_rhos.append(sample_rho)
            for name, gradient in gradients.items():
                error_sq = float(np.sum((gradient - true_gradient) ** 2))
                errors[name].append(error_sq)
                change = float("nan")
                if redraw < config.improvement_redraws:
                    next_value = population_value(
                        state.weights + config.diagnostic_step_size * gradient,
                        features,
                        labels,
                    )
                    change = next_value - value
                    changes[name].append(change)
                    harm[name].append(float(change < 0.0))
                draw_rows.append(
                    {
                        "state_id": float(state.state_id),
                        "state_order": float(ordinal),
                        "redraw": float(redraw),
                        "trajectory": state.trajectory,
                        "population_rho": rho,
                        "sample_rho": sample_rho,
                        "signal_sq": signal_sq,
                        "estimator": name,
                        "error_sq": error_sq,
                        "relative_error": error_sq / max(signal_sq, 1e-16),
                        "reward_change": change,
                    }
                )

        oracle_value = population_value(
            state.weights + config.diagnostic_step_size * true_gradient,
            features,
            labels,
        )
        row: dict[str, float] = {
            "state_id": float(state.state_id),
            "state_order": float(ordinal),
            "replication": float(state.replication),
            "rollout_cycle": float(state.rollout_cycle),
            "minibatch": float(state.minibatch),
            "trajectory": state.trajectory,
            "population_rho": rho,
            "mean_sample_rho": float(np.mean(sample_rhos)),
            "signal_sq": signal_sq,
            "current_value": value,
            "oracle_change": oracle_value - value,
        }
        for name in errors:
            array = np.asarray(errors[name], dtype=float)
            row[f"{name}_mse"] = float(np.mean(array))
            row[f"{name}_mse_se"] = standard_error(array)
            row[f"{name}_relative_mse"] = float(
                np.mean(array) / max(signal_sq, 1e-16)
            )
            row[f"{name}_mean_change"] = float(np.mean(changes[name]))
            row[f"{name}_harm_rate"] = float(np.mean(harm[name]))
        state_rows.append(row)
    return state_rows, draw_rows


def quantile_bin_rows(state_rows: list[dict[str, float]], bins: int = 6) -> list[dict[str, float]]:
    rho = np.asarray([row["population_rho"] for row in state_rows], dtype=float)
    edges = np.unique(np.quantile(rho, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(float(np.min(rho)), float(np.max(rho)) + 1e-8, 3)
    assignments = np.digitize(rho, edges[1:-1], right=True)
    output: list[dict[str, float]] = []
    for bin_index in range(len(edges) - 1):
        indices = np.where(assignments == bin_index)[0]
        if not len(indices):
            continue
        selected = [state_rows[index] for index in indices]
        row: dict[str, float] = {
            "bin": float(bin_index),
            "rho_left": float(edges[bin_index]),
            "rho_right": float(edges[bin_index + 1]),
            "rho_median": float(np.median(rho[indices])),
            "states": float(len(indices)),
        }
        oracle = np.asarray([item["oracle_change"] for item in selected])
        row["oracle_mean_change"] = float(np.mean(oracle))
        row["oracle_change_se"] = standard_error(oracle)
        for name in ("raw", "truncation", "ppo"):
            values = np.asarray([item[f"{name}_mse"] for item in selected])
            row[f"{name}_mse"] = float(np.mean(values))
            row[f"{name}_mse_se"] = standard_error(values)
            changes = np.asarray([item[f"{name}_mean_change"] for item in selected])
            harms = np.asarray([item[f"{name}_harm_rate"] for item in selected])
            row[f"{name}_mean_change"] = float(np.mean(changes))
            row[f"{name}_change_se"] = standard_error(changes)
            row[f"{name}_harm_rate"] = float(np.mean(harms))
        output.append(row)
    return output


def relative_error_bin_rows(draw_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    raw = [row for row in draw_rows if row["estimator"] == "raw" and np.isfinite(row["reward_change"])]
    boundaries = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])
    ratios = np.asarray([row["relative_error"] for row in raw])
    assignments = np.digitize(ratios, boundaries[1:-1], right=False)
    output: list[dict[str, float]] = []
    for index in range(len(boundaries) - 1):
        selected = [raw[j] for j in np.where(assignments == index)[0]]
        if not selected:
            continue
        changes = np.asarray([row["reward_change"] for row in selected])
        output.append(
            {
                "bin": float(index),
                "relative_error_left": float(boundaries[index]),
                "relative_error_right": float(boundaries[index + 1]),
                "relative_error_median": float(
                    np.median([row["relative_error"] for row in selected])
                ),
                "count": float(len(selected)),
                "mean_change": float(np.mean(changes)),
                "change_se": standard_error(changes),
                "harm_rate": float(np.mean(changes < 0.0)),
            }
        )
    return output


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        raise RuntimeError(f"no rows for {path}")
    fieldnames = list(materialized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def make_figure(
    state_rows: list[dict[str, float]],
    bin_rows: list[dict[str, float]],
    error_rows: list[dict[str, float]],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    rho = np.asarray([row["population_rho"] for row in state_rows])
    raw_mse = np.asarray([row["raw_mse"] for row in state_rows])
    ax.scatter(rho, raw_mse, s=22, alpha=0.45)
    x = np.asarray([row["rho_median"] for row in bin_rows])
    y = np.asarray([row["raw_mse"] for row in bin_rows])
    yerr = np.asarray([row["raw_mse_se"] for row in bin_rows])
    order = np.argsort(x)
    ax.errorbar(x[order], y[order], yerr=yerr[order], marker="o", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Unmodified gradient MSE")
    ax.set_title("ESS and estimator error")

    ax = axes[1]
    x = np.asarray([row["relative_error_median"] for row in error_rows])
    harm = np.asarray([row["harm_rate"] for row in error_rows])
    ax.plot(x, harm, marker="o")
    ax.axvline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(-0.01, max(0.2, float(np.max(harm)) + 0.04))
    ax.set_xlabel(r"Realized squared error / $\|g\|_2^2$")
    ax.set_ylabel("Probability of negative change")
    ax.set_title("Error relative to gradient signal")

    ax = axes[2]
    raw_change = np.asarray([row["raw_mean_change"] for row in bin_rows])
    raw_se = np.asarray([row["raw_change_se"] for row in bin_rows])
    oracle_change = np.asarray([row["oracle_mean_change"] for row in bin_rows])
    oracle_se = np.asarray([row["oracle_change_se"] for row in bin_rows])
    ax.errorbar(x=np.asarray([row["rho_median"] for row in bin_rows])[order],
                y=raw_change[order], yerr=raw_se[order], marker="o",
                capsize=3, label="Sampled gradient")
    ax.errorbar(x=np.asarray([row["rho_median"] for row in bin_rows])[order],
                y=oracle_change[order], yerr=oracle_se[order], marker="s",
                capsize=3, label="Population gradient")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("Sampled and oracle updates")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def aggregate_final_values(final_values: dict[str, list[float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for trajectory, values in final_values.items():
        array = np.asarray(values)
        rows.append(
            {
                "trajectory": trajectory,
                "replications": float(len(array)),
                "mean_final_value": float(np.mean(array)),
                "se_final_value": standard_error(array),
                "median_final_value": float(np.median(array)),
            }
        )
    return rows


def correlation_summary(
    state_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
    final_rows: list[dict[str, float]],
) -> str:
    rho = np.asarray([row["population_rho"] for row in state_rows])
    raw_mse = np.asarray([row["raw_mse"] for row in state_rows])
    positive = rho > 0.0
    log_corr = float(
        np.corrcoef(np.log(rho[positive]), np.log(raw_mse[positive]))[0, 1]
    )
    raw_draws = [
        row
        for row in draw_rows
        if row["estimator"] == "raw" and np.isfinite(row["reward_change"])
    ]
    below = [row for row in raw_draws if row["relative_error"] < 1.0]
    above = [row for row in raw_draws if row["relative_error"] >= 1.0]
    below_harm = float(np.mean([row["reward_change"] < 0.0 for row in below]))
    above_harm = float(np.mean([row["reward_change"] < 0.0 for row in above]))
    oracle_negative = float(np.mean([row["oracle_change"] < 0.0 for row in state_rows]))

    lowest = min(state_rows, key=lambda row: row["population_rho"])
    highest = max(state_rows, key=lambda row: row["population_rho"])
    lines = [
        f"states={len(state_rows)}",
        f"population_rho_min={np.min(rho):.10f}",
        f"population_rho_median={np.median(rho):.6f}",
        f"population_rho_max={np.max(rho):.6f}",
        f"corr_log_rho_log_raw_mse={log_corr:.6f}",
        f"lowest_rho_raw_mse={lowest['raw_mse']:.8f}",
        f"highest_rho_raw_mse={highest['raw_mse']:.8f}",
        f"low_to_high_mse_ratio={lowest['raw_mse']/highest['raw_mse']:.6f}",
        f"relative_error_below_one_count={len(below)}",
        f"relative_error_below_one_harm_rate={below_harm:.6f}",
        f"relative_error_above_one_count={len(above)}",
        f"relative_error_above_one_harm_rate={above_harm:.6f}",
        f"oracle_negative_rate={oracle_negative:.6f}",
    ]
    for row in final_rows:
        lines.append(
            f"final_value_{row['trajectory']}_mean={row['mean_final_value']:.8f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(replications=args.replications)
    root = Path(__file__).resolve().parents[1]
    features, labels = load_optdigits(root / "simulation" / "data", args.synthetic)
    initial_weights = fit_initial_policy(features, labels, config)
    initial_value = population_value(initial_weights, features, labels)

    all_states: list[FrozenState] = []
    path_rows: list[dict[str, float]] = []
    final_values = {"raw": [], "ppo": []}
    next_state_id = 0

    for replication in range(config.replications):
        rng = np.random.default_rng(config.seed + replication)
        draws = common_randomness(rng, len(features), config)
        for trajectory in ("raw", "ppo"):
            states, rows, final_value = run_trajectory(
                trajectory,
                initial_weights,
                features,
                labels,
                draws,
                config,
                replication,
                next_state_id,
            )
            next_state_id += len(states)
            all_states.extend(states)
            path_rows.extend(rows)
            final_values[trajectory].append(final_value)

    selected_states = choose_states(all_states, config.checkpoints)
    diagnostic_rng = np.random.default_rng(config.seed + 100000)
    state_rows, draw_rows = evaluate_frozen_states(
        selected_states, features, labels, config, diagnostic_rng
    )
    bin_rows = quantile_bin_rows(state_rows)
    error_rows = relative_error_bin_rows(draw_rows)
    final_rows = aggregate_final_values(final_values)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_categorical_paths.csv", path_rows)
    write_csv(result_dir / "optdigits_categorical_frozen_states.csv", state_rows)
    write_csv(result_dir / "optdigits_categorical_redraws.csv", draw_rows)
    write_csv(result_dir / "optdigits_categorical_ess_bins.csv", bin_rows)
    write_csv(result_dir / "optdigits_categorical_error_bins.csv", error_rows)
    write_csv(result_dir / "optdigits_categorical_final_values.csv", final_rows)

    summary = f"initial_value={initial_value:.8f}\n" + correlation_summary(
        state_rows, draw_rows, final_rows
    )
    (result_dir / "optdigits_categorical_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    make_figure(
        state_rows,
        bin_rows,
        error_rows,
        root / "figures" / "optdigits_categorical_theory",
    )
    print(summary)


if __name__ == "__main__":
    main()
