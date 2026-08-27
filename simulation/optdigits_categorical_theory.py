"""Optdigits contextual-bandit experiments for theory validation and update selection.

The action space is the ten digit classes. Each policy iteration draws a fresh
on-policy rollout, performs one epoch of sequential minibatch updates, and then
discards the rollout. The first experiment freezes policy states and measures
gradient error and one-step population change. The second experiment compares
static raw and PPO updates with an MSE oracle and several sample-ESS gates.
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
ESS_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
RAW_COLOR = "#3B6FB6"
PPO_COLOR = "#E07A1F"
ORACLE_COLOR = "#2A9D8F"
ESS_COLOR = "#7A5195"
HARM_COLOR = "#D1495B"
NEUTRAL_COLOR = "#667085"
LIGHT_GRID = "#D9DEE8"


@dataclass(frozen=True)
class Config:
    seed: int = 20260826
    replications: int = 40
    rollout_cycles: int = 6
    rollout_size: int = 600
    minibatches: int = 12
    training_learning_rate: float = 2.0
    diagnostic_step_size: float = 0.25
    ppo_epsilon: float = 0.2
    classifier_steps: int = 400
    classifier_learning_rate: float = 0.5
    initialization_scale: float = 0.35
    checkpoints: int = 30
    redraws: int = 80
    improvement_redraws: int = 12
    redraw_batch_size: int = 128

    @property
    def minibatch_size(self) -> int:
        if self.rollout_size % self.minibatches != 0:
            raise ValueError("rollout_size must be divisible by minibatches")
        return self.rollout_size // self.minibatches


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


def load_optdigits(
    data_dir: Path,
    synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
    batch_features = rollout["features"][indices]
    actions = rollout["actions"][indices]
    advantages = rollout["advantages"][indices]
    old_action_probabilities = rollout["old_action_probabilities"][indices]
    probabilities = softmax(batch_features @ weights.T)
    current_action_probabilities = probabilities[np.arange(len(actions)), actions]
    ratios = current_action_probabilities / np.clip(
        old_action_probabilities,
        1e-12,
        None,
    )
    scores = score_rows(probabilities, actions)

    raw_coefficients = ratios * advantages
    positive = advantages >= 0.0
    ppo_mask = np.where(
        positive,
        ratios <= 1.0 + config.ppo_epsilon,
        ratios >= 1.0 - config.ppo_epsilon,
    )
    ppo_coefficients = raw_coefficients * ppo_mask

    gradients = {
        "raw": gradient_from_coefficients(raw_coefficients, scores, batch_features),
        "ppo": gradient_from_coefficients(ppo_coefficients, scores, batch_features),
    }
    denominator = len(ratios) * np.sum(ratios**2)
    sample_rho = float(np.sum(ratios) ** 2 / denominator) if denominator > 0 else 0.0
    return gradients, sample_rho, ratios


def exact_estimator_risks(
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    config: Config,
) -> dict[str, float]:
    """Return exact iid minibatch MSE for raw and PPO gradient estimators."""
    current = np.clip(softmax(features @ weights.T), 1e-12, 1.0)
    rollout = np.clip(softmax(features @ rollout_weights.T), 1e-12, 1.0)
    population_size, classes = current.shape
    correct_current = current[np.arange(population_size), labels]
    target_coefficients = -correct_current[:, None] * current
    target_coefficients[np.arange(population_size), labels] += correct_current
    true_gradient = target_coefficients.T @ features / population_size

    correct_rollout = rollout[np.arange(population_size), labels]
    one_hot = np.eye(classes)[labels]
    advantages = one_hot - correct_rollout[:, None]
    ratios = current / rollout
    raw_coefficients = ratios * advantages
    positive = advantages >= 0.0
    ppo_mask = np.where(
        positive,
        ratios <= 1.0 + config.ppo_epsilon,
        ratios >= 1.0 - config.ppo_epsilon,
    )
    ppo_coefficients = raw_coefficients * ppo_mask

    feature_norm_sq = np.sum(features**2, axis=1)
    probability_norm_sq = np.sum(current**2, axis=1, keepdims=True)
    score_norm_sq = feature_norm_sq[:, None] * (
        1.0 - 2.0 * current + probability_norm_sq
    )

    risks: dict[str, float] = {}
    for name, coefficients in (
        ("raw", raw_coefficients),
        ("ppo", ppo_coefficients),
    ):
        weighted_coefficients = rollout * coefficients
        adjusted = weighted_coefficients - current * np.sum(
            weighted_coefficients,
            axis=1,
            keepdims=True,
        )
        mean_gradient = adjusted.T @ features / population_size
        second_moment = float(
            np.mean(
                np.sum(
                    rollout * coefficients**2 * score_norm_sq,
                    axis=1,
                )
            )
        )
        mean_norm_sq = float(np.sum(mean_gradient**2))
        variance_trace = max(second_moment - mean_norm_sq, 0.0)
        bias_sq = float(np.sum((mean_gradient - true_gradient) ** 2))
        risks[f"{name}_bias_sq"] = bias_sq
        risks[f"{name}_variance"] = variance_trace
        risks[f"{name}_risk"] = bias_sq + variance_trace / batch_size
    return risks


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
            replace=True,
        )
        uniforms = rng.random(config.rollout_size)
        order = rng.permutation(config.rollout_size)
        draws.append({"contexts": contexts, "uniforms": uniforms, "order": order})
    return draws


def method_name(threshold: float) -> str:
    return f"ess_{threshold:.2f}"


def run_trajectory(
    method: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: Config,
    replication: int,
    state_offset: int,
    threshold: float | None = None,
    collect_states: bool = False,
) -> tuple[list[FrozenState], list[dict[str, float]], dict[str, float]]:
    valid = {"raw", "ppo", "mse_oracle", "ess"}
    if method not in valid:
        raise ValueError(f"unsupported method {method}")
    if method == "ess" and threshold is None:
        raise ValueError("ESS method requires a threshold")

    label = method_name(threshold) if threshold is not None else method
    weights = initial_weights.copy()
    states: list[FrozenState] = []
    path_rows: list[dict[str, float]] = []
    state_id = state_offset
    update = 0
    ppo_updates = 0
    initial_value = population_value(weights, features, labels)
    path_rows.append(
        {
            "replication": float(replication),
            "method": label,
            "threshold": float(threshold) if threshold is not None else float("nan"),
            "update": 0.0,
            "rollout_cycle": 0.0,
            "minibatch": 0.0,
            "population_value": initial_value,
            "population_rho": 1.0,
            "sample_rho": 1.0,
            "chosen_ppo": 0.0,
            "raw_risk": float("nan"),
            "ppo_risk": float("nan"),
        }
    )

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
            if collect_states:
                states.append(
                    FrozenState(
                        state_id=state_id,
                        trajectory=label,
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
                weights,
                rollout,
                indices,
                config,
            )
            raw_risk = float("nan")
            ppo_risk = float("nan")
            if method == "raw":
                selected = "raw"
            elif method == "ppo":
                selected = "ppo"
            elif method == "ess":
                selected = "ppo" if sample_rho < float(threshold) else "raw"
            else:
                risks = exact_estimator_risks(
                    weights,
                    rollout_weights,
                    features,
                    labels,
                    len(indices),
                    config,
                )
                raw_risk = risks["raw_risk"]
                ppo_risk = risks["ppo_risk"]
                selected = "ppo" if ppo_risk < raw_risk else "raw"

            weights = weights + config.training_learning_rate * gradients[selected]
            update += 1
            ppo_updates += int(selected == "ppo")
            path_rows.append(
                {
                    "replication": float(replication),
                    "method": label,
                    "threshold": float(threshold) if threshold is not None else float("nan"),
                    "update": float(update),
                    "rollout_cycle": float(cycle),
                    "minibatch": float(minibatch),
                    "population_value": population_value(weights, features, labels),
                    "population_rho": rho,
                    "sample_rho": sample_rho,
                    "chosen_ppo": float(selected == "ppo"),
                    "raw_risk": raw_risk,
                    "ppo_risk": ppo_risk,
                }
            )

    final_value = population_value(weights, features, labels)
    summary = {
        "replication": float(replication),
        "method": label,
        "threshold": float(threshold) if threshold is not None else float("nan"),
        "final_value": final_value,
        "ppo_fraction": ppo_updates / max(update, 1),
    }
    return states, path_rows, summary


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
            np.arange(len(indices)),
            actions,
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
            state.weights,
            features,
            labels,
        )
        rho = population_rho(state.weights, state.rollout_weights, features)
        signal_sq = float(np.sum(true_gradient**2))
        exact_risks = exact_estimator_risks(
            state.weights,
            state.rollout_weights,
            features,
            labels,
            config.redraw_batch_size,
            config,
        )
        oracle_ppo = float(exact_risks["ppo_risk"] < exact_risks["raw_risk"])
        errors = {name: [] for name in ("raw", "ppo")}
        changes = {name: [] for name in ("raw", "ppo")}
        harm = {name: [] for name in ("raw", "ppo")}
        sample_rhos: list[float] = []

        for redraw in range(config.redraws):
            rollout = draw_fresh_batch(rng, state, features, labels, config)
            indices = np.arange(config.redraw_batch_size)
            gradients, sample_rho, _ = estimate_gradients(
                state.weights,
                rollout,
                indices,
                config,
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
                        "oracle_ppo": oracle_ppo,
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
            "exact_raw_risk": exact_risks["raw_risk"],
            "exact_ppo_risk": exact_risks["ppo_risk"],
            "oracle_ppo": oracle_ppo,
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


def quantile_bin_rows(
    state_rows: list[dict[str, float]],
    bins: int = 6,
) -> list[dict[str, float]]:
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
        for name in ("raw", "ppo"):
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


def relative_error_bin_rows(
    draw_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    raw = [
        row
        for row in draw_rows
        if row["estimator"] == "raw" and np.isfinite(row["reward_change"])
    ]
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


def aggregate_final_values(
    run_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    methods = sorted({str(row["method"]) for row in run_rows})
    output: list[dict[str, float]] = []
    for method in methods:
        selected = [row for row in run_rows if row["method"] == method]
        values = np.asarray([row["final_value"] for row in selected], dtype=float)
        fractions = np.asarray([row["ppo_fraction"] for row in selected], dtype=float)
        threshold_values = [
            row["threshold"] for row in selected if np.isfinite(row["threshold"])
        ]
        output.append(
            {
                "method": method,
                "threshold": (
                    float(threshold_values[0]) if threshold_values else float("nan")
                ),
                "replications": float(len(values)),
                "mean_final_value": float(np.mean(values)),
                "se_final_value": standard_error(values),
                "median_final_value": float(np.median(values)),
                "mean_ppo_fraction": float(np.mean(fractions)),
                "se_ppo_fraction": standard_error(fractions),
            }
        )
    return output


def threshold_summary_rows(
    final_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    raw_draws = [row for row in draw_rows if row["estimator"] == "raw"]
    by_method = {str(row["method"]): row for row in final_rows}
    output: list[dict[str, float]] = []
    for threshold in ESS_THRESHOLDS:
        method = method_name(threshold)
        final = by_method[method]
        agreements = [
            float((row["sample_rho"] < threshold) == bool(row["oracle_ppo"]))
            for row in raw_draws
        ]
        output.append(
            {
                "threshold": threshold,
                "mean_final_value": final["mean_final_value"],
                "se_final_value": final["se_final_value"],
                "mean_ppo_fraction": final["mean_ppo_fraction"],
                "se_ppo_fraction": final["se_ppo_fraction"],
                "frozen_redraw_oracle_agreement": float(np.mean(agreements)),
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


def set_plot_defaults() -> None:
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


def make_theory_figure(
    bin_rows: list[dict[str, float]],
    error_rows: list[dict[str, float]],
    figure_path: Path,
) -> None:
    set_plot_defaults()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    rho = np.asarray([row["rho_median"] for row in bin_rows])
    order = np.argsort(rho)

    ax = axes[0]
    raw_mse = np.asarray([row["raw_mse"] for row in bin_rows])
    raw_se = np.asarray([row["raw_mse_se"] for row in bin_rows])
    ax.errorbar(
        rho[order],
        raw_mse[order],
        yerr=raw_se[order],
        color=RAW_COLOR,
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Unmodified gradient MSE")
    ax.set_title("Effective support and gradient error")

    ax = axes[1]
    labels = []
    harms = []
    for row in error_rows:
        left = row["relative_error_left"]
        right = row["relative_error_right"]
        labels.append(f"{left:g}+" if np.isinf(right) else f"{left:g}-{right:g}")
        harms.append(row["harm_rate"])
    positions = np.arange(len(labels))
    ax.bar(positions, harms, color=HARM_COLOR, alpha=0.88, width=0.72)
    ax.axvline(2.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.1)
    ax.set_xticks(positions, labels, rotation=28, ha="right")
    ax.set_ylim(0.0, max(0.18, max(harms) * 1.22))
    ax.set_xlabel(r"Realized squared error / $\|g\|_2^2$")
    ax.set_ylabel("Harmful-update rate")
    ax.set_title("Failure after error reaches signal scale")

    ax = axes[2]
    raw_change = np.asarray([row["raw_mean_change"] for row in bin_rows])
    raw_change_se = np.asarray([row["raw_change_se"] for row in bin_rows])
    oracle_change = np.asarray([row["oracle_mean_change"] for row in bin_rows])
    oracle_change_se = np.asarray([row["oracle_change_se"] for row in bin_rows])
    ax.errorbar(
        rho[order],
        raw_change[order],
        yerr=raw_change_se[order],
        color=RAW_COLOR,
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="Sampled gradient",
    )
    ax.errorbar(
        rho[order],
        oracle_change[order],
        yerr=oracle_change_se[order],
        color=ORACLE_COLOR,
        marker="s",
        linewidth=2.0,
        capsize=3,
        label="Population gradient",
    )
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("Estimation failure, not absence of direction")
    ax.legend(frameon=False)

    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def curve_statistics(
    path_rows: list[dict[str, float]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in path_rows if row["method"] == method]
    updates = np.asarray(sorted({int(row["update"]) for row in selected}))
    means = []
    ses = []
    for update in updates:
        values = np.asarray(
            [
                row["population_value"]
                for row in selected
                if int(row["update"]) == update
            ]
        )
        means.append(float(np.mean(values)))
        ses.append(standard_error(values))
    return updates, np.asarray(means), np.asarray(ses)


def make_control_figure(
    path_rows: list[dict[str, float]],
    final_rows: list[dict[str, float]],
    threshold_rows: list[dict[str, float]],
    figure_path: Path,
) -> str:
    set_plot_defaults()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    by_method = {str(row["method"]): row for row in final_rows}
    best_threshold_row = max(threshold_rows, key=lambda row: row["mean_final_value"])
    best_method = method_name(best_threshold_row["threshold"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    ax = axes[0]
    curve_specs = [
        ("raw", "Unmodified", RAW_COLOR, "-"),
        ("ppo", "PPO", PPO_COLOR, "-"),
        ("mse_oracle", "MSE oracle", ORACLE_COLOR, "-"),
        (
            best_method,
            rf"ESS gate, $\tau={best_threshold_row['threshold']:.2g}$",
            ESS_COLOR,
            "--",
        ),
    ]
    for method, label, color, linestyle in curve_specs:
        updates, mean, se = curve_statistics(path_rows, method)
        ax.plot(
            updates,
            mean,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(
            updates,
            mean - 1.96 * se,
            mean + 1.96 * se,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    ax.set_xlabel("Minibatch updates")
    ax.set_ylabel("Population value")
    ax.set_title("Static rules and MSE-oracle selection")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    thresholds = np.asarray([row["threshold"] for row in threshold_rows])
    values = np.asarray([row["mean_final_value"] for row in threshold_rows])
    errors = np.asarray([row["se_final_value"] for row in threshold_rows])
    ax.errorbar(
        thresholds,
        values,
        yerr=errors,
        color=ESS_COLOR,
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="Sample-ESS gate",
    )
    for method, label, color, style in (
        ("raw", "Unmodified", RAW_COLOR, ":"),
        ("ppo", "PPO", PPO_COLOR, ":"),
        ("mse_oracle", "MSE oracle", ORACLE_COLOR, "--"),
    ):
        ax.axhline(
            by_method[method]["mean_final_value"],
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"ESS threshold $\tau$")
    ax.set_ylabel("Final population value")
    ax.set_title("Heuristic threshold sweep")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    agreements = np.asarray(
        [row["frozen_redraw_oracle_agreement"] for row in threshold_rows]
    )
    fractions = np.asarray([row["mean_ppo_fraction"] for row in threshold_rows])
    ax.plot(
        thresholds,
        agreements,
        color=ORACLE_COLOR,
        marker="s",
        linewidth=2.0,
        label="Agreement with MSE oracle",
    )
    ax.plot(
        thresholds,
        fractions,
        color=ESS_COLOR,
        marker="o",
        linewidth=2.0,
        label="PPO update fraction",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"ESS threshold $\tau$")
    ax.set_ylabel("Fraction")
    ax.set_title("What the heuristic approximates")
    ax.legend(frameon=False, fontsize=7.8)

    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return best_method


def summary_text(
    initial_value: float,
    state_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
    final_rows: list[dict[str, float]],
    threshold_rows: list[dict[str, float]],
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
    below_harm = (
        float(np.mean([row["reward_change"] < 0.0 for row in below]))
        if below
        else float("nan")
    )
    above_harm = (
        float(np.mean([row["reward_change"] < 0.0 for row in above]))
        if above
        else float("nan")
    )
    oracle_negative = float(np.mean([row["oracle_change"] < 0.0 for row in state_rows]))
    by_method = {str(row["method"]): row for row in final_rows}
    best = max(threshold_rows, key=lambda row: row["mean_final_value"])
    lines = [
        f"initial_value={initial_value:.8f}",
        f"states={len(state_rows)}",
        f"population_rho_min={np.min(rho):.10f}",
        f"population_rho_median={np.median(rho):.6f}",
        f"population_rho_max={np.max(rho):.6f}",
        f"corr_log_rho_log_raw_mse={log_corr:.6f}",
        f"relative_error_below_one_count={len(below)}",
        f"relative_error_below_one_harm_rate={below_harm:.6f}",
        f"relative_error_above_one_count={len(above)}",
        f"relative_error_above_one_harm_rate={above_harm:.6f}",
        f"population_gradient_negative_rate={oracle_negative:.6f}",
        f"final_value_raw_mean={by_method['raw']['mean_final_value']:.8f}",
        f"final_value_ppo_mean={by_method['ppo']['mean_final_value']:.8f}",
        f"final_value_mse_oracle_mean={by_method['mse_oracle']['mean_final_value']:.8f}",
        f"mse_oracle_ppo_fraction={by_method['mse_oracle']['mean_ppo_fraction']:.8f}",
        f"best_ess_threshold={best['threshold']:.6f}",
        f"best_ess_final_value={best['mean_final_value']:.8f}",
        f"best_ess_ppo_fraction={best['mean_ppo_fraction']:.8f}",
        f"best_ess_oracle_agreement={best['frozen_redraw_oracle_agreement']:.8f}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=40)
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
    run_rows: list[dict[str, float]] = []
    next_state_id = 0

    for replication in range(config.replications):
        rng = np.random.default_rng(config.seed + replication)
        draws = common_randomness(rng, len(features), config)
        method_specs: list[tuple[str, float | None, bool]] = [
            ("raw", None, True),
            ("ppo", None, True),
            ("mse_oracle", None, False),
        ]
        method_specs.extend(
            ("ess", threshold, False) for threshold in ESS_THRESHOLDS
        )
        for method, threshold, collect_states in method_specs:
            states, rows, run_summary = run_trajectory(
                method,
                initial_weights,
                features,
                labels,
                draws,
                config,
                replication,
                next_state_id,
                threshold=threshold,
                collect_states=collect_states,
            )
            next_state_id += len(states)
            all_states.extend(states)
            path_rows.extend(rows)
            run_rows.append(run_summary)

    selected_states = choose_states(all_states, config.checkpoints)
    diagnostic_rng = np.random.default_rng(config.seed + 100000)
    state_rows, draw_rows = evaluate_frozen_states(
        selected_states,
        features,
        labels,
        config,
        diagnostic_rng,
    )
    bin_rows = quantile_bin_rows(state_rows)
    error_rows = relative_error_bin_rows(draw_rows)
    final_rows = aggregate_final_values(run_rows)
    threshold_rows = threshold_summary_rows(final_rows, draw_rows)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_categorical_paths.csv", path_rows)
    write_csv(result_dir / "optdigits_categorical_runs.csv", run_rows)
    write_csv(result_dir / "optdigits_categorical_frozen_states.csv", state_rows)
    write_csv(result_dir / "optdigits_categorical_redraws.csv", draw_rows)
    write_csv(result_dir / "optdigits_categorical_ess_bins.csv", bin_rows)
    write_csv(result_dir / "optdigits_categorical_error_bins.csv", error_rows)
    write_csv(result_dir / "optdigits_categorical_final_values.csv", final_rows)
    write_csv(result_dir / "optdigits_categorical_thresholds.csv", threshold_rows)

    make_theory_figure(
        bin_rows,
        error_rows,
        root / "figures" / "optdigits_categorical_theory",
    )
    make_control_figure(
        path_rows,
        final_rows,
        threshold_rows,
        root / "figures" / "optdigits_categorical_control",
    )
    summary = summary_text(
        initial_value,
        state_rows,
        draw_rows,
        final_rows,
        threshold_rows,
    )
    (result_dir / "optdigits_categorical_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
