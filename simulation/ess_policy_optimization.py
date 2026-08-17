"""RLVR-style simulation of ESS-gated policy optimization.

Optdigits images serve as prompts.  A policy emits a sixteen-token binary
response, and a deterministic verifier returns one exactly when the response
encodes the image label.  Every rollout is followed by sixteen sequential PPO
minibatch updates.  The raw, PPO-masked, and ESS-gated methods share prompts,
sampling uniforms, and minibatch orders within each replication.

The leave-one-out group advantage is used because it has the same sign as the
usual group-centered advantage while remaining an unbiased policy-gradient
estimator.  This permits an exact comparison with the finite-population policy
gradient at every diagnostic update.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

@dataclass(frozen=True)
class Config:
    seed: int = 73
    replications: int = 100
    diagnostic_replications: int = 20
    crossover_checkpoints_per_bin: int = 8
    crossover_draws_per_checkpoint: int = 32
    crossover_minibatches_per_draw: int = 4
    formula_calibration_blocks: int = 100
    formula_evaluation_draws: int = 100
    rollout_steps: int = 8
    optimization_epochs: int = 1
    prompts_per_rollout: int = 128
    responses_per_prompt: int = 16
    minibatches_per_rollout: int = 16
    response_tokens: int = 16
    classifier_steps: int = 500
    classifier_learning_rate: float = 0.8
    initialization_scale: float = 0.55
    policy_learning_rate: float = 3.0
    ppo_epsilon: float = 0.2
    ess_threshold: float = 0.1


METHODS = ("Raw", "PPO masked", "ESS gated")
FORMULA_MULTIPLIERS = (1, 2, 4, 8)
FORMULA_METHODS = (
    "Raw",
    "PPO masked",
    "Formula oracle",
    "Fitted ESS",
    "Effective-count transfer",
    "Fixed ESS 0.1",
)
ESS_BOUNDARIES = np.asarray(
    [0.0, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000001]
)
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/80/"
    "optical+recognition+of+handwritten+digits.zip"
)


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


def standard_error(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def digit_codes(labels: np.ndarray, response_tokens: int) -> np.ndarray:
    """Return a redundant binary code for each digit label."""
    if response_tokens % 4 != 0:
        raise ValueError("response_tokens must be a multiple of four")
    base = ((labels[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)
    return np.tile(base, (1, response_tokens // 4))


def fit_initial_policy(
    features: np.ndarray,
    codes: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Fit token heads by supervised logistic regression, then soften them."""
    weights = np.zeros((codes.shape[1], features.shape[1]))
    for _ in range(config.classifier_steps):
        probabilities = sigmoid(features @ weights.T)
        gradient = (probabilities - codes).T @ features / len(features)
        weights -= config.classifier_learning_rate * gradient
    return config.initialization_scale * weights


def population_value_and_gradient(
    weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute exact reward and gradient over the finite prompt population."""
    probabilities = sigmoid(features @ weights.T)
    correct_token_probability = (
        codes * probabilities + (1.0 - codes) * (1.0 - probabilities)
    )
    success_probability = np.prod(correct_token_probability, axis=1)
    value = float(np.mean(success_probability))
    residual = codes - probabilities
    gradient = (
        (success_probability[:, None] * residual).T @ features / len(features)
    )
    return value, gradient


def population_value(
    weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
) -> float:
    """Compute exact reward over the finite prompt population."""
    probabilities = sigmoid(features @ weights.T)
    correct_token_probability = (
        codes * probabilities + (1.0 - codes) * (1.0 - probabilities)
    )
    return float(np.mean(np.prod(correct_token_probability, axis=1)))


def population_sequence_ess(
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
) -> float:
    """Compute population normalized ESS for the sequence likelihood ratio."""
    current = np.clip(sigmoid(features @ weights.T), 1e-12, 1.0 - 1e-12)
    rollout = np.clip(
        sigmoid(features @ rollout_weights.T),
        1e-12,
        1.0 - 1e-12,
    )
    token_second_moment = (
        current**2 / rollout
        + (1.0 - current) ** 2 / (1.0 - rollout)
    )
    sequence_second_moment = np.prod(token_second_moment, axis=1)
    return float(1.0 / np.mean(sequence_second_moment))


def sequence_log_probability(
    actions: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return np.sum(
        actions * np.log(probabilities)
        + (1.0 - actions) * np.log(1.0 - probabilities),
        axis=-1,
    )


def make_replication_randomness(
    rng: np.random.Generator,
    population_size: int,
    config: Config,
) -> list[dict[str, np.ndarray]]:
    draws: list[dict[str, np.ndarray]] = []
    for _ in range(config.rollout_steps):
        prompts = rng.choice(
            population_size,
            size=config.prompts_per_rollout,
            replace=False,
        )
        uniforms = rng.random(
            (
                config.prompts_per_rollout,
                config.responses_per_prompt,
                config.response_tokens,
            )
        )
        prompt_order = rng.permutation(config.prompts_per_rollout)
        draws.append(
            {
                "prompts": prompts,
                "uniforms": uniforms,
                "prompt_order": prompt_order,
            }
        )
    return draws


def make_minibatch_randomness(
    rng: np.random.Generator,
    population_size: int,
    config: Config,
) -> dict[str, np.ndarray]:
    """Draw one fresh minibatch with the same marginal law as training."""
    prompts_per_minibatch = (
        config.prompts_per_rollout // config.minibatches_per_rollout
    )
    return {
        "prompts": rng.choice(
            population_size,
            size=prompts_per_minibatch,
            replace=False,
        ),
        "uniforms": rng.random(
            (
                prompts_per_minibatch,
                config.responses_per_prompt,
                config.response_tokens,
            )
        ),
        "prompt_order": np.arange(prompts_per_minibatch),
    }


def draw_base_gradient_pair(
    rng: np.random.Generator,
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw paired raw and PPO gradients for one training-size minibatch."""
    random_draw = make_minibatch_randomness(rng, len(features), config)
    rollout = collect_rollout(
        rollout_weights,
        features,
        codes,
        random_draw,
    )
    group_indices = rollout["prompt_order"]
    raw_gradient, masked_gradient, _, ratios = minibatch_gradients(
        weights,
        rollout,
        group_indices,
        config,
    )
    return raw_gradient, masked_gradient, ratios


def collect_rollout(
    rollout_weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
    random_draw: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    prompt_indices = random_draw["prompts"]
    prompt_features = features[prompt_indices]
    prompt_codes = codes[prompt_indices]
    rollout_probabilities = sigmoid(prompt_features @ rollout_weights.T)
    actions = (
        random_draw["uniforms"] < rollout_probabilities[:, None, :]
    ).astype(float)
    rewards = np.all(
        actions == prompt_codes[:, None, :], axis=-1
    ).astype(float)

    reward_sum = np.sum(rewards, axis=1, keepdims=True)
    advantages = rewards - (reward_sum - rewards) / (rewards.shape[1] - 1)
    rollout_log_probabilities = sequence_log_probability(
        actions,
        rollout_probabilities[:, None, :],
    )
    return {
        "prompt_indices": prompt_indices,
        "features": prompt_features,
        "codes": prompt_codes,
        "actions": actions,
        "rewards": rewards,
        "advantages": advantages,
        "rollout_log_probabilities": rollout_log_probabilities,
        "prompt_order": random_draw["prompt_order"],
    }


def minibatch_gradients(
    weights: np.ndarray,
    rollout: dict[str, np.ndarray],
    group_indices: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    batch_features = rollout["features"][group_indices]
    actions = rollout["actions"][group_indices]
    advantages = rollout["advantages"][group_indices]
    rollout_log_probabilities = rollout["rollout_log_probabilities"][group_indices]

    current_probabilities = sigmoid(batch_features @ weights.T)
    current_log_probabilities = sequence_log_probability(
        actions,
        current_probabilities[:, None, :],
    )
    log_ratios = np.clip(
        current_log_probabilities - rollout_log_probabilities,
        -30.0,
        30.0,
    )
    ratios = np.exp(log_ratios)

    positive = advantages >= 0.0
    ppo_mask = np.where(
        positive,
        ratios <= 1.0 + config.ppo_epsilon,
        ratios >= 1.0 - config.ppo_epsilon,
    )
    residual = actions - current_probabilities[:, None, :]
    raw_coefficients = ratios * advantages
    masked_coefficients = raw_coefficients * ppo_mask
    denominator = actions.shape[0] * actions.shape[1]
    raw_gradient = np.einsum(
        "bg,bgl,bd->ld",
        raw_coefficients,
        residual,
        batch_features,
        optimize=True,
    ) / denominator
    masked_gradient = np.einsum(
        "bg,bgl,bd->ld",
        masked_coefficients,
        residual,
        batch_features,
        optimize=True,
    ) / denominator

    flat_ratios = ratios.ravel()
    normalized_ess = float(
        np.sum(flat_ratios) ** 2
        / (len(flat_ratios) * np.sum(flat_ratios**2))
    )
    return raw_gradient, masked_gradient, normalized_ess, flat_ratios


def simulate_method(
    method: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
    random_draws: list[dict[str, np.ndarray]],
    config: Config,
    replication: int,
    collect_diagnostics: bool,
    diagnostic_ess_ceiling: float | None = None,
    collect_population_ess: bool = True,
    collect_checkpoints: bool = True,
) -> tuple[
    np.ndarray,
    list[dict[str, float]],
    dict[str, float],
    list[dict[str, object]],
]:
    weights = initial_weights.copy()
    initial_value, _ = population_value_and_gradient(weights, features, codes)
    values = [initial_value]
    diagnostic_rows: list[dict[str, float]] = []
    diagnostic_checkpoints: list[dict[str, object]] = []
    gate_raw_updates = 0
    gate_masked_updates = 0
    all_ess: list[float] = []

    for rollout_index, random_draw in enumerate(random_draws, start=1):
        rollout_weights = weights.copy()
        rollout = collect_rollout(
            rollout_weights,
            features,
            codes,
            random_draw,
        )
        minibatches = np.split(
            rollout["prompt_order"],
            config.minibatches_per_rollout,
        )
        update_groups = [
            group_indices
            for _ in range(config.optimization_epochs)
            for group_indices in minibatches
        ]
        for minibatch_index, group_indices in enumerate(update_groups, start=1):
            raw_gradient, masked_gradient, normalized_ess, ratios = (
                minibatch_gradients(
                    weights,
                    rollout,
                    group_indices,
                    config,
                )
            )
            all_ess.append(normalized_ess)

            should_collect = collect_diagnostics and (
                diagnostic_ess_ceiling is None
                or normalized_ess < diagnostic_ess_ceiling
            )
            if should_collect:
                value_before, true_gradient = population_value_and_gradient(
                    weights,
                    features,
                    codes,
                )
                raw_value = population_value(
                    weights + config.policy_learning_rate * raw_gradient,
                    features,
                    codes,
                )
                ppo_value = population_value(
                    weights + config.policy_learning_rate * masked_gradient,
                    features,
                    codes,
                )
                oracle_value = (
                    population_value(
                        weights + config.policy_learning_rate * true_gradient,
                        features,
                        codes,
                    )
                    if collect_checkpoints
                    else float("nan")
                )
                exact_ess = (
                    population_sequence_ess(
                        weights,
                        rollout_weights,
                        features,
                    )
                    if collect_population_ess
                    else float("nan")
                )
                diagnostic_rows.append(
                    {
                        "replication": float(replication),
                        "rollout_batch": float(rollout_index),
                        "minibatch": float(minibatch_index),
                        "normalized_ess": normalized_ess,
                        "population_sequence_ess": exact_ess,
                        "effective_sequences": normalized_ess * len(ratios),
                        "mean_ratio": float(np.mean(ratios)),
                        "max_ratio": float(np.max(ratios)),
                        "raw_gradient_mse": float(
                            np.sum((raw_gradient - true_gradient) ** 2)
                        ),
                        "ppo_gradient_mse": float(
                            np.sum((masked_gradient - true_gradient) ** 2)
                        ),
                        "raw_minus_ppo_mse": float(
                            np.sum((raw_gradient - true_gradient) ** 2)
                            - np.sum((masked_gradient - true_gradient) ** 2)
                        ),
                        "raw_relative_gain_pct": float(
                            100.0 * (raw_value - value_before) / value_before
                        ),
                        "ppo_relative_gain_pct": float(
                            100.0 * (ppo_value - value_before) / value_before
                        ),
                        "ppo_minus_raw_gain_pct": float(
                            100.0 * (ppo_value - raw_value) / value_before
                        ),
                        "oracle_relative_gain_pct": float(
                            100.0 * (oracle_value - value_before) / value_before
                        ),
                    }
                )
                if collect_checkpoints:
                    diagnostic_checkpoints.append(
                        {
                            "replication": replication,
                            "rollout_batch": rollout_index,
                            "minibatch": minibatch_index,
                            "observed_normalized_ess": normalized_ess,
                            "population_sequence_ess": exact_ess,
                            "weights": weights.copy(),
                            "rollout_weights": rollout_weights.copy(),
                        }
                    )

            if method == "Raw":
                selected_gradient = raw_gradient
            elif method == "PPO masked":
                selected_gradient = masked_gradient
            elif normalized_ess >= config.ess_threshold:
                selected_gradient = raw_gradient
                gate_raw_updates += 1
            else:
                selected_gradient = masked_gradient
                gate_masked_updates += 1
            weights += config.policy_learning_rate * selected_gradient

        value, _ = population_value_and_gradient(weights, features, codes)
        values.append(value)

    metadata = {
        "initial_value": initial_value,
        "gate_raw_updates": float(gate_raw_updates),
        "gate_masked_updates": float(gate_masked_updates),
        "mean_minibatch_ess": float(np.mean(all_ess)),
        "minimum_minibatch_ess": float(np.min(all_ess)),
    }
    return np.asarray(values), diagnostic_rows, metadata, diagnostic_checkpoints


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def aggregate_paths(
    values_by_method: dict[str, list[np.ndarray]],
    methods: tuple[str, ...] = METHODS,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method in methods:
        values = np.stack(values_by_method[method])
        relative = 100.0 * (values - values[:, [0]]) / values[:, [0]]
        for step in range(values.shape[1]):
            rows.append(
                {
                    "method": method,
                    "rollout_batch": float(step),
                    "mean_population_reward": float(np.mean(values[:, step])),
                    "population_reward_se": standard_error(values[:, step]),
                    "mean_relative_improvement_pct": float(
                        np.mean(relative[:, step])
                    ),
                    "relative_improvement_pct_se": standard_error(
                        relative[:, step]
                    ),
                }
            )
    return rows


def summarize_runs(
    values_by_method: dict[str, list[np.ndarray]],
    metadata_by_method: dict[str, list[dict[str, float]]],
    config: Config,
    methods: tuple[str, ...] = METHODS,
) -> list[dict[str, float | str]]:
    final_values = {
        method: np.asarray([path[-1] for path in values_by_method[method]])
        for method in methods
    }
    initial_values = np.asarray([path[0] for path in values_by_method[methods[0]]])
    rows: list[dict[str, float | str]] = []
    for method in methods:
        relative = 100.0 * (
            final_values[method] - initial_values
        ) / initial_values
        gate_minus_raw = final_values[method] - final_values["Raw"]
        gate_minus_ppo = final_values[method] - final_values["PPO masked"]
        fitted_difference = (
            final_values[method] - final_values["ESS fitted"]
            if "ESS fitted" in final_values
            else None
        )
        ordered_final = np.sort(final_values[method])
        tail_count = max(1, int(math.ceil(0.1 * len(ordered_final))))
        metadata = metadata_by_method[method]
        rows.append(
            {
                "method": method,
                "final_population_reward": float(np.mean(final_values[method])),
                "final_population_reward_se": standard_error(final_values[method]),
                "median_final_population_reward": float(
                    np.median(final_values[method])
                ),
                "p10_final_population_reward": float(
                    np.quantile(final_values[method], 0.1)
                ),
                "lower_tail_mean_final_population_reward": float(
                    np.mean(ordered_final[:tail_count])
                ),
                "final_relative_improvement_pct": float(np.mean(relative)),
                "final_relative_improvement_pct_se": standard_error(relative),
                "paired_difference_from_raw": float(np.mean(gate_minus_raw)),
                "paired_difference_from_raw_se": standard_error(gate_minus_raw),
                "paired_win_rate_vs_raw_pct": float(
                    100.0 * np.mean(gate_minus_raw > 0.0)
                ),
                "paired_difference_from_ppo": float(np.mean(gate_minus_ppo)),
                "paired_difference_from_ppo_se": standard_error(gate_minus_ppo),
                "paired_difference_from_fitted": (
                    ""
                    if fitted_difference is None
                    else float(np.mean(fitted_difference))
                ),
                "paired_difference_from_fitted_se": (
                    ""
                    if fitted_difference is None
                    else standard_error(fitted_difference)
                ),
                "paired_win_rate_vs_fitted_pct": (
                    ""
                    if fitted_difference is None
                    else float(100.0 * np.mean(fitted_difference > 0.0))
                ),
                "mean_gate_raw_updates": float(
                    np.mean([row["gate_raw_updates"] for row in metadata])
                ),
                "mean_gate_masked_updates": float(
                    np.mean([row["gate_masked_updates"] for row in metadata])
                ),
                "mean_minibatch_ess": float(
                    np.mean([row["mean_minibatch_ess"] for row in metadata])
                ),
                "mean_minimum_minibatch_ess": float(
                    np.mean([row["minimum_minibatch_ess"] for row in metadata])
                ),
                "replications": float(config.replications),
                "rollout_batches": float(config.rollout_steps),
                "minibatches_per_rollout": float(config.minibatches_per_rollout),
                "optimization_epochs": float(config.optimization_epochs),
                "prompts_per_rollout": float(config.prompts_per_rollout),
                "responses_per_prompt": float(config.responses_per_prompt),
                "response_tokens": float(config.response_tokens),
                "policy_learning_rate": config.policy_learning_rate,
                "ppo_epsilon": config.ppo_epsilon,
                "ess_threshold": config.ess_threshold,
            }
        )
    return rows


def run_scaled_batch_optimization(
    initial_weights: np.ndarray,
    features: np.ndarray,
    codes: np.ndarray,
    threshold_rows: list[dict[str, float | str]],
    config: Config,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
]:
    """Run a paired optimization study with 512 sequences per update."""
    threshold_row = next(
        row
        for row in threshold_rows
        if int(float(row["sequences_per_estimator"])) == 512
    )
    method_thresholds: dict[str, float | None] = {
        "Raw": None,
        "PPO masked": None,
        "ESS fitted": float(threshold_row["fitted_ess_threshold"]),
        "ESS count transfer": float(
            threshold_row["effective_count_threshold"]
        ),
        "ESS fixed 0.1": 0.1,
    }
    methods = tuple(method_thresholds)
    scaled_config = replace(
        config,
        prompts_per_rollout=512,
        minibatches_per_rollout=16,
        optimization_epochs=4,
    )
    values_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in methods
    }
    metadata_by_method: dict[str, list[dict[str, float]]] = {
        method: [] for method in methods
    }
    on_trajectory_diagnostics: list[dict[str, float | str]] = []
    diagnostic_ceiling = max(
        float(threshold)
        for threshold in method_thresholds.values()
        if threshold is not None
    )
    master_rng = np.random.default_rng(config.seed + 424242)
    for replication in range(config.replications):
        replication_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
        random_draws = make_replication_randomness(
            np.random.default_rng(replication_seed),
            len(features),
            scaled_config,
        )
        for method, threshold in method_thresholds.items():
            if method == "Raw":
                simulation_method = "Raw"
                method_config = scaled_config
            elif method == "PPO masked":
                simulation_method = "PPO masked"
                method_config = scaled_config
            else:
                simulation_method = "ESS gated"
                method_config = replace(
                    scaled_config,
                    ess_threshold=float(threshold),
                )
            collect_diagnostics = method == "ESS fixed 0.1"
            values, diagnostic_rows, metadata, _ = simulate_method(
                simulation_method,
                initial_weights,
                features,
                codes,
                random_draws,
                method_config,
                replication,
                collect_diagnostics,
                diagnostic_ess_ceiling=diagnostic_ceiling,
                collect_population_ess=False,
                collect_checkpoints=False,
            )
            values_by_method[method].append(values)
            metadata_by_method[method].append(metadata)
            for row in diagnostic_rows:
                row["trajectory_method"] = method
            on_trajectory_diagnostics.extend(diagnostic_rows)

    path_rows = aggregate_paths(values_by_method, methods)
    summary_rows = summarize_runs(
        values_by_method,
        metadata_by_method,
        scaled_config,
        methods,
    )
    for row in summary_rows:
        threshold = method_thresholds[str(row["method"])]
        row["ess_threshold"] = "" if threshold is None else float(threshold)
        row["sequences_per_update"] = 512.0
        row["responses_per_rollout"] = float(
            scaled_config.prompts_per_rollout
            * scaled_config.responses_per_prompt
        )
    for row in path_rows:
        row["sequences_per_update"] = 512.0
        row["responses_per_rollout"] = float(
            scaled_config.prompts_per_rollout
            * scaled_config.responses_per_prompt
        )
    on_trajectory_bins, on_trajectory_gate_summary = (
        aggregate_on_trajectory_gate_diagnostics(
            on_trajectory_diagnostics,
            {
                method: float(threshold)
                for method, threshold in method_thresholds.items()
                if threshold is not None
            },
            total_updates=(
                scaled_config.rollout_steps
                * scaled_config.minibatches_per_rollout
                * scaled_config.optimization_epochs
            ),
        )
    )
    return (
        path_rows,
        summary_rows,
        on_trajectory_diagnostics,
        on_trajectory_bins,
        on_trajectory_gate_summary,
    )


def aggregate_on_trajectory_gate_diagnostics(
    rows: list[dict[str, float | str]],
    thresholds: dict[str, float],
    total_updates: int,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
]:
    """Summarize shadow raw-versus-PPO decisions on the fixed-0.1 trajectory."""
    if not rows:
        raise ValueError("No on-trajectory diagnostics were collected")
    replications = sorted({int(float(row["replication"])) for row in rows})
    boundaries = sorted({0.0, *thresholds.values()})
    bin_rows: list[dict[str, float | str]] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        per_replication: list[dict[str, float]] = []
        for replication in replications:
            chunk = [
                row
                for row in rows
                if int(float(row["replication"])) == replication
                and lower <= float(row["normalized_ess"]) < upper
            ]
            if not chunk:
                continue
            raw_mse = np.asarray(
                [float(row["raw_gradient_mse"]) for row in chunk]
            )
            ppo_mse = np.asarray(
                [float(row["ppo_gradient_mse"]) for row in chunk]
            )
            ppo_reward_advantage = np.asarray(
                [float(row["ppo_minus_raw_gain_pct"]) for row in chunk]
            )
            raw_total = float(np.sum(raw_mse))
            mse_reduction = (
                100.0 * (raw_total - float(np.sum(ppo_mse))) / raw_total
                if raw_total > 0.0
                else 0.0
            )
            per_replication.append(
                {
                    "updates": float(len(chunk)),
                    "mean_raw_mse": float(np.mean(raw_mse)),
                    "mean_ppo_mse": float(np.mean(ppo_mse)),
                    "ppo_mse_reduction_pct": mse_reduction,
                    "ppo_minus_raw_gain_pct": float(
                        np.mean(ppo_reward_advantage)
                    ),
                    "ppo_mse_win_rate_pct": float(
                        100.0 * np.mean(ppo_mse < raw_mse)
                    ),
                    "ppo_reward_win_rate_pct": float(
                        100.0 * np.mean(ppo_reward_advantage > 0.0)
                    ),
                }
            )

        def values(key: str) -> np.ndarray:
            return np.asarray([row[key] for row in per_replication])

        bin_rows.append(
            {
                "ess_bin_lower": lower,
                "ess_bin_upper": upper,
                "ess_bin_label": f"[{lower:.3f},{upper:.3f})",
                "mean_updates_per_replication": float(
                    np.mean(values("updates"))
                ),
                "updates_per_replication_se": standard_error(values("updates")),
                "mean_raw_gradient_mse": float(
                    np.mean(values("mean_raw_mse"))
                ),
                "raw_gradient_mse_se": standard_error(values("mean_raw_mse")),
                "mean_ppo_gradient_mse": float(
                    np.mean(values("mean_ppo_mse"))
                ),
                "ppo_gradient_mse_se": standard_error(values("mean_ppo_mse")),
                "ppo_mse_reduction_pct": float(
                    np.mean(values("ppo_mse_reduction_pct"))
                ),
                "ppo_mse_reduction_pct_se": standard_error(
                    values("ppo_mse_reduction_pct")
                ),
                "ppo_minus_raw_gain_pct": float(
                    np.mean(values("ppo_minus_raw_gain_pct"))
                ),
                "ppo_minus_raw_gain_pct_se": standard_error(
                    values("ppo_minus_raw_gain_pct")
                ),
                "ppo_mse_win_rate_pct": float(
                    np.mean(values("ppo_mse_win_rate_pct"))
                ),
                "ppo_mse_win_rate_pct_se": standard_error(
                    values("ppo_mse_win_rate_pct")
                ),
                "ppo_reward_win_rate_pct": float(
                    np.mean(values("ppo_reward_win_rate_pct"))
                ),
                "ppo_reward_win_rate_pct_se": standard_error(
                    values("ppo_reward_win_rate_pct")
                ),
                "replications_with_updates": float(len(per_replication)),
            }
        )

    gate_rows: list[dict[str, float | str]] = []
    for method, threshold in thresholds.items():
        per_replication = []
        for replication in replications:
            chunk = [
                row
                for row in rows
                if int(float(row["replication"])) == replication
            ]
            raw_mse = np.asarray(
                [float(row["raw_gradient_mse"]) for row in chunk]
            )
            ppo_mse = np.asarray(
                [float(row["ppo_gradient_mse"]) for row in chunk]
            )
            ess = np.asarray([float(row["normalized_ess"]) for row in chunk])
            ppo_selected = ess < threshold
            selected_mse = np.where(ppo_selected, ppo_mse, raw_mse)
            raw_gain = np.asarray(
                [float(row["raw_relative_gain_pct"]) for row in chunk]
            )
            ppo_gain = np.asarray(
                [float(row["ppo_relative_gain_pct"]) for row in chunk]
            )
            selected_gain = np.where(ppo_selected, ppo_gain, raw_gain)
            raw_total = float(np.sum(raw_mse))
            selected_count = int(np.sum(ppo_selected))
            per_replication.append(
                {
                    "selected_updates": float(selected_count),
                    "mse_reduction_vs_raw_pct": (
                        100.0
                        * (raw_total - float(np.sum(selected_mse)))
                        / raw_total
                        if raw_total > 0.0
                        else 0.0
                    ),
                    "one_step_gain_advantage_vs_raw_pp": float(
                        np.sum(selected_gain - raw_gain) / total_updates
                    ),
                    "selected_mse_win_rate_pct": (
                        float(
                            100.0
                            * np.mean(ppo_mse[ppo_selected] < raw_mse[ppo_selected])
                        )
                        if selected_count
                        else 0.0
                    ),
                    "selected_reward_win_rate_pct": (
                        float(
                            100.0
                            * np.mean(ppo_gain[ppo_selected] > raw_gain[ppo_selected])
                        )
                        if selected_count
                        else 0.0
                    ),
                }
            )

        def gate_values(key: str) -> np.ndarray:
            return np.asarray([row[key] for row in per_replication])

        gate_rows.append(
            {
                "method": method,
                "sample_ess_threshold": threshold,
                "mean_selected_updates": float(
                    np.mean(gate_values("selected_updates"))
                ),
                "selected_updates_se": standard_error(
                    gate_values("selected_updates")
                ),
                "mse_reduction_vs_raw_pct": float(
                    np.mean(gate_values("mse_reduction_vs_raw_pct"))
                ),
                "mse_reduction_vs_raw_pct_se": standard_error(
                    gate_values("mse_reduction_vs_raw_pct")
                ),
                "one_step_gain_advantage_vs_raw_pp": float(
                    np.mean(gate_values("one_step_gain_advantage_vs_raw_pp"))
                ),
                "one_step_gain_advantage_vs_raw_pp_se": standard_error(
                    gate_values("one_step_gain_advantage_vs_raw_pp")
                ),
                "selected_mse_win_rate_pct": float(
                    np.mean(gate_values("selected_mse_win_rate_pct"))
                ),
                "selected_mse_win_rate_pct_se": standard_error(
                    gate_values("selected_mse_win_rate_pct")
                ),
                "selected_reward_win_rate_pct": float(
                    np.mean(gate_values("selected_reward_win_rate_pct"))
                ),
                "selected_reward_win_rate_pct_se": standard_error(
                    gate_values("selected_reward_win_rate_pct")
                ),
                "replications": float(len(per_replication)),
                "total_updates_per_replication": float(total_updates),
                "trajectory_method": "ESS fixed 0.1",
            }
        )
    return bin_rows, gate_rows


def bin_diagnostics(
    diagnostic_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    ess_values = np.asarray(
        [float(row["normalized_ess"]) for row in diagnostic_rows]
    )
    output: list[dict[str, float]] = []
    for index, (lower, upper) in enumerate(
        zip(ESS_BOUNDARIES[:-1], ESS_BOUNDARIES[1:]), start=1
    ):
        chunk = np.flatnonzero((ess_values >= lower) & (ess_values < upper))
        if len(chunk) == 0:
            continue
        ess = np.asarray(
            [float(diagnostic_rows[i]["normalized_ess"]) for i in chunk]
        )
        mse = np.asarray(
            [float(diagnostic_rows[i]["raw_gradient_mse"]) for i in chunk]
        )
        ppo_mse = np.asarray(
            [float(diagnostic_rows[i]["ppo_gradient_mse"]) for i in chunk]
        )
        mse_difference = np.asarray(
            [float(diagnostic_rows[i]["raw_minus_ppo_mse"]) for i in chunk]
        )
        raw_gain = np.asarray(
            [float(diagnostic_rows[i]["raw_relative_gain_pct"]) for i in chunk]
        )
        oracle_gain = np.asarray(
            [float(diagnostic_rows[i]["oracle_relative_gain_pct"]) for i in chunk]
        )
        ppo_gain = np.asarray(
            [float(diagnostic_rows[i]["ppo_relative_gain_pct"]) for i in chunk]
        )
        gain_difference = np.asarray(
            [float(diagnostic_rows[i]["ppo_minus_raw_gain_pct"]) for i in chunk]
        )
        output.append(
            {
                "ess_bin": float(index),
                "ess_bin_lower": float(lower),
                "ess_bin_upper": float(upper),
                "ess_min": float(np.min(ess)),
                "ess_max": float(np.max(ess)),
                "median_normalized_ess": float(np.median(ess)),
                "mean_raw_gradient_mse": float(np.mean(mse)),
                "raw_gradient_mse_se": standard_error(mse),
                "mean_ppo_gradient_mse": float(np.mean(ppo_mse)),
                "ppo_gradient_mse_se": standard_error(ppo_mse),
                "mean_raw_minus_ppo_mse": float(np.mean(mse_difference)),
                "raw_minus_ppo_mse_se": standard_error(mse_difference),
                "mean_raw_relative_gain_pct": float(np.mean(raw_gain)),
                "raw_relative_gain_pct_se": standard_error(raw_gain),
                "mean_oracle_relative_gain_pct": float(np.mean(oracle_gain)),
                "oracle_relative_gain_pct_se": standard_error(oracle_gain),
                "mean_ppo_relative_gain_pct": float(np.mean(ppo_gain)),
                "ppo_relative_gain_pct_se": standard_error(ppo_gain),
                "mean_ppo_minus_raw_gain_pct": float(
                    np.mean(gain_difference)
                ),
                "ppo_minus_raw_gain_pct_se": standard_error(gain_difference),
                "observations": float(len(chunk)),
            }
        )
    return output


def select_crossover_checkpoints(
    checkpoints: list[dict[str, object]],
    config: Config,
) -> list[dict[str, object]]:
    """Select a fixed number of policy states from each population-ESS bin."""
    rng = np.random.default_rng(config.seed + 99173)
    selected: list[dict[str, object]] = []
    for bin_index, (lower, upper) in enumerate(
        zip(ESS_BOUNDARIES[:-1], ESS_BOUNDARIES[1:]), start=1
    ):
        candidates = [
            checkpoint
            for checkpoint in checkpoints
            if lower
            <= float(checkpoint["population_sequence_ess"])
            < upper
        ]
        if not candidates:
            continue
        count = min(config.crossover_checkpoints_per_bin, len(candidates))
        indices = rng.choice(len(candidates), size=count, replace=False)
        for index in np.sort(indices):
            checkpoint = dict(candidates[int(index)])
            checkpoint["selection_bin"] = bin_index
            selected.append(checkpoint)
    return selected


def run_crossover_diagnostics(
    checkpoints: list[dict[str, object]],
    features: np.ndarray,
    codes: np.ndarray,
    config: Config,
) -> list[dict[str, float]]:
    """Estimate raw and PPO-masked MSE at frozen policy checkpoints."""
    rng = np.random.default_rng(config.seed + 271828)
    rows: list[dict[str, float]] = []
    for checkpoint_id, checkpoint in enumerate(checkpoints, start=1):
        weights = np.asarray(checkpoint["weights"])
        rollout_weights = np.asarray(checkpoint["rollout_weights"])
        value_before, true_gradient = population_value_and_gradient(
            weights,
            features,
            codes,
        )
        for draw_index in range(1, config.crossover_draws_per_checkpoint + 1):
            raw_gradients: list[np.ndarray] = []
            masked_gradients: list[np.ndarray] = []
            ratio_batches: list[np.ndarray] = []
            for _ in range(config.crossover_minibatches_per_draw):
                random_draw = make_minibatch_randomness(
                    rng,
                    len(features),
                    config,
                )
                rollout = collect_rollout(
                    rollout_weights,
                    features,
                    codes,
                    random_draw,
                )
                group_indices = rollout["prompt_order"]
                raw_gradient, masked_gradient, _, ratios = minibatch_gradients(
                    weights,
                    rollout,
                    group_indices,
                    config,
                )
                raw_gradients.append(raw_gradient)
                masked_gradients.append(masked_gradient)
                ratio_batches.append(ratios)
            multipliers = sorted(
                {1, config.crossover_minibatches_per_draw}
            )
            for minibatches_per_estimator in multipliers:
                raw_gradient = np.mean(
                    np.stack(raw_gradients[:minibatches_per_estimator]),
                    axis=0,
                )
                masked_gradient = np.mean(
                    np.stack(masked_gradients[:minibatches_per_estimator]),
                    axis=0,
                )
                all_ratios = np.concatenate(
                    ratio_batches[:minibatches_per_estimator]
                )
                sample_ess = float(
                    np.sum(all_ratios) ** 2
                    / (len(all_ratios) * np.sum(all_ratios**2))
                )
                raw_value = population_value(
                    weights + config.policy_learning_rate * raw_gradient,
                    features,
                    codes,
                )
                ppo_value = population_value(
                    weights + config.policy_learning_rate * masked_gradient,
                    features,
                    codes,
                )
                raw_mse = float(
                    np.sum((raw_gradient - true_gradient) ** 2)
                )
                ppo_mse = float(
                    np.sum((masked_gradient - true_gradient) ** 2)
                )
                raw_gain = float(
                    100.0 * (raw_value - value_before) / value_before
                )
                ppo_gain = float(
                    100.0 * (ppo_value - value_before) / value_before
                )
                rows.append(
                    {
                        "checkpoint_id": float(checkpoint_id),
                        "replication": float(checkpoint["replication"]),
                        "rollout_batch": float(checkpoint["rollout_batch"]),
                        "minibatch": float(checkpoint["minibatch"]),
                        "selection_bin": float(checkpoint["selection_bin"]),
                        "draw": float(draw_index),
                        "minibatches_per_estimator": float(
                            minibatches_per_estimator
                        ),
                        "population_sequence_ess": float(
                            checkpoint["population_sequence_ess"]
                        ),
                        "training_sample_ess": float(
                            checkpoint["observed_normalized_ess"]
                        ),
                        "diagnostic_sample_ess": sample_ess,
                        "effective_sequences": sample_ess * len(all_ratios),
                        "sequences_per_diagnostic": float(len(all_ratios)),
                        "raw_gradient_squared_error": raw_mse,
                        "ppo_gradient_squared_error": ppo_mse,
                        "raw_minus_ppo_squared_error": raw_mse - ppo_mse,
                        "raw_relative_gain_pct": raw_gain,
                        "ppo_relative_gain_pct": ppo_gain,
                        "ppo_minus_raw_gain_pct": ppo_gain - raw_gain,
                    }
                )
    return rows


def aggregate_crossover_checkpoints(
    diagnostic_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    """Average independent diagnostic redraws within each fixed checkpoint."""
    groups = sorted(
        {
            (
                int(float(row["checkpoint_id"])),
                int(float(row["minibatches_per_estimator"])),
            )
            for row in diagnostic_rows
        }
    )
    output: list[dict[str, float]] = []
    for checkpoint_id, minibatches_per_estimator in groups:
        chunk = [
            row
            for row in diagnostic_rows
            if int(float(row["checkpoint_id"])) == checkpoint_id
            and int(float(row["minibatches_per_estimator"]))
            == minibatches_per_estimator
        ]
        first = chunk[0]

        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in chunk])

        output.append(
            {
                "checkpoint_id": float(checkpoint_id),
                "replication": float(first["replication"]),
                "rollout_batch": float(first["rollout_batch"]),
                "minibatch": float(first["minibatch"]),
                "selection_bin": float(first["selection_bin"]),
                "minibatches_per_estimator": float(
                    minibatches_per_estimator
                ),
                "sequences_per_diagnostic": float(
                    first["sequences_per_diagnostic"]
                ),
                "population_sequence_ess": float(
                    first["population_sequence_ess"]
                ),
                "training_sample_ess": float(first["training_sample_ess"]),
                "mean_diagnostic_sample_ess": float(
                    np.mean(values("diagnostic_sample_ess"))
                ),
                "raw_gradient_mse": float(
                    np.mean(values("raw_gradient_squared_error"))
                ),
                "ppo_gradient_mse": float(
                    np.mean(values("ppo_gradient_squared_error"))
                ),
                "raw_minus_ppo_mse": float(
                    np.mean(values("raw_minus_ppo_squared_error"))
                ),
                "ppo_mse_reduction_pct": float(
                    100.0
                    * np.mean(values("raw_minus_ppo_squared_error"))
                    / np.mean(values("raw_gradient_squared_error"))
                ),
                "raw_relative_gain_pct": float(
                    np.mean(values("raw_relative_gain_pct"))
                ),
                "ppo_relative_gain_pct": float(
                    np.mean(values("ppo_relative_gain_pct"))
                ),
                "ppo_minus_raw_gain_pct": float(
                    np.mean(values("ppo_minus_raw_gain_pct"))
                ),
                "diagnostic_draws": float(len(chunk)),
            }
        )
    return output


def bin_crossover_checkpoints(
    checkpoint_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    """Aggregate fixed-checkpoint MSE estimates in prespecified ESS bins."""
    output: list[dict[str, float]] = []
    multipliers = sorted(
        {
            int(float(row["minibatches_per_estimator"]))
            for row in checkpoint_rows
        }
    )
    for minibatches_per_estimator in multipliers:
        matching_rows = [
            row
            for row in checkpoint_rows
            if int(float(row["minibatches_per_estimator"]))
            == minibatches_per_estimator
        ]
        for bin_index, (lower, upper) in enumerate(
            zip(ESS_BOUNDARIES[:-1], ESS_BOUNDARIES[1:]), start=1
        ):
            chunk = [
                row
                for row in matching_rows
                if lower <= float(row["population_sequence_ess"]) < upper
            ]
            if not chunk:
                continue

            def values(key: str) -> np.ndarray:
                return np.asarray([float(row[key]) for row in chunk])

            population_ess = values("population_sequence_ess")
            raw_mse = values("raw_gradient_mse")
            ppo_mse = values("ppo_gradient_mse")
            mse_difference = values("raw_minus_ppo_mse")
            relative_mse_reduction = values("ppo_mse_reduction_pct")
            raw_gain = values("raw_relative_gain_pct")
            ppo_gain = values("ppo_relative_gain_pct")
            gain_difference = values("ppo_minus_raw_gain_pct")
            output.append(
                {
                    "minibatches_per_estimator": float(
                        minibatches_per_estimator
                    ),
                    "sequences_per_diagnostic": float(
                        chunk[0]["sequences_per_diagnostic"]
                    ),
                    "ess_bin": float(bin_index),
                    "ess_bin_lower": float(lower),
                    "ess_bin_upper": float(upper),
                    "median_population_sequence_ess": float(
                        np.median(population_ess)
                    ),
                    "mean_raw_gradient_mse": float(np.mean(raw_mse)),
                    "raw_gradient_mse_se": standard_error(raw_mse),
                    "mean_ppo_gradient_mse": float(np.mean(ppo_mse)),
                    "ppo_gradient_mse_se": standard_error(ppo_mse),
                    "mean_raw_minus_ppo_mse": float(
                        np.mean(mse_difference)
                    ),
                    "raw_minus_ppo_mse_se": standard_error(mse_difference),
                    "mean_ppo_mse_reduction_pct": float(
                        np.mean(relative_mse_reduction)
                    ),
                    "ppo_mse_reduction_pct_se": standard_error(
                        relative_mse_reduction
                    ),
                    "mean_raw_relative_gain_pct": float(np.mean(raw_gain)),
                    "raw_relative_gain_pct_se": standard_error(raw_gain),
                    "mean_ppo_relative_gain_pct": float(np.mean(ppo_gain)),
                    "ppo_relative_gain_pct_se": standard_error(ppo_gain),
                    "mean_ppo_minus_raw_gain_pct": float(
                        np.mean(gain_difference)
                    ),
                    "ppo_minus_raw_gain_pct_se": standard_error(
                        gain_difference
                    ),
                    "checkpoints": float(len(chunk)),
                    "diagnostic_draws": float(
                        np.sum(values("diagnostic_draws"))
                    ),
                }
            )
    return output


def assign_formula_splits(
    checkpoints: list[dict[str, object]],
    config: Config,
) -> list[dict[str, object]]:
    """Split selected checkpoints within every ESS bin for threshold fitting."""
    indexed: list[dict[str, object]] = []
    for checkpoint_id, checkpoint in enumerate(checkpoints, start=1):
        row = dict(checkpoint)
        row["checkpoint_id"] = checkpoint_id
        indexed.append(row)

    rng = np.random.default_rng(config.seed + 161803)
    split_by_id: dict[int, str] = {}
    for bin_index in range(1, len(ESS_BOUNDARIES)):
        members = [
            row
            for row in indexed
            if int(row["selection_bin"]) == bin_index
        ]
        order = rng.permutation(len(members))
        fit_count = len(members) // 2
        for position, member_index in enumerate(order):
            checkpoint_id = int(members[int(member_index)]["checkpoint_id"])
            split_by_id[checkpoint_id] = (
                "fit" if position < fit_count else "test"
            )
    for row in indexed:
        row["formula_split"] = split_by_id[int(row["checkpoint_id"])]
    return indexed


def estimate_formula_components(
    checkpoints: list[dict[str, object]],
    features: np.ndarray,
    codes: np.ndarray,
    config: Config,
) -> list[dict[str, float | str]]:
    """Estimate the bias and covariance terms in the crossover formula."""
    rng = np.random.default_rng(config.seed + 314159)
    rows: list[dict[str, float | str]] = []
    base_sequences = (
        config.prompts_per_rollout
        // config.minibatches_per_rollout
        * config.responses_per_prompt
    )
    calibration_blocks = config.formula_calibration_blocks
    for checkpoint in checkpoints:
        weights = np.asarray(checkpoint["weights"])
        rollout_weights = np.asarray(checkpoint["rollout_weights"])
        _, true_gradient = population_value_and_gradient(
            weights,
            features,
            codes,
        )
        raw_blocks: list[np.ndarray] = []
        ppo_blocks: list[np.ndarray] = []
        for _ in range(calibration_blocks):
            raw_gradient, ppo_gradient, _ = draw_base_gradient_pair(
                rng,
                weights,
                rollout_weights,
                features,
                codes,
                config,
            )
            raw_blocks.append(raw_gradient)
            ppo_blocks.append(ppo_gradient)

        raw_array = np.stack(raw_blocks)
        ppo_array = np.stack(ppo_blocks)
        ppo_mean = np.mean(ppo_array, axis=0)
        ppo_centered = ppo_array - ppo_mean
        raw_variance_trace = float(
            np.mean(np.sum((raw_array - true_gradient) ** 2, axis=(1, 2)))
        )
        ppo_variance_trace = float(
            np.sum(ppo_centered**2)
            / (calibration_blocks - 1)
        )
        ppo_bias_sq_naive = float(np.sum((ppo_mean - true_gradient) ** 2))
        ppo_bias_sq = max(
            0.0,
            ppo_bias_sq_naive
            - ppo_variance_trace / calibration_blocks,
        )
        raw_mean = np.mean(raw_array, axis=0)
        empirical_raw_bias_sq = float(
            np.sum((raw_mean - true_gradient) ** 2)
        )

        for multiplier in FORMULA_MULTIPLIERS:
            raw_formula_mse = raw_variance_trace / multiplier
            ppo_formula_mse = (
                ppo_bias_sq + ppo_variance_trace / multiplier
            )
            rows.append(
                {
                    "checkpoint_id": float(checkpoint["checkpoint_id"]),
                    "replication": float(checkpoint["replication"]),
                    "rollout_batch": float(checkpoint["rollout_batch"]),
                    "minibatch": float(checkpoint["minibatch"]),
                    "selection_bin": float(checkpoint["selection_bin"]),
                    "formula_split": str(checkpoint["formula_split"]),
                    "population_sequence_ess": float(
                        checkpoint["population_sequence_ess"]
                    ),
                    "blocks_per_estimator": float(multiplier),
                    "sequences_per_estimator": float(
                        base_sequences * multiplier
                    ),
                    "calibration_blocks": float(calibration_blocks),
                    "raw_variance_trace": raw_variance_trace,
                    "ppo_variance_trace": ppo_variance_trace,
                    "ppo_bias_squared": ppo_bias_sq,
                    "ppo_bias_squared_naive": ppo_bias_sq_naive,
                    "empirical_raw_bias_squared": empirical_raw_bias_sq,
                    "raw_formula_mse": raw_formula_mse,
                    "ppo_formula_mse": ppo_formula_mse,
                    "raw_minus_ppo_formula_mse": (
                        raw_formula_mse - ppo_formula_mse
                    ),
                    "ppo_formula_reduction_pct": (
                        100.0
                        * (raw_formula_mse - ppo_formula_mse)
                        / raw_formula_mse
                    ),
                    "formula_prefers_ppo": float(
                        ppo_formula_mse < raw_formula_mse
                    ),
                }
            )
    return rows


def fit_formula_ess_thresholds(
    component_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    """Fit one population-ESS threshold per estimator size on fit states."""
    output: list[dict[str, float]] = []
    for multiplier in FORMULA_MULTIPLIERS:
        rows = [
            row
            for row in component_rows
            if int(float(row["blocks_per_estimator"])) == multiplier
            and row["formula_split"] == "fit"
        ]
        ess = np.asarray(
            [float(row["population_sequence_ess"]) for row in rows]
        )
        raw_risk = np.asarray([float(row["raw_formula_mse"]) for row in rows])
        ppo_risk = np.asarray([float(row["ppo_formula_mse"]) for row in rows])
        unique = np.sort(np.unique(ess))
        candidates = np.concatenate(
            (
                np.asarray([0.0]),
                (unique[:-1] + unique[1:]) / 2.0,
                np.asarray([1.000001]),
            )
        )
        candidate_risks = np.asarray(
            [
                np.mean(np.where(ess < threshold, ppo_risk, raw_risk))
                for threshold in candidates
            ]
        )
        best_index = int(np.argmin(candidate_risks))
        output.append(
            {
                "blocks_per_estimator": float(multiplier),
                "sequences_per_estimator": float(
                    rows[0]["sequences_per_estimator"]
                ),
                "fitted_ess_threshold": float(candidates[best_index]),
                "fit_formula_mse": float(candidate_risks[best_index]),
                "fit_states": float(len(rows)),
            }
        )

    base_threshold = float(output[0]["fitted_ess_threshold"])
    for row in output:
        multiplier = int(row["blocks_per_estimator"])
        row["effective_count_threshold"] = min(
            1.000001,
            base_threshold / multiplier,
        )
        row["fixed_ess_threshold"] = 0.1
    return output


def run_formula_oracle_evaluation(
    checkpoints: list[dict[str, object]],
    component_rows: list[dict[str, float | str]],
    threshold_rows: list[dict[str, float]],
    features: np.ndarray,
    codes: np.ndarray,
    config: Config,
) -> list[dict[str, float]]:
    """Evaluate formula and ESS decisions on independent gradient redraws."""
    rng = np.random.default_rng(config.seed + 271901)
    component_lookup = {
        (
            int(float(row["checkpoint_id"])),
            int(float(row["blocks_per_estimator"])),
        ): row
        for row in component_rows
    }
    threshold_lookup = {
        int(float(row["blocks_per_estimator"])): row
        for row in threshold_rows
    }
    rows: list[dict[str, float]] = []
    for checkpoint in checkpoints:
        if checkpoint["formula_split"] != "test":
            continue
        checkpoint_id = int(checkpoint["checkpoint_id"])
        weights = np.asarray(checkpoint["weights"])
        rollout_weights = np.asarray(checkpoint["rollout_weights"])
        _, true_gradient = population_value_and_gradient(
            weights,
            features,
            codes,
        )
        for draw_index in range(1, config.formula_evaluation_draws + 1):
            raw_blocks: list[np.ndarray] = []
            ppo_blocks: list[np.ndarray] = []
            ratio_blocks: list[np.ndarray] = []
            for _ in range(max(FORMULA_MULTIPLIERS)):
                raw_gradient, ppo_gradient, ratios = draw_base_gradient_pair(
                    rng,
                    weights,
                    rollout_weights,
                    features,
                    codes,
                    config,
                )
                raw_blocks.append(raw_gradient)
                ppo_blocks.append(ppo_gradient)
                ratio_blocks.append(ratios)

            for multiplier in FORMULA_MULTIPLIERS:
                raw_gradient = np.mean(
                    np.stack(raw_blocks[:multiplier]),
                    axis=0,
                )
                ppo_gradient = np.mean(
                    np.stack(ppo_blocks[:multiplier]),
                    axis=0,
                )
                ratios = np.concatenate(ratio_blocks[:multiplier])
                sample_ess = float(
                    np.sum(ratios) ** 2
                    / (len(ratios) * np.sum(ratios**2))
                )
                component = component_lookup[(checkpoint_id, multiplier)]
                threshold = threshold_lookup[multiplier]
                population_ess = float(
                    checkpoint["population_sequence_ess"]
                )
                rows.append(
                    {
                        "checkpoint_id": float(checkpoint_id),
                        "selection_bin": float(checkpoint["selection_bin"]),
                        "draw": float(draw_index),
                        "blocks_per_estimator": float(multiplier),
                        "sequences_per_estimator": float(len(ratios)),
                        "population_sequence_ess": population_ess,
                        "sample_sequence_ess": sample_ess,
                        "raw_gradient_squared_error": float(
                            np.sum((raw_gradient - true_gradient) ** 2)
                        ),
                        "ppo_gradient_squared_error": float(
                            np.sum((ppo_gradient - true_gradient) ** 2)
                        ),
                        "formula_prefers_ppo": float(
                            component["formula_prefers_ppo"]
                        ),
                        "fitted_ess_prefers_ppo": float(
                            population_ess
                            < threshold["fitted_ess_threshold"]
                        ),
                        "effective_count_prefers_ppo": float(
                            population_ess
                            < threshold["effective_count_threshold"]
                        ),
                        "fixed_ess_prefers_ppo": float(
                            population_ess < 0.1
                        ),
                    }
                )
    return rows


def aggregate_formula_oracle_evaluation(
    evaluation_rows: list[dict[str, float | str]],
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Aggregate held-out redraws by checkpoint and then across checkpoints."""
    decision_keys = {
        "Raw": None,
        "PPO masked": "always_ppo",
        "Formula oracle": "formula_prefers_ppo",
        "Fitted ESS": "fitted_ess_prefers_ppo",
        "Effective-count transfer": "effective_count_prefers_ppo",
        "Fixed ESS 0.1": "fixed_ess_prefers_ppo",
    }
    groups = sorted(
        {
            (
                int(float(row["checkpoint_id"])),
                int(float(row["blocks_per_estimator"])),
            )
            for row in evaluation_rows
        }
    )
    checkpoint_rows: list[dict[str, float | str]] = []
    for checkpoint_id, multiplier in groups:
        chunk = [
            row
            for row in evaluation_rows
            if int(float(row["checkpoint_id"])) == checkpoint_id
            and int(float(row["blocks_per_estimator"])) == multiplier
        ]
        first = chunk[0]
        raw_error = np.asarray(
            [float(row["raw_gradient_squared_error"]) for row in chunk]
        )
        ppo_error = np.asarray(
            [float(row["ppo_gradient_squared_error"]) for row in chunk]
        )
        formula_decision = bool(float(first["formula_prefers_ppo"]))
        for method, decision_key in decision_keys.items():
            if method == "Raw":
                selects_ppo = False
            elif method == "PPO masked":
                selects_ppo = True
            else:
                selects_ppo = bool(float(first[decision_key]))
            selected_error = ppo_error if selects_ppo else raw_error
            checkpoint_rows.append(
                {
                    "checkpoint_id": float(checkpoint_id),
                    "selection_bin": float(first["selection_bin"]),
                    "blocks_per_estimator": float(multiplier),
                    "sequences_per_estimator": float(
                        first["sequences_per_estimator"]
                    ),
                    "population_sequence_ess": float(
                        first["population_sequence_ess"]
                    ),
                    "method": method,
                    "selected_ppo": float(selects_ppo),
                    "agrees_with_formula_oracle": float(
                        selects_ppo == formula_decision
                    ),
                    "heldout_gradient_mse": float(np.mean(selected_error)),
                    "evaluation_draws": float(len(chunk)),
                }
            )

    summary_rows: list[dict[str, float | str]] = []
    for multiplier in FORMULA_MULTIPLIERS:
        by_method: dict[str, list[dict[str, float | str]]] = {
            method: [
                row
                for row in checkpoint_rows
                if int(float(row["blocks_per_estimator"])) == multiplier
                and row["method"] == method
            ]
            for method in FORMULA_METHODS
        }
        mean_static = {
            method: float(
                np.mean(
                    [float(row["heldout_gradient_mse"]) for row in rows]
                )
            )
            for method, rows in by_method.items()
            if method in {"Raw", "PPO masked"}
        }
        best_static = min(mean_static, key=mean_static.get)
        base_by_checkpoint = {
            int(float(row["checkpoint_id"])): float(
                row["heldout_gradient_mse"]
            )
            for row in by_method[best_static]
        }
        for method in FORMULA_METHODS:
            rows = by_method[method]
            mse = np.asarray(
                [float(row["heldout_gradient_mse"]) for row in rows]
            )
            relative = np.asarray(
                [
                    100.0
                    * (
                        float(row["heldout_gradient_mse"])
                        - base_by_checkpoint[int(float(row["checkpoint_id"]))]
                    )
                    / base_by_checkpoint[int(float(row["checkpoint_id"]))]
                    for row in rows
                ]
            )
            summary_rows.append(
                {
                    "blocks_per_estimator": float(multiplier),
                    "sequences_per_estimator": float(
                        rows[0]["sequences_per_estimator"]
                    ),
                    "method": method,
                    "mean_heldout_gradient_mse": float(np.mean(mse)),
                    "heldout_gradient_mse_se": standard_error(mse),
                    "best_static_method": best_static,
                    "mean_excess_mse_vs_best_static_pct": float(
                        np.mean(relative)
                    ),
                    "excess_mse_vs_best_static_pct_se": standard_error(
                        relative
                    ),
                    "formula_agreement_pct": float(
                        100.0
                        * np.mean(
                            [
                                float(row["agrees_with_formula_oracle"])
                                for row in rows
                            ]
                        )
                    ),
                    "selected_ppo_pct": float(
                        100.0
                        * np.mean(
                            [float(row["selected_ppo"]) for row in rows]
                        )
                    ),
                    "test_checkpoints": float(len(rows)),
                    "evaluation_draws_per_checkpoint": float(
                        rows[0]["evaluation_draws"]
                    ),
                }
            )
    return checkpoint_rows, summary_rows


def make_crossover_figure(
    crossover_bins: list[dict[str, float | str]],
    output_base: Path,
) -> None:
    """Render the fixed-checkpoint estimator-crossover diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    blue = "#0072B2"
    vermillion = "#D55E00"
    gray = "#6F6F6F"
    light_gray = "#9A9A9A"
    multipliers = sorted(
        {
            int(float(row["minibatches_per_estimator"]))
            for row in crossover_bins
        }
    )
    if len(multipliers) != 2:
        raise ValueError("Crossover figure requires two diagnostic batch sizes")
    all_x = np.asarray(
        [float(row["median_population_sequence_ess"]) for row in crossover_bins]
    )
    x_lower = max(1e-4, float(np.min(all_x)) * 0.72)
    pooled_multiplier = max(multipliers)
    pooled_rows = [
        row
        for row in crossover_bins
        if int(float(row["minibatches_per_estimator"])) == pooled_multiplier
    ]
    pooled_x = np.asarray(
        [float(row["median_population_sequence_ess"]) for row in pooled_rows]
    )
    for key, error_key, label, color, marker, linestyle in (
        (
            "mean_raw_gradient_mse",
            "raw_gradient_mse_se",
            "Permissive",
            blue,
            "o",
            "-",
        ),
        (
            "mean_ppo_gradient_mse",
            "ppo_gradient_mse_se",
            "PPO masked",
            gray,
            "s",
            "--",
        ),
    ):
        axes[0].errorbar(
            pooled_x,
            np.asarray([float(row[key]) for row in pooled_rows]),
            yerr=np.asarray([float(row[error_key]) for row in pooled_rows]),
            marker=marker,
            linestyle=linestyle,
            color=color,
            capsize=1.8,
            elinewidth=0.9,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
            zorder=3,
        )
    pooled_sequences = int(
        float(pooled_rows[0]["sequences_per_diagnostic"])
    )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Gradient MSE")
    axes[0].set_title(
        f"(a) Raw versus PPO, $N={pooled_sequences}$",
        loc="left",
        pad=5,
    )
    axes[0].legend(loc="best", handlelength=2.0)

    reduction_styles = {
        min(multipliers): (gray, "--", "s"),
        max(multipliers): (vermillion, "-", "D"),
    }
    for multiplier in multipliers:
        rows = [
            row
            for row in crossover_bins
            if int(float(row["minibatches_per_estimator"])) == multiplier
        ]
        x = np.asarray(
            [float(row["median_population_sequence_ess"]) for row in rows]
        )
        sequences = int(float(rows[0]["sequences_per_diagnostic"]))
        color, linestyle, marker = reduction_styles[multiplier]
        axes[1].errorbar(
            x,
            np.asarray(
                [float(row["mean_ppo_mse_reduction_pct"]) for row in rows]
            ),
            yerr=np.asarray(
                [float(row["ppo_mse_reduction_pct_se"]) for row in rows]
            ),
            marker=marker,
            linestyle=linestyle,
            color=color,
            capsize=1.8,
            elinewidth=0.9,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=f"$N={sequences}$",
            zorder=3,
        )
    axes[1].axhline(0.0, color=light_gray, linewidth=0.7)
    axes[1].set_ylabel("PPO MSE reduction (%)")
    axes[1].set_title("(b) Batch size shifts crossover", loc="left", pad=5)
    axes[1].legend(loc="best", handlelength=2.0)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(x_lower, 1.05)
        axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Population normalized ESS")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", which="major", pad=2)
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.205,
        top=0.89,
        wspace=0.34,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(figure)


def make_formula_oracle_figure(
    component_rows: list[dict[str, float | str]],
    threshold_rows: list[dict[str, float | str]],
    _summary_rows: list[dict[str, float | str]],
    scaled_path_rows: list[dict[str, float | str]],
    on_trajectory_bin_rows: list[dict[str, float | str]],
    output_base: Path,
) -> None:
    """Render the formula-oracle and N-dependent ESS-rule diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.45,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    figure, axes_grid = plt.subplots(2, 2, figsize=(6.5, 4.15))
    axes = axes_grid.ravel()
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    markers = ("o", "s", "D", "^")
    linestyles = ("-", "--", "-.", ":")

    test_components = [
        row for row in component_rows if row["formula_split"] == "test"
    ]
    informative_multipliers = [
        int(float(row["blocks_per_estimator"]))
        for row in threshold_rows
        if 0.0 < float(row["fitted_ess_threshold"]) < 1.0
    ]
    if not informative_multipliers:
        informative_multipliers = list(FORMULA_MULTIPLIERS)
    all_ess = np.asarray(
        [
            float(row["population_sequence_ess"])
            for row in test_components
            if int(float(row["blocks_per_estimator"]))
            in informative_multipliers
        ]
    )
    x_lower = max(1e-4, float(np.min(all_ess)) * 0.72)
    for index, multiplier in enumerate(informative_multipliers):
        x_values: list[float] = []
        means: list[float] = []
        errors: list[float] = []
        for bin_index in range(1, len(ESS_BOUNDARIES)):
            rows = [
                row
                for row in test_components
                if int(float(row["blocks_per_estimator"])) == multiplier
                and int(float(row["selection_bin"])) == bin_index
            ]
            values = np.asarray(
                [float(row["ppo_formula_reduction_pct"]) for row in rows]
            )
            x_values.append(
                float(
                    np.median(
                        [float(row["population_sequence_ess"]) for row in rows]
                    )
                )
            )
            means.append(float(np.mean(values)))
            errors.append(standard_error(values))
        axes[0].errorbar(
            x_values,
            means,
            yerr=errors,
            color=colors[index],
            marker=markers[index],
            linestyle=linestyles[index],
            capsize=1.6,
            elinewidth=0.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=f"$N={128 * multiplier}$",
            zorder=3,
        )
    axes[0].axhline(0.0, color="#8A8A8A", linewidth=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlim(x_lower, 1.05)
    axes[0].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("Population normalized ESS")
    axes[0].set_ylabel("Predicted PPO MSE reduction (%)")
    axes[0].set_title("(a) Predicted crossover at informative $N$", loc="left", pad=5)
    axes[0].legend(loc="best", handlelength=2.0)

    threshold_rows = sorted(
        threshold_rows,
        key=lambda row: float(row["sequences_per_estimator"]),
    )
    sequence_counts = np.asarray(
        [float(row["sequences_per_estimator"]) for row in threshold_rows]
    )
    axes[1].plot(
        sequence_counts,
        [
            min(1.0, float(row["fitted_ess_threshold"]))
            for row in threshold_rows
        ],
        color="#0072B2",
        marker="o",
        label="Fitted for each $N$",
    )
    axes[1].plot(
        sequence_counts,
        [
            min(1.0, float(row["effective_count_threshold"]))
            for row in threshold_rows
        ],
        color="#009E73",
        marker="D",
        linestyle="--",
        label=r"Transfer fixed $N\rho$",
    )
    axes[1].axhline(
        0.1,
        color="#D55E00",
        linewidth=1.1,
        linestyle=":",
        label="Fixed 0.1",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sequence_counts)
    axes[1].set_xticklabels([str(int(value)) for value in sequence_counts])
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_xlabel("Sequences per estimator $N$")
    axes[1].set_ylabel("ESS decision threshold")
    axes[1].set_title("(b) The boundary moves with $N$", loc="left", pad=5)
    axes[1].legend(loc="best", handlelength=2.0)

    on_trajectory_bin_rows = sorted(
        on_trajectory_bin_rows,
        key=lambda row: float(row["ess_bin_lower"]),
    )
    positions = np.arange(len(on_trajectory_bin_rows))
    bars = axes[2].bar(
        positions,
        [float(row["ppo_mse_win_rate_pct"]) for row in on_trajectory_bin_rows],
        yerr=[
            float(row["ppo_mse_win_rate_pct_se"])
            for row in on_trajectory_bin_rows
        ],
        color=("#D55E00", "#56B4E9", "#56B4E9"),
        width=0.62,
        capsize=2.0,
        linewidth=0.0,
        zorder=3,
    )
    axes[2].axhline(
        50.0,
        color="#6F6F6F",
        linewidth=0.8,
        linestyle="--",
        label="No predictive value",
    )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(("$<0.1$", "$0.1$--$0.25$", "$0.25$--$0.42$"))
    axes[2].set_xlabel("Sample normalized ESS")
    axes[2].set_ylabel("Updates where PPO has lower MSE (%)")
    axes[2].set_ylim(45.0, 100.0)
    axes[2].set_title("(c) The 0.1 gate selects high-risk updates", loc="left", pad=5)
    axes[2].legend(loc="lower left", handlelength=1.8)
    for bar, row in zip(bars, on_trajectory_bin_rows):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            min(98.5, bar.get_height() + 2.0),
            f'{float(row["mean_updates_per_replication"]):.0f} updates',
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    trajectory_styles = {
        "Raw": ("#0072B2", "--", "o", "Raw"),
        "PPO masked": ("#6F6F6F", "-.", "^", "PPO"),
        "ESS fitted": ("#56B4E9", "-", "s", "Fitted ESS"),
        "ESS count transfer": ("#009E73", "-", "D", "$N\\rho$ transfer"),
        "ESS fixed 0.1": ("#D55E00", "-", "o", "Fixed 0.1"),
    }
    for method, (color, linestyle, marker, label) in trajectory_styles.items():
        rows = sorted(
            [row for row in scaled_path_rows if row["method"] == method],
            key=lambda row: float(row["rollout_batch"]),
        )
        responses_per_rollout = float(rows[0]["responses_per_rollout"])
        x = np.asarray(
            [float(row["rollout_batch"]) * responses_per_rollout / 1000.0 for row in rows]
        )
        mean = np.asarray(
            [float(row["mean_relative_improvement_pct"]) for row in rows]
        )
        interval = 1.96 * np.asarray(
            [float(row["relative_improvement_pct_se"]) for row in rows]
        )
        axes[3].plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=label,
            zorder=3,
        )
        axes[3].fill_between(
            x,
            mean - interval,
            mean + interval,
            color=color,
            alpha=0.11,
            linewidth=0.0,
        )
    axes[3].axhline(100.0, color="#8A8A8A", linewidth=0.7, linestyle=":")
    axes[3].set_xlabel("Verifier responses ($\\times10^3$)")
    axes[3].set_ylabel("Reward improvement (%)")
    axes[3].set_title("(d) Four-epoch optimization at $N=512$", loc="left", pad=5)
    axes[3].legend(loc="lower right", ncol=1, handlelength=1.8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", which="major", pad=2)
    figure.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.115,
        top=0.94,
        wspace=0.31,
        hspace=0.48,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(figure)


def make_figure(
    diagnostic_bins: list[dict[str, float | str]],
    path_rows: list[dict[str, float | str]],
    ess_threshold: float,
    output_base: Path,
    responses_per_rollout: int = 2048,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    # Render at the paper's final 6.5-inch text width so font sizes are not
    # silently reduced when LaTeX scales the figure to \linewidth.
    figure, axes = plt.subplots(1, 3, figsize=(6.5, 2.18))
    blue = "#0072B2"
    vermillion = "#D55E00"
    green = "#009E73"
    gray = "#6F6F6F"
    light_gray = "#9A9A9A"

    x = np.asarray([float(row["median_normalized_ess"]) for row in diagnostic_bins])
    mse = np.asarray([float(row["mean_raw_gradient_mse"]) for row in diagnostic_bins])
    mse_se = np.asarray([float(row["raw_gradient_mse_se"]) for row in diagnostic_bins])
    axes[0].errorbar(
        x,
        mse,
        yerr=mse_se,
        marker="o",
        color=blue,
        capsize=1.8,
        elinewidth=0.9,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )
    axes[0].axvline(ess_threshold, color=light_gray, linestyle=":", linewidth=0.9)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0065, 1.05)
    axes[0].xaxis.set_major_locator(LogLocator(base=10, numticks=3))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("Normalized ESS")
    axes[0].set_ylabel("Gradient MSE")
    axes[0].set_title("(a) Gradient estimation error", loc="left", pad=5)

    raw_gain = np.asarray(
        [float(row["mean_raw_relative_gain_pct"]) for row in diagnostic_bins]
    )
    raw_gain_se = np.asarray(
        [float(row["raw_relative_gain_pct_se"]) for row in diagnostic_bins]
    )
    oracle_gain = np.asarray(
        [float(row["mean_oracle_relative_gain_pct"]) for row in diagnostic_bins]
    )
    oracle_gain_se = np.asarray(
        [float(row["oracle_relative_gain_pct_se"]) for row in diagnostic_bins]
    )
    axes[1].errorbar(
        x,
        raw_gain,
        yerr=raw_gain_se,
        marker="o",
        color=blue,
        capsize=1.8,
        elinewidth=0.9,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Permissive",
        zorder=3,
    )
    axes[1].errorbar(
        x,
        oracle_gain,
        yerr=oracle_gain_se,
        marker="s",
        linestyle="--",
        color=green,
        capsize=1.8,
        elinewidth=0.9,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Oracle",
        zorder=2,
    )
    axes[1].axhline(0.0, color=light_gray, linewidth=0.7)
    axes[1].axvline(ess_threshold, color=light_gray, linestyle=":", linewidth=0.9)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.0065, 1.05)
    axes[1].xaxis.set_major_locator(LogLocator(base=10, numticks=3))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Normalized ESS")
    axes[1].set_ylabel("Reward change (%)")
    axes[1].set_title("(b) One-step policy improvement", loc="left", pad=5)
    axes[1].legend(loc="lower right", handlelength=1.8, borderaxespad=0.2)

    styles = {
        "Raw": (blue, "--", "s", 1.35),
        "PPO masked": (gray, "-.", "^", 1.35),
        "ESS gated": (vermillion, "-", "o", 2.0),
    }
    display_names = {
        "Raw": "Permissive",
        "PPO masked": "PPO masked",
        "ESS gated": "ESS gate",
    }
    for method in METHODS:
        rows = [row for row in path_rows if row["method"] == method]
        steps = np.asarray([float(row["rollout_batch"]) for row in rows])
        responses = steps * responses_per_rollout / 1000.0
        means = np.asarray(
            [float(row["mean_relative_improvement_pct"]) for row in rows]
        )
        errors = np.asarray(
            [float(row["relative_improvement_pct_se"]) for row in rows]
        )
        color, linestyle, marker, linewidth = styles[method]
        axes[2].plot(
            responses,
            means,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linewidth=linewidth,
            markersize=3.5 if method != "ESS gated" else 4.2,
            label=display_names[method],
            zorder=4 if method == "ESS gated" else 3,
        )
        axes[2].fill_between(
            responses,
            means - 1.96 * errors,
            means + 1.96 * errors,
            color=color,
            alpha=0.12 if method == "ESS gated" else 0.07,
            linewidth=0,
            zorder=1,
        )
        target_indices = np.flatnonzero(means >= 100.0)
        if len(target_indices):
            target_index = int(target_indices[0])
            axes[2].scatter(
                responses[target_index],
                means[target_index],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=5,
            )
            vertical_offset = {
                "Raw": 8,
                "PPO masked": -11,
                "ESS gated": 8,
            }[method]
            axes[2].annotate(
                f"{responses[target_index]:.1f}k",
                (responses[target_index], means[target_index]),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if vertical_offset > 0 else "top",
                color=color,
                fontsize=7,
            )
    axes[2].axhline(100.0, color=light_gray, linestyle=":", linewidth=0.9)
    axes[2].set_xticks([0, 4, 8, 12, 16])
    axes[2].set_xlim(-0.6, 16.9)
    axes[2].set_xlabel(r"Verifier responses ($\times 10^3$)")
    axes[2].set_ylabel("Reward improvement (%)")
    axes[2].set_title("(c) Optimization trajectory", loc="left", pad=5)
    axes[2].legend(loc="upper left", handlelength=2.0, borderaxespad=0.2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", which="major", pad=2)
    figure.subplots_adjust(left=0.074, right=0.995, bottom=0.205, top=0.89, wspace=0.52)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(figure)


def run(
    config: Config,
    skip_figure: bool = False,
    scaled_only: bool = False,
) -> None:
    if config.replications > 100:
        raise ValueError("The simulation is limited to at most 100 replications")
    if config.prompts_per_rollout % config.minibatches_per_rollout != 0:
        raise ValueError("prompts_per_rollout must divide evenly into minibatches")
    if config.responses_per_prompt < 2:
        raise ValueError("responses_per_prompt must be at least two")

    root = Path(__file__).resolve().parents[1]
    features, labels = load_optdigits(root / "simulation" / "data")
    codes = digit_codes(labels, config.response_tokens)
    initial_weights = fit_initial_policy(features, codes, config)
    initial_value, _ = population_value_and_gradient(
        initial_weights,
        features,
        codes,
    )
    results = root / "simulation" / "results"

    if scaled_only:
        formula_components = read_csv(
            results / "rlvr_formula_oracle_components.csv"
        )
        formula_thresholds = read_csv(
            results / "rlvr_formula_oracle_thresholds.csv"
        )
        formula_summary = read_csv(
            results / "rlvr_formula_oracle_summary.csv"
        )
        (
            scaled_optimization_paths,
            scaled_optimization_summary,
            scaled_on_trajectory_diagnostics,
            scaled_on_trajectory_bins,
            scaled_on_trajectory_gate_summary,
        ) = run_scaled_batch_optimization(
            initial_weights,
            features,
            codes,
            formula_thresholds,
            config,
        )
        write_csv(
            results / "rlvr_n512_optimization_paths.csv",
            scaled_optimization_paths,
        )
        write_csv(
            results / "rlvr_n512_optimization_summary.csv",
            scaled_optimization_summary,
        )
        write_csv(
            results / "rlvr_n512_on_trajectory_diagnostics.csv",
            scaled_on_trajectory_diagnostics,
        )
        write_csv(
            results / "rlvr_n512_on_trajectory_bins.csv",
            scaled_on_trajectory_bins,
        )
        write_csv(
            results / "rlvr_n512_on_trajectory_gate_summary.csv",
            scaled_on_trajectory_gate_summary,
        )
        if not skip_figure:
            make_formula_oracle_figure(
                formula_components,
                formula_thresholds,
                formula_summary,
                scaled_optimization_paths,
                scaled_on_trajectory_bins,
                root / "figures" / "ess_formula_oracle",
            )
        print(f"Initial exact population reward: {initial_value:.6f}")
        for row in scaled_optimization_summary:
            print(
                f"{row['method']}: final reward "
                f"{float(row['final_population_reward']):.6f}"
            )
        return

    values_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in METHODS
    }
    metadata_by_method: dict[str, list[dict[str, float]]] = {
        method: [] for method in METHODS
    }
    diagnostics: list[dict[str, float | str]] = []
    diagnostic_checkpoints: list[dict[str, object]] = []

    master_rng = np.random.default_rng(config.seed)
    for replication in range(config.replications):
        replication_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
        random_draws = make_replication_randomness(
            np.random.default_rng(replication_seed),
            len(features),
            config,
        )
        for method in METHODS:
            collect_diagnostics = (
                method == "ESS gated"
                and replication < config.diagnostic_replications
            )
            values, diagnostic_rows, metadata, checkpoint_rows = simulate_method(
                method,
                initial_weights,
                features,
                codes,
                random_draws,
                config,
                replication,
                collect_diagnostics,
            )
            values_by_method[method].append(values)
            metadata_by_method[method].append(metadata)
            diagnostics.extend(diagnostic_rows)
            diagnostic_checkpoints.extend(checkpoint_rows)

    path_rows = aggregate_paths(values_by_method)
    summary_rows = summarize_runs(values_by_method, metadata_by_method, config)
    diagnostic_bins = bin_diagnostics(diagnostics)
    selected_checkpoints = select_crossover_checkpoints(
        diagnostic_checkpoints,
        config,
    )
    crossover_diagnostics = run_crossover_diagnostics(
        selected_checkpoints,
        features,
        codes,
        config,
    )
    crossover_checkpoints = aggregate_crossover_checkpoints(
        crossover_diagnostics
    )
    crossover_bins = bin_crossover_checkpoints(crossover_checkpoints)
    formula_checkpoints = assign_formula_splits(
        selected_checkpoints,
        config,
    )
    formula_components = estimate_formula_components(
        formula_checkpoints,
        features,
        codes,
        config,
    )
    formula_thresholds = fit_formula_ess_thresholds(formula_components)
    formula_evaluations = run_formula_oracle_evaluation(
        formula_checkpoints,
        formula_components,
        formula_thresholds,
        features,
        codes,
        config,
    )
    formula_checkpoint_results, formula_summary = (
        aggregate_formula_oracle_evaluation(formula_evaluations)
    )
    (
        scaled_optimization_paths,
        scaled_optimization_summary,
        scaled_on_trajectory_diagnostics,
        scaled_on_trajectory_bins,
        scaled_on_trajectory_gate_summary,
    ) = run_scaled_batch_optimization(
        initial_weights,
        features,
        codes,
        formula_thresholds,
        config,
    )
    write_csv(results / "rlvr_training_paths.csv", path_rows)
    write_csv(results / "rlvr_summary.csv", summary_rows)
    write_csv(results / "rlvr_minibatch_diagnostics.csv", diagnostics)
    write_csv(results / "rlvr_diagnostic_bins.csv", diagnostic_bins)
    write_csv(
        results / "rlvr_crossover_diagnostics.csv",
        crossover_diagnostics,
    )
    write_csv(
        results / "rlvr_crossover_checkpoints.csv",
        crossover_checkpoints,
    )
    write_csv(results / "rlvr_crossover_bins.csv", crossover_bins)
    write_csv(
        results / "rlvr_formula_oracle_components.csv",
        formula_components,
    )
    write_csv(
        results / "rlvr_formula_oracle_thresholds.csv",
        formula_thresholds,
    )
    write_csv(
        results / "rlvr_formula_oracle_evaluations.csv",
        formula_evaluations,
    )
    write_csv(
        results / "rlvr_formula_oracle_checkpoints.csv",
        formula_checkpoint_results,
    )
    write_csv(
        results / "rlvr_formula_oracle_summary.csv",
        formula_summary,
    )
    write_csv(
        results / "rlvr_n512_optimization_paths.csv",
        scaled_optimization_paths,
    )
    write_csv(
        results / "rlvr_n512_optimization_summary.csv",
        scaled_optimization_summary,
    )
    write_csv(
        results / "rlvr_n512_on_trajectory_diagnostics.csv",
        scaled_on_trajectory_diagnostics,
    )
    write_csv(
        results / "rlvr_n512_on_trajectory_bins.csv",
        scaled_on_trajectory_bins,
    )
    write_csv(
        results / "rlvr_n512_on_trajectory_gate_summary.csv",
        scaled_on_trajectory_gate_summary,
    )
    if not skip_figure:
        make_figure(
            diagnostic_bins,
            path_rows,
            config.ess_threshold,
            root / "figures" / "ess_policy_validation",
            config.prompts_per_rollout * config.responses_per_prompt,
        )
        make_crossover_figure(
            crossover_bins,
            root / "figures" / "ess_estimator_crossover",
        )
        make_formula_oracle_figure(
            formula_components,
            formula_thresholds,
            formula_summary,
            scaled_optimization_paths,
            scaled_on_trajectory_bins,
            root / "figures" / "ess_formula_oracle",
        )

    print(f"Initial exact population reward: {initial_value:.6f}")
    for row in summary_rows:
        print(
            f"{row['method']}: final reward "
            f"{float(row['final_population_reward']):.6f}; relative improvement "
            f"{float(row['final_relative_improvement_pct']):.2f}%"
        )
    gate = next(row for row in summary_rows if row["method"] == "ESS gated")
    print(
        "ESS gate mean branch counts: "
        f"raw={float(gate['mean_gate_raw_updates']):.2f}, "
        f"masked={float(gate['mean_gate_masked_updates']):.2f}"
    )
    print(
        "Fixed-checkpoint crossover diagnostic: "
        f"{len(selected_checkpoints)} checkpoints, "
        f"{config.crossover_draws_per_checkpoint} paired redraws per checkpoint"
    )
    print("Formula-oracle ESS thresholds:")
    for row in formula_thresholds:
        print(
            f"  N={int(float(row['sequences_per_estimator']))}: "
            f"fitted={min(1.0, float(row['fitted_ess_threshold'])):.3f}, "
            "effective-count transfer="
            f"{min(1.0, float(row['effective_count_threshold'])):.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--diagnostic-replications", type=int, default=20)
    parser.add_argument("--crossover-checkpoints-per-bin", type=int, default=8)
    parser.add_argument("--crossover-draws-per-checkpoint", type=int, default=32)
    parser.add_argument("--crossover-minibatches-per-draw", type=int, default=4)
    parser.add_argument("--formula-calibration-blocks", type=int, default=100)
    parser.add_argument("--formula-evaluation-draws", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--policy-learning-rate", type=float, default=3.0)
    parser.add_argument("--response-tokens", type=int, default=16)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--scaled-only", action="store_true")
    arguments = parser.parse_args()
    run(
        replace(
            Config(),
            replications=arguments.replications,
            diagnostic_replications=min(
                arguments.diagnostic_replications,
                arguments.replications,
            ),
            crossover_checkpoints_per_bin=(
                arguments.crossover_checkpoints_per_bin
            ),
            crossover_draws_per_checkpoint=(
                arguments.crossover_draws_per_checkpoint
            ),
            crossover_minibatches_per_draw=(
                arguments.crossover_minibatches_per_draw
            ),
            formula_calibration_blocks=arguments.formula_calibration_blocks,
            formula_evaluation_draws=arguments.formula_evaluation_draws,
            rollout_steps=arguments.rollout_steps,
            policy_learning_rate=arguments.policy_learning_rate,
            response_tokens=arguments.response_tokens,
            responses_per_prompt=arguments.responses_per_prompt,
        ),
        skip_figure=arguments.skip_figure,
        scaled_only=arguments.scaled_only,
    )
