"""RLVR-style simulation of ESS-gated policy optimization.

Optdigits images serve as prompts.  A policy emits an eight-token binary
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
    rollout_steps: int = 8
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
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    weights = initial_weights.copy()
    initial_value, _ = population_value_and_gradient(weights, features, codes)
    values = [initial_value]
    diagnostic_rows: list[dict[str, float]] = []
    gate_raw_updates = 0
    gate_masked_updates = 0
    all_ess: list[float] = []

    for rollout_index, random_draw in enumerate(random_draws, start=1):
        rollout = collect_rollout(
            weights,
            features,
            codes,
            random_draw,
        )
        minibatches = np.split(
            rollout["prompt_order"],
            config.minibatches_per_rollout,
        )
        for minibatch_index, group_indices in enumerate(minibatches, start=1):
            raw_gradient, masked_gradient, normalized_ess, ratios = (
                minibatch_gradients(
                    weights,
                    rollout,
                    group_indices,
                    config,
                )
            )
            all_ess.append(normalized_ess)

            if collect_diagnostics:
                value_before, true_gradient = population_value_and_gradient(
                    weights,
                    features,
                    codes,
                )
                raw_value, _ = population_value_and_gradient(
                    weights + config.policy_learning_rate * raw_gradient,
                    features,
                    codes,
                )
                oracle_value, _ = population_value_and_gradient(
                    weights + config.policy_learning_rate * true_gradient,
                    features,
                    codes,
                )
                diagnostic_rows.append(
                    {
                        "replication": float(replication),
                        "rollout_batch": float(rollout_index),
                        "minibatch": float(minibatch_index),
                        "normalized_ess": normalized_ess,
                        "effective_sequences": normalized_ess * len(ratios),
                        "mean_ratio": float(np.mean(ratios)),
                        "max_ratio": float(np.max(ratios)),
                        "raw_gradient_mse": float(
                            np.sum((raw_gradient - true_gradient) ** 2)
                        ),
                        "raw_relative_gain_pct": float(
                            100.0 * (raw_value - value_before) / value_before
                        ),
                        "oracle_relative_gain_pct": float(
                            100.0 * (oracle_value - value_before) / value_before
                        ),
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
    return np.asarray(values), diagnostic_rows, metadata


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_paths(
    values_by_method: dict[str, list[np.ndarray]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method in METHODS:
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
) -> list[dict[str, float | str]]:
    final_values = {
        method: np.asarray([path[-1] for path in values_by_method[method]])
        for method in METHODS
    }
    initial_values = np.asarray([path[0] for path in values_by_method["Raw"]])
    rows: list[dict[str, float | str]] = []
    for method in METHODS:
        relative = 100.0 * (
            final_values[method] - initial_values
        ) / initial_values
        gate_minus_raw = final_values[method] - final_values["Raw"]
        gate_minus_ppo = final_values[method] - final_values["PPO masked"]
        metadata = metadata_by_method[method]
        rows.append(
            {
                "method": method,
                "final_population_reward": float(np.mean(final_values[method])),
                "final_population_reward_se": standard_error(final_values[method]),
                "final_relative_improvement_pct": float(np.mean(relative)),
                "final_relative_improvement_pct_se": standard_error(relative),
                "paired_difference_from_raw": float(np.mean(gate_minus_raw)),
                "paired_difference_from_raw_se": standard_error(gate_minus_raw),
                "paired_difference_from_ppo": float(np.mean(gate_minus_ppo)),
                "paired_difference_from_ppo_se": standard_error(gate_minus_ppo),
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
                "prompts_per_rollout": float(config.prompts_per_rollout),
                "responses_per_prompt": float(config.responses_per_prompt),
                "response_tokens": float(config.response_tokens),
                "policy_learning_rate": config.policy_learning_rate,
                "ppo_epsilon": config.ppo_epsilon,
                "ess_threshold": config.ess_threshold,
            }
        )
    return rows


def bin_diagnostics(
    diagnostic_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    ess_values = np.asarray(
        [float(row["normalized_ess"]) for row in diagnostic_rows]
    )
    boundaries = np.asarray([0.0, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000001])
    output: list[dict[str, float]] = []
    for index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
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
        raw_gain = np.asarray(
            [float(diagnostic_rows[i]["raw_relative_gain_pct"]) for i in chunk]
        )
        oracle_gain = np.asarray(
            [float(diagnostic_rows[i]["oracle_relative_gain_pct"]) for i in chunk]
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
                "mean_raw_relative_gain_pct": float(np.mean(raw_gain)),
                "raw_relative_gain_pct_se": standard_error(raw_gain),
                "mean_oracle_relative_gain_pct": float(np.mean(oracle_gain)),
                "oracle_relative_gain_pct_se": standard_error(oracle_gain),
                "observations": float(len(chunk)),
            }
        )
    return output


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

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.85))

    x = np.asarray([float(row["median_normalized_ess"]) for row in diagnostic_bins])
    mse = np.asarray([float(row["mean_raw_gradient_mse"]) for row in diagnostic_bins])
    mse_se = np.asarray([float(row["raw_gradient_mse_se"]) for row in diagnostic_bins])
    axes[0].errorbar(x, mse, yerr=mse_se, marker="o", color="#0072B2", capsize=2)
    axes[0].axvline(ess_threshold, color="0.35", linestyle=":", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Normalized sequence ESS")
    axes[0].set_ylabel("Raw-gradient MSE")
    axes[0].set_title("(a) ESS predicts gradient error")

    raw_gain = np.asarray(
        [float(row["mean_raw_relative_gain_pct"]) for row in diagnostic_bins]
    )
    raw_gain_se = np.asarray(
        [float(row["raw_relative_gain_pct_se"]) for row in diagnostic_bins]
    )
    oracle_gain = np.asarray(
        [float(row["mean_oracle_relative_gain_pct"]) for row in diagnostic_bins]
    )
    axes[1].errorbar(
        x,
        raw_gain,
        yerr=raw_gain_se,
        marker="o",
        color="#D55E00",
        capsize=2,
        label="Raw estimate",
    )
    axes[1].plot(x, oracle_gain, marker="s", color="#009E73", label="Exact gradient")
    axes[1].axhline(0.0, color="0.55", linewidth=0.8)
    axes[1].axvline(ess_threshold, color="0.35", linestyle=":", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Normalized sequence ESS")
    axes[1].set_ylabel("One-update reward change (%)")
    axes[1].set_title("(b) Error changes update quality")
    axes[1].legend(frameon=False)

    styles = {
        "Raw": ("#D55E00", "--"),
        "PPO masked": ("#0072B2", "-.") ,
        "ESS gated": ("#009E73", "-"),
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
        color, linestyle = styles[method]
        axes[2].plot(
            responses,
            means,
            color=color,
            linestyle=linestyle,
            label=method,
        )
        axes[2].fill_between(
            responses,
            means - 1.96 * errors,
            means + 1.96 * errors,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        target_indices = np.flatnonzero(means >= 100.0)
        if len(target_indices):
            target_index = int(target_indices[0])
            axes[2].scatter(
                responses[target_index],
                means[target_index],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
            vertical_offset = {
                "Raw": 9,
                "PPO masked": -15,
                "ESS gated": 9,
            }[method]
            axes[2].annotate(
                f"{responses[target_index]:.1f}k",
                (responses[target_index], means[target_index]),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if vertical_offset > 0 else "top",
                color=color,
                fontsize=7.5,
            )
    axes[2].axhline(100.0, color="0.45", linestyle=":", linewidth=0.9)
    axes[2].set_xlabel("Verifier responses processed (thousands)")
    axes[2].set_ylabel("Relative reward improvement (%)")
    axes[2].set_title("(c) ESS gate improves sample efficiency")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.15, linewidth=0.5)
    figure.tight_layout(w_pad=1.2)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(config: Config, skip_figure: bool = False) -> None:
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

    values_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in METHODS
    }
    metadata_by_method: dict[str, list[dict[str, float]]] = {
        method: [] for method in METHODS
    }
    diagnostics: list[dict[str, float | str]] = []

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
            values, diagnostic_rows, metadata = simulate_method(
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

    path_rows = aggregate_paths(values_by_method)
    summary_rows = summarize_runs(values_by_method, metadata_by_method, config)
    diagnostic_bins = bin_diagnostics(diagnostics)
    results = root / "simulation" / "results"
    write_csv(results / "rlvr_training_paths.csv", path_rows)
    write_csv(results / "rlvr_summary.csv", summary_rows)
    write_csv(results / "rlvr_minibatch_diagnostics.csv", diagnostics)
    write_csv(results / "rlvr_diagnostic_bins.csv", diagnostic_bins)
    if not skip_figure:
        make_figure(
            diagnostic_bins,
            path_rows,
            config.ess_threshold,
            root / "figures" / "ess_policy_validation",
            config.prompts_per_rollout * config.responses_per_prompt,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--diagnostic-replications", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--policy-learning-rate", type=float, default=3.0)
    parser.add_argument("--response-tokens", type=int, default=16)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--skip-figure", action="store_true")
    arguments = parser.parse_args()
    run(
        replace(
            Config(),
            replications=arguments.replications,
            diagnostic_replications=min(
                arguments.diagnostic_replications,
                arguments.replications,
            ),
            rollout_steps=arguments.rollout_steps,
            policy_learning_rate=arguments.policy_learning_rate,
            response_tokens=arguments.response_tokens,
            responses_per_prompt=arguments.responses_per_prompt,
        ),
        skip_figure=arguments.skip_figure,
    )
