"""Reproduce the contextual-bandit ESS and ratio-truncation study.

The environment is a bounded linear contextual bandit with Bernoulli rewards
and a softmax policy.  Its finite context and action sets let us enumerate the
population gradient, ESS, bias, covariance, and MSE exactly.  Monte Carlo is
used only for the paired one-step intervention.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    seed: int = 17
    contexts: int = 96
    actions: int = 8
    features: int = 6
    batch_size: int = 32
    repetitions: int = 2000
    step_size: float = 6.0
    reuse_steps: int = 8
    max_drift: float = 18.0
    drift_points: int = 19


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def policy(theta: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    return softmax(contexts @ theta.T)


def policy_value(theta: np.ndarray, contexts: np.ndarray, reward_mean: np.ndarray) -> float:
    return float(np.mean(np.sum(policy(theta, contexts) * reward_mean, axis=1)))


def batched_policy_value(
    theta: np.ndarray, contexts: np.ndarray, reward_mean: np.ndarray
) -> np.ndarray:
    logits = np.einsum("rkd,md->rmk", theta, contexts)
    probabilities = softmax(logits, axis=2)
    return np.mean(np.sum(probabilities * reward_mean[None, :, :], axis=2), axis=1)


def make_problem(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    contexts = rng.normal(size=(config.contexts, config.features))
    contexts /= np.maximum(np.linalg.norm(contexts, axis=1, keepdims=True), 1.0)

    behavior_theta = 0.85 * rng.normal(size=(config.actions, config.features))
    reward_theta = rng.normal(size=(config.actions, config.features))
    reward_theta /= np.linalg.norm(reward_theta, axis=1, keepdims=True)
    reward_mean = 0.5 + 0.4 * (contexts @ reward_theta.T)

    if np.max(np.linalg.norm(contexts, axis=1)) > 1.0 + 1e-12:
        raise AssertionError("Context norm exceeds its declared bound.")
    if np.min(reward_mean) < 0.0 or np.max(reward_mean) > 1.0:
        raise AssertionError("Bernoulli reward mean is outside [0, 1].")

    # Most drift is orthogonal to the reward model. This represents policy
    # movement that changes coverage without making every large ratio useful.
    # A smaller reward-aligned component keeps the target path non-degenerate.
    reward_direction = reward_theta / np.linalg.norm(reward_theta)
    nuisance = rng.normal(size=reward_theta.shape)
    nuisance -= np.sum(nuisance * reward_direction) * reward_direction
    nuisance /= np.linalg.norm(nuisance)
    direction = 0.25 * reward_direction + np.sqrt(1.0 - 0.25**2) * nuisance
    return contexts, behavior_theta, reward_mean, direction


def score_tensor(probabilities: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    context_count, action_count = probabilities.shape
    eye = np.eye(action_count)
    action_residual = eye[None, :, :] - probabilities[:, None, :]
    return action_residual[:, :, :, None] * contexts[:, None, None, :]


def exact_metrics(
    behavior: np.ndarray,
    target: np.ndarray,
    reward_mean: np.ndarray,
    contexts: np.ndarray,
    batch_size: int,
    caps: tuple[float | None, ...],
) -> dict[str, dict[str, float | np.ndarray]]:
    context_count, action_count = behavior.shape
    baseline = np.sum(behavior * reward_mean, axis=1)
    scores = score_tensor(target, contexts)
    ratios = target / behavior
    qa_probability = behavior / context_count

    mean_advantage = reward_mean - baseline[:, None]
    second_advantage = (
        reward_mean * (1.0 - baseline[:, None]) ** 2
        + (1.0 - reward_mean) * baseline[:, None] ** 2
    )
    mean_h = mean_advantage[:, :, None, None] * scores
    norm_score_sq = np.sum(scores**2, axis=(2, 3))
    second_h_norm = second_advantage * norm_score_sq

    raw_mean = np.sum(
        qa_probability[:, :, None, None] * ratios[:, :, None, None] * mean_h,
        axis=(0, 1),
    )
    ew2 = float(np.sum(qa_probability * ratios**2))
    raw_second = float(np.sum(qa_probability * ratios**2 * second_h_norm))
    rho = 1.0 / ew2
    g2 = raw_second / ew2

    output: dict[str, dict[str, float | np.ndarray]] = {}
    for cap in caps:
        name = "raw" if cap is None else f"cap_{cap:g}"
        coefficient = ratios if cap is None else np.minimum(ratios, cap)
        estimator_mean = np.sum(
            qa_probability[:, :, None, None] * coefficient[:, :, None, None] * mean_h,
            axis=(0, 1),
        )
        second_moment = float(np.sum(qa_probability * coefficient**2 * second_h_norm))
        bias_sq = float(np.sum((estimator_mean - raw_mean) ** 2))
        trace_covariance = second_moment - float(np.sum(estimator_mean**2))
        mse = bias_sq + trace_covariance / batch_size
        tail_probability = 0.0
        tail_excess = 0.0
        if cap is not None:
            tail_probability = float(np.sum(qa_probability * (ratios > cap)))
            tail_excess = float(np.sum(qa_probability * np.maximum(ratios - cap, 0.0)))
        output[name] = {
            "mean": estimator_mean,
            "mse": mse,
            "bias_sq": bias_sq,
            "variance_over_n": trace_covariance / batch_size,
            "tail_probability": tail_probability,
            "tail_excess": tail_excess,
        }

    output["population"] = {
        "gradient": raw_mean,
        "rho": rho,
        "effective_count": batch_size * rho,
        "g2": g2,
        "raw_mse_identity": (g2 / rho - float(np.sum(raw_mean**2))) / batch_size,
        "max_ratio": float(np.max(ratios)),
    }
    return output


def paired_intervention(
    rng: np.random.Generator,
    behavior: np.ndarray,
    reward_mean: np.ndarray,
    contexts: np.ndarray,
    theta: np.ndarray,
    batch_size: int,
    repetitions: int,
    step_size: float,
    reuse_steps: int,
    caps: tuple[float | None, ...],
) -> dict[str, dict[str, float]]:
    context_count, action_count = behavior.shape
    context_index = rng.integers(0, context_count, size=(repetitions, batch_size))
    sampled_behavior = behavior[context_index]
    uniforms = rng.random(size=(repetitions, batch_size, 1))
    action = np.sum(uniforms > np.cumsum(sampled_behavior, axis=2), axis=2)
    sampled_reward_mean = reward_mean[context_index, action]
    reward = (rng.random(size=(repetitions, batch_size)) < sampled_reward_mean).astype(float)

    baseline = np.sum(behavior * reward_mean, axis=1)
    repetition_index = np.arange(repetitions)[:, None]
    batch_index = np.arange(batch_size)[None, :]
    sampled_context = contexts[context_index]
    advantage = reward - baseline[context_index]

    start_value = policy_value(theta, contexts, reward_mean)
    results: dict[str, dict[str, float]] = {}
    for cap in caps:
        name = "raw" if cap is None else f"cap_{cap:g}"
        updated_theta = np.broadcast_to(theta, (repetitions,) + theta.shape).copy()
        for _ in range(reuse_steps):
            logits = np.einsum("rkd,rnd->rnk", updated_theta, sampled_context)
            current_policy = softmax(logits, axis=2)
            selected_probability = current_policy[repetition_index, batch_index, action]
            sampled_ratio = selected_probability / sampled_behavior[
                repetition_index, batch_index, action
            ]
            residual = -current_policy
            residual[repetition_index, batch_index, action] += 1.0
            score = residual[:, :, :, None] * sampled_context[:, :, None, :]
            h = advantage[:, :, None, None] * score
            coefficient = sampled_ratio if cap is None else np.minimum(sampled_ratio, cap)
            gradient = np.mean(coefficient[:, :, None, None] * h, axis=1)
            updated_theta += step_size * gradient
        improvement = batched_policy_value(updated_theta, contexts, reward_mean) - start_value
        standard_error = float(np.std(improvement, ddof=1) / np.sqrt(repetitions))
        harm_probability = float(np.mean(improvement < 0.0))
        harm_standard_error = np.sqrt(
            harm_probability * (1.0 - harm_probability) / repetitions
        )
        results[name] = {
            "improvement_mean": float(np.mean(improvement)),
            "improvement_ci95": 1.96 * standard_error,
            "harm_probability": harm_probability,
            "harm_ci95": 1.96 * harm_standard_error,
        }
    return results


def write_csv(rows: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, float]], output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rho = np.array([row["rho"] for row in rows])
    colors = {"raw": "#20242b", "cap_3": "#0072B2", "cap_5": "#D55E00"}
    labels = {"raw": "Untruncated", "cap_3": "Upper cap 3", "cap_5": "Upper cap 5"}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)

    for method in ("raw", "cap_3", "cap_5"):
        axes[0].plot(
            rho,
            [row[f"{method}_mse"] for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[method],
            label=labels[method],
        )
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlim(max(rho) * 1.03, min(rho) * 0.97)
    axes[0].set_title("(a) Exact gradient MSE")
    axes[0].set_xlabel("Normalized sequence ESS, $\\rho$")
    axes[0].set_ylabel("MSE, log scale")
    axes[0].legend(frameon=False)

    for method in ("cap_3", "cap_5"):
        axes[1].plot(
            rho,
            [row[f"{method}_bias_sq"] for row in rows],
            linestyle="--",
            linewidth=1.6,
            color=colors[method],
            label=f"{labels[method]} bias$^2$",
        )
        axes[1].plot(
            rho,
            [row[f"{method}_variance_over_n"] for row in rows],
            linewidth=1.6,
            color=colors[method],
            label=f"{labels[method]} variance/$N$",
        )
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_xlim(max(rho) * 1.03, min(rho) * 0.97)
    axes[1].set_title("(b) Truncation decomposition")
    axes[1].set_xlabel("Normalized sequence ESS, $\\rho$")
    axes[1].set_ylabel("Risk component, log scale")
    axes[1].legend(frameon=False, ncol=1)

    for method in ("raw", "cap_3", "cap_5"):
        mean = np.array([row[f"{method}_harm_probability"] for row in rows])
        ci = np.array([row[f"{method}_harm_ci95"] for row in rows])
        axes[2].plot(
            rho,
            mean,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[method],
            label=labels[method],
        )
        axes[2].fill_between(
            rho,
            np.maximum(mean - ci, 0.0),
            np.minimum(mean + ci, 1.0),
            color=colors[method],
            alpha=0.14,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlim(max(rho) * 1.03, min(rho) * 0.97)
    axes[2].set_title("(c) Harm after eight reused updates")
    axes[2].set_xlabel("Initial normalized sequence ESS, $\\rho$")
    axes[2].set_ylabel("Probability of reward decrease")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(True, which="major", color="#dddddd", linewidth=0.6)
        axis.grid(True, which="minor", color="#eeeeee", linewidth=0.4)

    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--repetitions", type=int, default=Config.repetitions)
    args = parser.parse_args()

    config = Config(repetitions=args.repetitions)
    contexts, behavior_theta, reward_mean, direction = make_problem(config)
    behavior = policy(behavior_theta, contexts)
    caps: tuple[float | None, ...] = (None, 3.0, 5.0)
    drift_values = np.linspace(0.0, config.max_drift, config.drift_points)
    intervention_rng = np.random.default_rng(config.seed + 1)

    rows: list[dict[str, float]] = []
    for drift in drift_values:
        theta = behavior_theta + drift * direction
        target = policy(theta, contexts)
        exact = exact_metrics(
            behavior,
            target,
            reward_mean,
            contexts,
            config.batch_size,
            caps,
        )
        intervention = paired_intervention(
            intervention_rng,
            behavior,
            reward_mean,
            contexts,
            theta,
            config.batch_size,
            config.repetitions,
            config.step_size,
            config.reuse_steps,
            caps,
        )
        population = exact["population"]
        row: dict[str, float] = {
            "drift": float(drift),
            "batch_size": float(config.batch_size),
            "repetitions": float(config.repetitions),
            "step_size": float(config.step_size),
            "reuse_steps": float(config.reuse_steps),
            "rho": float(population["rho"]),
            "effective_count": float(population["effective_count"]),
            "g2": float(population["g2"]),
            "max_ratio": float(population["max_ratio"]),
            "raw_mse_identity": float(population["raw_mse_identity"]),
        }
        for method in ("raw", "cap_3", "cap_5"):
            row[f"{method}_mse"] = float(exact[method]["mse"])
            row[f"{method}_bias_sq"] = float(exact[method]["bias_sq"])
            row[f"{method}_variance_over_n"] = float(exact[method]["variance_over_n"])
            row[f"{method}_tail_probability"] = float(exact[method]["tail_probability"])
            row[f"{method}_tail_excess"] = float(exact[method]["tail_excess"])
            row[f"{method}_improvement_mean"] = intervention[method]["improvement_mean"]
            row[f"{method}_improvement_ci95"] = intervention[method]["improvement_ci95"]
            row[f"{method}_harm_probability"] = intervention[method]["harm_probability"]
            row[f"{method}_harm_ci95"] = intervention[method]["harm_ci95"]
        rows.append(row)

    csv_path = args.output_dir / "contextual_bandit_results.csv"
    figure_stem = args.figure_dir / "contextual_bandit_ess"
    write_csv(rows, csv_path)
    make_figure(rows, figure_stem)

    identity_error = max(abs(row["raw_mse"] - row["raw_mse_identity"]) for row in rows)
    if identity_error > 1e-12:
        raise AssertionError("Exact MSE identity failed its numerical check.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_stem.with_suffix('.pdf')}")
    print(f"Wrote {figure_stem.with_suffix('.png')}")
    print(f"Maximum raw-MSE identity error: {identity_error:.3e}")


if __name__ == "__main__":
    main()
