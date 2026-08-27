"""Fast scan for a stale-rollout Raw-to-PPO crossover using one fixed rollout."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale


SEED_START = 21000826


def run_curve(
    method: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    rollout: dict[str, np.ndarray],
    long_order: np.ndarray,
    batch_size: int,
    learning_rate: float,
    ppo_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    config = base.Config(training_learning_rate=learning_rate, ppo_epsilon=ppo_epsilon)
    weights = initial_weights.copy()
    rollout_weights = initial_weights.copy()
    values = [base.population_value(weights, features, labels)]
    rhos = [1.0]
    for indices in stale.chunks(long_order, batch_size):
        gradients, _, _ = base.estimate_gradients(weights, rollout, indices, config)
        weights = weights + learning_rate * gradients[method]
        values.append(base.population_value(weights, features, labels))
        rhos.append(base.population_rho(weights, rollout_weights, features))
    return np.asarray(values), np.asarray(rhos)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features, labels = stale.load_training_split(root)
    _, _, eta_max = stale.global_smoothness_bound(features)
    learning_rate = min(0.17, 0.98 * eta_max)
    ppo_epsilon = 0.20
    replications = 3
    rows = []

    for init_scale in (0.05, 0.10, 0.20, 0.35):
        config = base.Config(
            initialization_scale=init_scale,
            training_learning_rate=learning_rate,
            ppo_epsilon=ppo_epsilon,
        )
        initial_weights = base.fit_initial_policy(features, labels, config)
        for batch_size in (16, 32, 64):
            for epochs in (1, 2, 4, 8):
                raw_curves = []
                ppo_curves = []
                raw_rhos = []
                ppo_rhos = []
                for replication in range(replications):
                    rng = np.random.default_rng(SEED_START + replication)
                    rollout = base.collect_rollout(
                        initial_weights,
                        features,
                        labels,
                        np.arange(len(features), dtype=int),
                        rng.random(len(features)),
                    )
                    long_order = np.concatenate(
                        [rng.permutation(len(features)) for _ in range(epochs)]
                    )
                    raw, raw_rho = run_curve(
                        "raw", initial_weights, features, labels, rollout, long_order,
                        batch_size, learning_rate, ppo_epsilon,
                    )
                    ppo, ppo_rho = run_curve(
                        "ppo", initial_weights, features, labels, rollout, long_order,
                        batch_size, learning_rate, ppo_epsilon,
                    )
                    raw_curves.append(raw)
                    ppo_curves.append(ppo)
                    raw_rhos.append(raw_rho)
                    ppo_rhos.append(ppo_rho)

                raw_curves = np.asarray(raw_curves)
                ppo_curves = np.asarray(ppo_curves)
                raw_mean = np.mean(raw_curves, axis=0)
                ppo_mean = np.mean(ppo_curves, axis=0)
                gap = raw_mean - ppo_mean
                total_updates = len(gap) - 1
                early_end = max(2, total_updates // 4)
                max_early = float(np.max(gap[1 : early_end + 1]))
                max_early_update = int(np.argmax(gap[1 : early_end + 1]) + 1)
                persistent_cross = -1
                for update in range(max_early_update + 1, total_updates + 1):
                    if np.all(gap[update:] < 0.0):
                        persistent_cross = update
                        break
                final_gap_rep = raw_curves[:, -1] - ppo_curves[:, -1]
                final_gap = float(np.mean(final_gap_rep))
                final_gap_se = stale.standard_error(final_gap_rep)
                min_raw_rho = float(np.mean(np.min(np.asarray(raw_rhos), axis=1)))
                min_ppo_rho = float(np.mean(np.min(np.asarray(ppo_rhos), axis=1)))
                transition = float(
                    max_early > 0.003 and final_gap < -0.003 and persistent_cross > 0
                )
                score = (
                    50.0 * max_early
                    + 50.0 * max(0.0, -final_gap)
                    + 2.0 * max(0.0, 0.8 - min_raw_rho)
                    + transition
                )
                row = {
                    "initialization_scale": init_scale,
                    "batch_size": float(batch_size),
                    "epochs": float(epochs),
                    "updates": float(total_updates),
                    "learning_rate": learning_rate,
                    "eta_max": eta_max,
                    "replications": float(replications),
                    "initial_value": float(raw_mean[0]),
                    "raw_final": float(raw_mean[-1]),
                    "ppo_final": float(ppo_mean[-1]),
                    "final_raw_minus_ppo": final_gap,
                    "final_gap_se": final_gap_se,
                    "max_early_raw_advantage": max_early,
                    "max_early_update": float(max_early_update),
                    "persistent_crossover_update": float(persistent_cross),
                    "minimum_raw_population_rho": min_raw_rho,
                    "minimum_ppo_population_rho": min_ppo_rho,
                    "transition_visible": transition,
                    "score": score,
                }
                rows.append(row)
                print(row)

    rows.sort(key=lambda row: row["score"], reverse=True)
    path = root / "simulation" / "results" / "optdigits_stale_static_scan.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
