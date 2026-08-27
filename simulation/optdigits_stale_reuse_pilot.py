"""Pilot repeated optimization epochs over one fixed full Optdigits rollout."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale


SEED_START = 20900826


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features, labels = stale.load_training_split(root)
    _, _, eta_max = stale.global_smoothness_bound(features)
    rows: list[dict[str, float]] = []

    batch_sizes = (64, 128)
    epochs_grid = (2, 4)
    learning_rates = (0.14, min(0.17, 0.98 * eta_max))
    initialization_scale = 0.35
    ppo_epsilon = 0.20
    replications = 4

    for batch_size in batch_sizes:
        for epochs in epochs_grid:
            for learning_rate in learning_rates:
                method_finals = {name: [] for name in ("raw", "ppo", "oracle")}
                oracle_ppo = []
                oracle_distinct = []
                oracle_ppo_distinct = []
                minimum_rho = []
                for replication in range(replications):
                    config = base.Config(
                        initialization_scale=initialization_scale,
                        training_learning_rate=learning_rate,
                        ppo_epsilon=ppo_epsilon,
                    )
                    initial_weights = base.fit_initial_policy(features, labels, config)
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
                    for method in ("raw", "ppo", "oracle"):
                        _, summary = stale.run_fixed_rollout(
                            method,
                            None,
                            initial_weights,
                            features,
                            labels,
                            rollout,
                            long_order,
                            batch_size,
                            learning_rate,
                            ppo_epsilon,
                            replication,
                        )
                        method_finals[method].append(float(summary["final_value"]))
                        if method == "oracle":
                            oracle_ppo.append(float(summary["ppo_fraction"]))
                            oracle_distinct.append(float(summary["distinct_fraction"]))
                            oracle_ppo_distinct.append(float(summary["ppo_given_distinct"]))
                            minimum_rho.append(float(summary["minimum_population_rho"]))

                raw = float(np.mean(method_finals["raw"]))
                ppo = float(np.mean(method_finals["ppo"]))
                oracle = float(np.mean(method_finals["oracle"]))
                mix = float(np.mean(oracle_ppo_distinct))
                gain = oracle - max(raw, ppo)
                mixing_bonus = max(0.0, 1.0 - 2.0 * abs(mix - 0.5))
                rows.append(
                    {
                        "batch_size": float(batch_size),
                        "epochs": float(epochs),
                        "updates": float(math.ceil(len(features) / batch_size) * epochs),
                        "learning_rate": learning_rate,
                        "eta_max": eta_max,
                        "initialization_scale": initialization_scale,
                        "replications": float(replications),
                        "raw_final": raw,
                        "ppo_final": ppo,
                        "oracle_final": oracle,
                        "oracle_gain_vs_best_static": gain,
                        "oracle_ppo_fraction": float(np.mean(oracle_ppo)),
                        "oracle_distinct_fraction": float(np.mean(oracle_distinct)),
                        "oracle_ppo_given_distinct": mix,
                        "minimum_population_rho": float(np.mean(minimum_rho)),
                        "score": 100.0 * gain + mixing_bonus,
                    }
                )
                print(rows[-1])

    rows.sort(key=lambda row: row["score"], reverse=True)
    write_csv(root / "simulation" / "results" / "optdigits_stale_reuse_pilot.csv", rows)


if __name__ == "__main__":
    main()
