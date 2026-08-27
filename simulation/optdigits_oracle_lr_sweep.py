from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from optdigits_categorical_theory import (
    Config,
    common_randomness,
    fit_initial_policy,
    load_optdigits,
    run_trajectory,
    standard_error,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = Config(replications=12)
    features, labels = load_optdigits(root / "simulation" / "data")
    initial_weights = fit_initial_policy(features, labels, base)
    rows = []
    for learning_rate in (0.25, 0.5, 1.0, 2.0, 3.0):
        config = replace(base, training_learning_rate=learning_rate)
        outcomes = {method: [] for method in ("raw", "ppo", "mse_oracle")}
        for replication in range(config.replications):
            rng = np.random.default_rng(config.seed + replication)
            draws = common_randomness(rng, len(features), config)
            for method in outcomes:
                _, _, summary = run_trajectory(
                    method,
                    initial_weights,
                    features,
                    labels,
                    draws,
                    config,
                    replication,
                    0,
                )
                outcomes[method].append(summary["final_value"])
        for method, values in outcomes.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "method": method,
                    "replications": len(array),
                    "mean_final_value": float(np.mean(array)),
                    "se_final_value": standard_error(array),
                    "median_final_value": float(np.median(array)),
                }
            )
    output = root / "simulation" / "results" / "optdigits_oracle_lr_sweep.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
