"""Targeted pilot search for an Optdigits regime where the exact MSE oracle
uses both unmodified and PPO estimators and improves over both static rules.

All candidate learning rates satisfy the global smoothness certificate. The
search uses pilot seeds only and is not a reported result.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base


LEARNING_RATE = 0.17
ROLLOUT_SIZE = 320
POLICY_ITERATIONS = 25
PILOT_REPLICATIONS = 12
SEED_START = 20700826


@dataclass(frozen=True)
class Candidate:
    name: str
    minibatches: int
    epsilon: float
    initialization_scale: float


CANDIDATES = (
    Candidate("N40_eps005_init035", 8, 0.05, 0.35),
    Candidate("N40_eps010_init035", 8, 0.10, 0.35),
    Candidate("N20_eps010_init035", 16, 0.10, 0.35),
    Candidate("N20_eps020_init035", 16, 0.20, 0.35),
    Candidate("N16_eps010_init035", 20, 0.10, 0.35),
    Candidate("N16_eps020_init035", 20, 0.20, 0.35),
    Candidate("N20_eps020_init020", 16, 0.20, 0.20),
    Candidate("N16_eps020_init020", 20, 0.20, 0.20),
)


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def run_static(
    method: str,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: base.Config,
) -> np.ndarray:
    weights = initial_weights.copy()
    values = [base.population_value(weights, features, labels)]
    for draw in draws:
        rollout_weights = weights.copy()
        rollout = base.collect_rollout(
            rollout_weights,
            features,
            labels,
            draw["contexts"],
            draw["uniforms"],
        )
        for indices in np.split(draw["order"], config.minibatches):
            gradients, _, _ = base.estimate_gradients(weights, rollout, indices, config)
            weights = weights + config.training_learning_rate * gradients[method]
        values.append(base.population_value(weights, features, labels))
    return np.asarray(values)


def run_oracle(
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    draws: list[dict[str, np.ndarray]],
    config: base.Config,
) -> tuple[np.ndarray, float, float, float]:
    weights = initial_weights.copy()
    values = [base.population_value(weights, features, labels)]
    ppo = 0
    raw_distinct = 0
    distinct = 0
    total = 0
    for draw in draws:
        rollout_weights = weights.copy()
        rollout = base.collect_rollout(
            rollout_weights,
            features,
            labels,
            draw["contexts"],
            draw["uniforms"],
        )
        for indices in np.split(draw["order"], config.minibatches):
            gradients, _, _ = base.estimate_gradients(weights, rollout, indices, config)
            total += 1
            if np.array_equal(gradients["raw"], gradients["ppo"]):
                selected = "raw"
            else:
                distinct += 1
                risks = base.exact_estimator_risks(
                    weights,
                    rollout_weights,
                    features,
                    labels,
                    len(indices),
                    config,
                )
                if risks["ppo_risk"] < risks["raw_risk"]:
                    selected = "ppo"
                    ppo += 1
                else:
                    selected = "raw"
                    raw_distinct += 1
            weights = weights + config.training_learning_rate * gradients[selected]
        values.append(base.population_value(weights, features, labels))
    return (
        np.asarray(values),
        ppo / max(total, 1),
        raw_distinct / max(total, 1),
        distinct / max(total, 1),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features, labels = base.load_optdigits(root / "simulation" / "data", False)
    covariance = features.T @ features / len(features)
    eta_max = 2.0 / float(np.linalg.eigvalsh(covariance)[-1])
    if LEARNING_RATE > eta_max:
        raise RuntimeError("pilot learning rate violates smoothness certificate")

    rows: list[dict[str, float | str]] = []
    for candidate in CANDIDATES:
        config = base.Config(
            replications=PILOT_REPLICATIONS,
            rollout_cycles=POLICY_ITERATIONS,
            rollout_size=ROLLOUT_SIZE,
            minibatches=candidate.minibatches,
            training_learning_rate=LEARNING_RATE,
            ppo_epsilon=candidate.epsilon,
            initialization_scale=candidate.initialization_scale,
        )
        initial_weights = base.fit_initial_policy(features, labels, config)
        raw_curves = []
        ppo_curves = []
        oracle_curves = []
        ppo_fractions = []
        raw_distinct_fractions = []
        distinct_fractions = []

        for replication in range(PILOT_REPLICATIONS):
            rng = np.random.default_rng(SEED_START + replication)
            draws = base.common_randomness(rng, len(features), config)
            raw_curves.append(run_static("raw", initial_weights, features, labels, draws, config))
            ppo_curves.append(run_static("ppo", initial_weights, features, labels, draws, config))
            oracle_curve, ppo_fraction, raw_distinct_fraction, distinct_fraction = run_oracle(
                initial_weights, features, labels, draws, config
            )
            oracle_curves.append(oracle_curve)
            ppo_fractions.append(ppo_fraction)
            raw_distinct_fractions.append(raw_distinct_fraction)
            distinct_fractions.append(distinct_fraction)

        raw = np.asarray(raw_curves)
        ppo_curve = np.asarray(ppo_curves)
        oracle = np.asarray(oracle_curves)
        best_static_final = np.maximum(raw[:, -1], ppo_curve[:, -1])
        oracle_gain = oracle[:, -1] - best_static_final

        early_end = min(5, POLICY_ITERATIONS)
        raw_early = np.mean(raw[:, 1 : early_end + 1], axis=1)
        ppo_early = np.mean(ppo_curve[:, 1 : early_end + 1], axis=1)
        oracle_early = np.mean(oracle[:, 1 : early_end + 1], axis=1)

        mean_ppo_fraction = float(np.mean(ppo_fractions))
        mean_raw_distinct_fraction = float(np.mean(raw_distinct_fractions))
        mixed = min(mean_ppo_fraction, mean_raw_distinct_fraction)
        score = float(np.mean(oracle_gain)) / (standard_error(oracle_gain) + 1e-4) + 20.0 * mixed

        rows.append(
            {
                "candidate": candidate.name,
                "rollout_size": float(ROLLOUT_SIZE),
                "minibatches": float(candidate.minibatches),
                "minibatch_size": float(config.minibatch_size),
                "policy_iterations": float(POLICY_ITERATIONS),
                "learning_rate": LEARNING_RATE,
                "eta_max": eta_max,
                "ppo_epsilon": candidate.epsilon,
                "initialization_scale": candidate.initialization_scale,
                "replications": float(PILOT_REPLICATIONS),
                "raw_final": float(np.mean(raw[:, -1])),
                "ppo_final": float(np.mean(ppo_curve[:, -1])),
                "oracle_final": float(np.mean(oracle[:, -1])),
                "oracle_gain_vs_best_static": float(np.mean(oracle_gain)),
                "oracle_gain_se": standard_error(oracle_gain),
                "oracle_ppo_fraction": mean_ppo_fraction,
                "oracle_raw_distinct_fraction": mean_raw_distinct_fraction,
                "oracle_distinct_fraction": float(np.mean(distinct_fractions)),
                "early_raw_minus_ppo": float(np.mean(raw_early - ppo_early)),
                "early_oracle_minus_raw": float(np.mean(oracle_early - raw_early)),
                "early_oracle_minus_ppo": float(np.mean(oracle_early - ppo_early)),
                "score": score,
            }
        )
        print(rows[-1])

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    output = root / "simulation" / "results" / "optdigits_oracle_regime_search.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
