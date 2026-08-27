"""Single-rollout Optdigits experiment on the complete official training split.

One action is sampled once for every training image from a frozen rollout policy
Q. The resulting 3,823 observations are shuffled once, partitioned into disjoint
minibatches, and consumed sequentially without collecting fresh data. This makes
staleness accumulate monotonically with the update index.

The script supports pilot sweeps and final paired comparisons among:
  * unmodified importance-weighted gradient (raw),
  * PPO masking,
  * an infeasible exact-MSE oracle choosing raw or PPO at each update,
  * sample-ESS gates choosing raw above a threshold and PPO below it.

All reported learning rates are checked against the same global smoothness
certificate used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import optdigits_categorical_theory as base


RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
ORACLE_COLOR = "#2F8F78"
ESS_COLOR = "#7A5195"
LIGHT_GRID = "#D9DEE8"
NEUTRAL_COLOR = "#667085"

ESS_THRESHOLDS = (0.20, 0.40, 0.60, 0.80, 0.90)


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_training_split(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data_dir = root / "simulation" / "data"
    # Ensure the archive is present and extracted using the existing loader.
    base.load_optdigits(data_dir, False)
    training = data_dir / "optdigits" / "optdigits.tra"
    array = np.loadtxt(training, delimiter=",")
    features = array[:, :-1] / 16.0
    labels = array[:, -1].astype(int)
    features = np.column_stack([features, np.ones(features.shape[0])])
    return features, labels


def global_smoothness_bound(features: np.ndarray) -> tuple[float, float, float]:
    covariance = features.T @ features / len(features)
    lambda_max = float(np.linalg.eigvalsh(covariance)[-1])
    smoothness = 0.5 * lambda_max
    return lambda_max, smoothness, 1.0 / smoothness


def chunks(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def method_label(method: str, threshold: float | None = None) -> str:
    if method == "ess":
        assert threshold is not None
        return f"ess_{threshold:.2f}"
    return method


def run_fixed_rollout(
    method: str,
    threshold: float | None,
    initial_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    rollout: dict[str, np.ndarray],
    order: np.ndarray,
    batch_size: int,
    learning_rate: float,
    ppo_epsilon: float,
    replication: int,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    config = base.Config(
        rollout_size=len(features),
        minibatches=1,
        training_learning_rate=learning_rate,
        ppo_epsilon=ppo_epsilon,
    )
    rollout_weights = initial_weights.copy()
    weights = initial_weights.copy()
    label = method_label(method, threshold)
    initial_value = base.population_value(weights, features, labels)
    rows: list[dict[str, float | str]] = []
    ppo_choices = 0
    raw_choices = 0
    distinct_choices = 0
    ppo_when_distinct = 0

    rows.append(
        {
            "replication": float(replication),
            "method": label,
            "update": 0.0,
            "examples_used": 0.0,
            "population_value": initial_value,
            "relative_improvement": 0.0,
            "population_rho": 1.0,
            "sample_rho": 1.0,
            "selected_ppo": 0.0,
            "raw_risk": float("nan"),
            "ppo_risk": float("nan"),
        }
    )

    examples_used = 0
    minibatches = chunks(order, batch_size)
    for update, indices in enumerate(minibatches, start=1):
        gradients, sample_rho, _ = base.estimate_gradients(
            weights,
            rollout,
            indices,
            config,
        )
        raw_gradient = gradients["raw"]
        ppo_gradient = gradients["ppo"]
        distinct = not np.allclose(raw_gradient, ppo_gradient, rtol=1e-12, atol=1e-14)
        raw_risk = float("nan")
        ppo_risk = float("nan")

        if method == "raw":
            selected = "raw"
        elif method == "ppo":
            selected = "ppo"
        elif method == "ess":
            if threshold is None:
                raise ValueError("ESS gate requires threshold")
            selected = "ppo" if sample_rho < threshold else "raw"
        elif method == "oracle":
            if distinct:
                risks = base.exact_estimator_risks(
                    weights,
                    rollout_weights,
                    features,
                    labels,
                    len(indices),
                    config,
                )
                raw_risk = float(risks["raw_risk"])
                ppo_risk = float(risks["ppo_risk"])
                selected = "ppo" if ppo_risk < raw_risk else "raw"
            else:
                selected = "raw"
        else:
            raise ValueError(method)

        if selected == "ppo":
            ppo_choices += 1
        else:
            raw_choices += 1
        if distinct:
            distinct_choices += 1
            if selected == "ppo":
                ppo_when_distinct += 1

        weights = weights + learning_rate * gradients[selected]
        examples_used += len(indices)
        value = base.population_value(weights, features, labels)
        rho = base.population_rho(weights, rollout_weights, features)
        rows.append(
            {
                "replication": float(replication),
                "method": label,
                "update": float(update),
                "examples_used": float(examples_used),
                "population_value": value,
                "relative_improvement": (value - initial_value) / max(1.0 - initial_value, 1e-12),
                "population_rho": rho,
                "sample_rho": sample_rho,
                "selected_ppo": float(selected == "ppo"),
                "raw_risk": raw_risk,
                "ppo_risk": ppo_risk,
            }
        )

    final_value = float(rows[-1]["population_value"])
    summary: dict[str, float | str] = {
        "replication": float(replication),
        "method": label,
        "batch_size": float(batch_size),
        "updates": float(len(minibatches)),
        "learning_rate": learning_rate,
        "ppo_epsilon": ppo_epsilon,
        "initial_value": initial_value,
        "final_value": final_value,
        "final_relative_improvement": float(rows[-1]["relative_improvement"]),
        "final_population_rho": float(rows[-1]["population_rho"]),
        "minimum_population_rho": float(min(float(row["population_rho"]) for row in rows)),
        "ppo_fraction": ppo_choices / max(len(minibatches), 1),
        "raw_fraction": raw_choices / max(len(minibatches), 1),
        "distinct_fraction": distinct_choices / max(len(minibatches), 1),
        "ppo_given_distinct": ppo_when_distinct / max(distinct_choices, 1),
    }
    return rows, summary


def run_replication(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    learning_rate: float,
    ppo_epsilon: float,
    initialization_scale: float,
    seed: int,
    replication: int,
    thresholds: tuple[float, ...],
    include_oracle: bool = True,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    config = base.Config(
        initialization_scale=initialization_scale,
        training_learning_rate=learning_rate,
        ppo_epsilon=ppo_epsilon,
    )
    initial_weights = base.fit_initial_policy(features, labels, config)
    rng = np.random.default_rng(seed + replication)
    context_indices = np.arange(len(features), dtype=int)
    uniforms = rng.random(len(features))
    rollout = base.collect_rollout(
        initial_weights,
        features,
        labels,
        context_indices,
        uniforms,
    )
    order = rng.permutation(len(features))

    paths: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    specs: list[tuple[str, float | None]] = [("raw", None), ("ppo", None)]
    if include_oracle:
        specs.append(("oracle", None))
    specs.extend(("ess", threshold) for threshold in thresholds)
    for method, threshold in specs:
        rows, summary = run_fixed_rollout(
            method,
            threshold,
            initial_weights,
            features,
            labels,
            rollout,
            order,
            batch_size,
            learning_rate,
            ppo_epsilon,
            replication,
        )
        paths.extend(rows)
        summaries.append(summary)
    return paths, summaries


def aggregate_summaries(
    summaries: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    methods = sorted({str(row["method"]) for row in summaries})
    output: list[dict[str, float | str]] = []
    for method in methods:
        selected = [row for row in summaries if row["method"] == method]
        values = np.asarray([float(row["final_value"]) for row in selected])
        rel = np.asarray([float(row["final_relative_improvement"]) for row in selected])
        rho = np.asarray([float(row["minimum_population_rho"]) for row in selected])
        ppo = np.asarray([float(row["ppo_fraction"]) for row in selected])
        distinct = np.asarray([float(row["distinct_fraction"]) for row in selected])
        ppo_distinct = np.asarray([float(row["ppo_given_distinct"]) for row in selected])
        output.append(
            {
                "method": method,
                "replications": float(len(selected)),
                "mean_final_value": float(np.mean(values)),
                "se_final_value": standard_error(values),
                "mean_final_relative_improvement": float(np.mean(rel)),
                "se_final_relative_improvement": standard_error(rel),
                "mean_minimum_population_rho": float(np.mean(rho)),
                "mean_ppo_fraction": float(np.mean(ppo)),
                "mean_distinct_fraction": float(np.mean(distinct)),
                "mean_ppo_given_distinct": float(np.mean(ppo_distinct)),
            }
        )
    return output


def pilot_score(rows: list[dict[str, float | str]]) -> float:
    by_method = {str(row["method"]): row for row in rows}
    raw = float(by_method["raw"]["mean_final_value"])
    ppo = float(by_method["ppo"]["mean_final_value"])
    oracle = float(by_method["oracle"]["mean_final_value"])
    oracle_mix = float(by_method["oracle"]["mean_ppo_given_distinct"])
    oracle_distinct = float(by_method["oracle"]["mean_distinct_fraction"])
    gain = oracle - max(raw, ppo)
    mixing_bonus = max(0.0, 1.0 - 2.0 * abs(oracle_mix - 0.5))
    return 100.0 * gain + 0.75 * mixing_bonus + 0.25 * oracle_distinct


def pilot(
    root: Path,
    replications: int,
    seed: int,
) -> None:
    features, labels = load_training_split(root)
    _, _, eta_max = global_smoothness_bound(features)
    batch_sizes = (32, 64, 128)
    learning_rates = (0.10, 0.14, min(0.17, 0.98 * eta_max))
    initialization_scales = (0.20, 0.35)
    ppo_epsilon = 0.20
    result_rows: list[dict[str, float | str]] = []

    for init_scale in initialization_scales:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                if learning_rate > eta_max:
                    continue
                summaries: list[dict[str, float | str]] = []
                for replication in range(replications):
                    _, rep_summaries = run_replication(
                        features,
                        labels,
                        batch_size,
                        learning_rate,
                        ppo_epsilon,
                        init_scale,
                        seed,
                        replication,
                        thresholds=(),
                        include_oracle=True,
                    )
                    summaries.extend(rep_summaries)
                aggregate = aggregate_summaries(summaries)
                by_method = {str(row["method"]): row for row in aggregate}
                row: dict[str, float | str] = {
                    "batch_size": float(batch_size),
                    "updates": float(math.ceil(len(features) / batch_size)),
                    "learning_rate": learning_rate,
                    "eta_max": eta_max,
                    "initialization_scale": init_scale,
                    "replications": float(replications),
                    "raw_final": float(by_method["raw"]["mean_final_value"]),
                    "ppo_final": float(by_method["ppo"]["mean_final_value"]),
                    "oracle_final": float(by_method["oracle"]["mean_final_value"]),
                    "oracle_gain_vs_best_static": float(by_method["oracle"]["mean_final_value"]) - max(
                        float(by_method["raw"]["mean_final_value"]),
                        float(by_method["ppo"]["mean_final_value"]),
                    ),
                    "oracle_ppo_given_distinct": float(by_method["oracle"]["mean_ppo_given_distinct"]),
                    "oracle_distinct_fraction": float(by_method["oracle"]["mean_distinct_fraction"]),
                    "minimum_population_rho": float(by_method["oracle"]["mean_minimum_population_rho"]),
                }
                row["score"] = pilot_score(aggregate)
                result_rows.append(row)
                print(row)

    result_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    write_csv(root / "simulation" / "results" / "optdigits_stale_pilot.csv", result_rows)


def aggregate_paths(
    paths: list[dict[str, float | str]],
    methods: list[str],
) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for method in methods:
        selected = [row for row in paths if row["method"] == method]
        updates = sorted({int(float(row["update"])) for row in selected})
        for update in updates:
            subset = [row for row in selected if int(float(row["update"])) == update]
            value = np.asarray([float(row["population_value"]) for row in subset])
            rel = np.asarray([float(row["relative_improvement"]) for row in subset])
            rho = np.asarray([float(row["population_rho"]) for row in subset])
            sample_rho = np.asarray([float(row["sample_rho"]) for row in subset])
            output.append(
                {
                    "method": method,
                    "update": float(update),
                    "examples_used": float(subset[0]["examples_used"]),
                    "mean_population_value": float(np.mean(value)),
                    "se_population_value": standard_error(value),
                    "mean_relative_improvement": float(np.mean(rel)),
                    "se_relative_improvement": standard_error(rel),
                    "mean_population_rho": float(np.mean(rho)),
                    "mean_sample_rho": float(np.mean(sample_rho)),
                }
            )
    return output


def make_final_figure(
    aggregate_paths_rows: list[dict[str, float | str]],
    shown_methods: list[str],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.7,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )
    styles = {
        "raw": ("Unmodified", RAW_COLOR, "o"),
        "ppo": ("PPO", PPO_COLOR, "s"),
        "oracle": ("Exact MSE oracle", ORACLE_COLOR, "D"),
    }
    for method in shown_methods:
        if method.startswith("ess_"):
            threshold = float(method.split("_")[1])
            styles[method] = (f"ESS gate ({threshold:.2f})", ESS_COLOR, "^")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(7.5, 4.35))
    for method in shown_methods:
        selected = sorted(
            [row for row in aggregate_paths_rows if row["method"] == method],
            key=lambda row: float(row["update"]),
        )
        x = np.asarray([float(row["update"]) for row in selected])
        mean = 100.0 * np.asarray([float(row["mean_relative_improvement"]) for row in selected])
        error = 100.0 * np.asarray([float(row["se_relative_improvement"]) for row in selected])
        label, color, marker = styles[method]
        ax.plot(x, mean, color=color, linewidth=2.1, marker=marker, markevery=max(1, len(x)//12), markersize=4.2, label=label)
        ax.fill_between(x, mean - 1.96 * error, mean + 1.96 * error, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Stale-rollout minibatch update")
    ax.set_ylabel("Relative improvement toward perfect policy (\%)")
    ax.set_title("Learning from one fixed rollout")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def final_run(
    root: Path,
    batch_size: int,
    learning_rate: float,
    initialization_scale: float,
    replications: int,
    seed: int,
) -> None:
    features, labels = load_training_split(root)
    lambda_max, smoothness, eta_max = global_smoothness_bound(features)
    if learning_rate > eta_max:
        raise RuntimeError(f"learning rate {learning_rate} exceeds certified {eta_max}")

    paths: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    for replication in range(replications):
        rep_paths, rep_summaries = run_replication(
            features,
            labels,
            batch_size,
            learning_rate,
            0.20,
            initialization_scale,
            seed,
            replication,
            ESS_THRESHOLDS,
            include_oracle=True,
        )
        paths.extend(rep_paths)
        summaries.extend(rep_summaries)

    aggregate = aggregate_summaries(summaries)
    by_method = {str(row["method"]): row for row in aggregate}
    gate_methods = [method for method in by_method if method.startswith("ess_")]
    # Choose the displayed gate by agreement with the exact oracle's switching rate,
    # not by final return.
    oracle_ppo = float(by_method["oracle"]["mean_ppo_fraction"])
    displayed_gate = min(
        gate_methods,
        key=lambda method: abs(float(by_method[method]["mean_ppo_fraction"]) - oracle_ppo),
    )
    shown = ["raw", "ppo", "oracle", displayed_gate]
    aggregated_paths = aggregate_paths(paths, shown)

    result_dir = root / "simulation" / "results"
    write_csv(result_dir / "optdigits_stale_final_runs.csv", summaries)
    write_csv(result_dir / "optdigits_stale_final_summary.csv", aggregate)
    write_csv(result_dir / "optdigits_stale_final_curve.csv", aggregated_paths)
    make_final_figure(
        aggregated_paths,
        shown,
        root / "figures" / "optdigits_stale_rollout",
    )

    summary_lines = [
        f"training_examples={len(features)}",
        f"batch_size={batch_size}",
        f"updates={math.ceil(len(features)/batch_size)}",
        f"learning_rate={learning_rate:.8f}",
        f"feature_cov_lambda_max={lambda_max:.8f}",
        f"global_smoothness_bound={smoothness:.8f}",
        f"certified_eta_max={eta_max:.8f}",
        f"initialization_scale={initialization_scale:.8f}",
        f"replications={replications}",
        f"displayed_gate={displayed_gate}",
    ]
    for method in shown:
        row = by_method[method]
        summary_lines.extend(
            [
                f"{method}_final_value={float(row['mean_final_value']):.8f}",
                f"{method}_final_value_se={float(row['se_final_value']):.8f}",
                f"{method}_relative_improvement={float(row['mean_final_relative_improvement']):.8f}",
                f"{method}_minimum_population_rho={float(row['mean_minimum_population_rho']):.8f}",
                f"{method}_ppo_fraction={float(row['mean_ppo_fraction']):.8f}",
            ]
        )
    (result_dir / "optdigits_stale_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pilot_parser = sub.add_parser("pilot")
    pilot_parser.add_argument("--replications", type=int, default=5)
    pilot_parser.add_argument("--seed", type=int, default=20700826)
    final_parser = sub.add_parser("final")
    final_parser.add_argument("--batch-size", type=int, required=True)
    final_parser.add_argument("--learning-rate", type=float, required=True)
    final_parser.add_argument("--initialization-scale", type=float, required=True)
    final_parser.add_argument("--replications", type=int, default=100)
    final_parser.add_argument("--seed", type=int, default=20800826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "pilot":
        pilot(root, args.replications, args.seed)
    else:
        final_run(
            root,
            args.batch_size,
            args.learning_rate,
            args.initialization_scale,
            args.replications,
            args.seed,
        )


if __name__ == "__main__":
    main()
