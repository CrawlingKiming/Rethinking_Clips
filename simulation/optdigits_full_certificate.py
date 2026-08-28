"""Exact Raw/PPO full-certificate comparison on categorical Optdigits.

The experiment follows a deterministic path of common frozen policy states.
At every state it exactly enumerates all official-train context-action pairs to
obtain the mean, second moment, and MSE of iid minibatch Raw and PPO gradient
estimators.  It then compares the MSE oracle, the fixed-step full-certificate
oracle, and a safe oracle that can choose a null update.

This is a controlled estimator experiment, not a cumulative-return oracle or
a longitudinal optimization experiment.  Its ESS proxy audit is path-specific
and uses disjoint alternating calibration and evaluation states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import optdigits_categorical_theory as base


ROOT = Path(__file__).resolve().parents[1]
RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
MSE_REDUCTION_COLOR = "#2A9D8F"
REQUIREMENT_COLOR = "#7A5195"
NEUTRAL_COLOR = "#59636E"
GRID_COLOR = "#D9DEE8"
ORACLE_TIE_TOLERANCE = 1e-12
REGION_COLORS = {
    "raw": "#DCEafa",
    "ppo": "#FDE7CF",
    "noop": "#ECEFF3",
}


def load_official_train(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the official Optdigits training split."""
    archive = data_dir / "optdigits.zip"
    extracted = data_dir / "optdigits"
    path = extracted / "optdigits.tra"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            urllib.request.urlretrieve(base.DATA_URL, archive)
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
    array = np.loadtxt(path, delimiter=",")
    features = array[:, :-1] / 16.0
    labels = array[:, -1].astype(int)
    features = np.column_stack([features, np.ones(features.shape[0])])
    return features, labels


def global_smoothness_bound(
    features: np.ndarray,
) -> tuple[float, float, float]:
    feature_second_moment = features.T @ features / len(features)
    lambda_max = float(np.linalg.eigvalsh(feature_second_moment)[-1])
    smoothness = 0.5 * lambda_max
    return lambda_max, smoothness, 1.0 / smoothness


def minimum_shifted_logit(
    weights: np.ndarray,
    features: np.ndarray,
) -> float:
    """Return the smallest logit after rowwise max subtraction."""
    logits = features @ weights.T
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return float(np.min(shifted))


def finite_difference_directional_error(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    step: float = 1e-5,
) -> float:
    """Independently check one deterministic population-gradient projection."""
    probe = np.sin(np.arange(1, weights.size + 1, dtype=float)).reshape(
        weights.shape
    )
    probe /= np.linalg.norm(probe)
    plus = base.population_value(weights + step * probe, features, labels)
    minus = base.population_value(weights - step * probe, features, labels)
    finite_difference = (plus - minus) / (2.0 * step)
    _, gradient = base.population_value_and_gradient(weights, features, labels)
    analytic = float(np.sum(gradient * probe))
    return abs(finite_difference - analytic)


def exact_population_rho(
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
) -> float:
    current = base.softmax(features @ weights.T)
    rollout = base.softmax(features @ rollout_weights.T)
    second_moment = np.mean(np.sum(current**2 / rollout, axis=1))
    return float(1.0 / second_moment)


def exact_estimator_moments(
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, dict[str, float | np.ndarray]]]:
    """Enumerate exact iid-minibatch moments for Raw and PPO estimators."""
    current = base.softmax(features @ weights.T)
    rollout = base.softmax(features @ rollout_weights.T)
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
        ratios <= 1.0 + epsilon,
        ratios >= 1.0 - epsilon,
    )
    ppo_coefficients = raw_coefficients * ppo_mask

    feature_norm_sq = np.sum(features**2, axis=1)
    probability_norm_sq = np.sum(current**2, axis=1, keepdims=True)
    score_norm_sq = feature_norm_sq[:, None] * (
        1.0 - 2.0 * current + probability_norm_sq
    )

    signal_sq = float(np.sum(true_gradient**2))
    moments: dict[str, dict[str, float | np.ndarray]] = {}
    for name, coefficients in (
        ("raw", raw_coefficients),
        ("ppo", ppo_coefficients),
    ):
        probability_weighted = rollout * coefficients
        centered_coefficients = probability_weighted - current * np.sum(
            probability_weighted,
            axis=1,
            keepdims=True,
        )
        mean_gradient = centered_coefficients.T @ features / population_size
        single_second_moment = float(
            np.mean(np.sum(rollout * coefficients**2 * score_norm_sq, axis=1))
        )
        mean_norm_sq = float(np.sum(mean_gradient**2))
        variance_residual = single_second_moment - mean_norm_sq
        variance_tolerance = 1e-12 * max(
            1.0,
            abs(single_second_moment),
            abs(mean_norm_sq),
        )
        if variance_residual < -variance_tolerance:
            raise FloatingPointError(
                f"negative {name} variance residual: {variance_residual}"
            )
        single_variance = max(variance_residual, 0.0)
        estimator_variance = single_variance / batch_size
        second_moment = mean_norm_sq + estimator_variance
        bias_sq = float(np.sum((mean_gradient - true_gradient) ** 2))
        mse = bias_sq + estimator_variance
        alignment = float(np.sum(true_gradient * mean_gradient))
        mse_identity = signal_sq + second_moment - 2.0 * alignment
        moments[name] = {
            "mean_gradient": mean_gradient,
            "mean_norm_sq": mean_norm_sq,
            "single_second_moment": single_second_moment,
            "single_variance": single_variance,
            "estimator_variance": estimator_variance,
            "second_moment": second_moment,
            "bias_sq": bias_sq,
            "mse": mse,
            "alignment": alignment,
            "mse_identity_error": mse_identity - mse,
        }
    return true_gradient, moments


def evaluate_state(
    path_scale: float,
    weights: np.ndarray,
    rollout_weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    epsilon: float,
    eta: float,
    smoothness: float,
) -> dict[str, float | str]:
    current_min_shifted_logit = minimum_shifted_logit(weights, features)
    rollout_min_shifted_logit = minimum_shifted_logit(
        rollout_weights,
        features,
    )
    if min(current_min_shifted_logit, rollout_min_shifted_logit) < -50.0:
        raise ValueError(
            "the inherited softmax -50 clip is active; reduce the path "
            "scale so the analytic gradient and certificate remain exact"
        )
    value, checked_gradient = base.population_value_and_gradient(
        weights,
        features,
        labels,
    )
    true_gradient, moments = exact_estimator_moments(
        weights,
        rollout_weights,
        features,
        labels,
        batch_size,
        epsilon,
    )
    signal_sq = float(np.sum(true_gradient**2))
    row: dict[str, float | str] = {
        "path_scale": path_scale,
        "population_rho": exact_population_rho(
            weights,
            rollout_weights,
            features,
        ),
        "population_value": value,
        "signal_sq": signal_sq,
        "current_min_shifted_logit": current_min_shifted_logit,
        "rollout_min_shifted_logit": rollout_min_shifted_logit,
        "gradient_check_max_abs": float(
            np.max(np.abs(true_gradient - checked_gradient))
        ),
    }

    for name in ("raw", "ppo"):
        item = moments[name]
        alignment = float(item["alignment"])
        second_moment = float(item["second_moment"])
        mse = float(item["mse"])
        certificate = eta * alignment - 0.5 * smoothness * eta**2 * second_moment
        polarized = (
            0.5 * eta * (signal_sq - mse)
            + 0.5 * eta * (1.0 - smoothness * eta) * second_moment
        )
        row[f"{name}_alignment"] = alignment
        row[f"{name}_mean_norm_sq"] = float(item["mean_norm_sq"])
        row[f"{name}_single_variance"] = float(item["single_variance"])
        row[f"{name}_variance"] = float(item["estimator_variance"])
        row[f"{name}_second_moment"] = second_moment
        row[f"{name}_bias_sq"] = float(item["bias_sq"])
        row[f"{name}_mse"] = mse
        row[f"{name}_mse_identity_error"] = float(item["mse_identity_error"])
        row[f"{name}_certificate"] = certificate
        row[f"{name}_certificate_identity_error"] = polarized - certificate

    row["mse_reduction"] = float(row["raw_mse"]) - float(row["ppo_mse"])
    row["mse_tie"] = float(
        abs(float(row["raw_mse"]) - float(row["ppo_mse"]))
        <= ORACLE_TIE_TOLERANCE
    )
    row["certificate_requirement"] = (1.0 - smoothness * eta) * (
        float(row["raw_second_moment"]) - float(row["ppo_second_moment"])
    )
    row["certificate_gap_ppo_minus_raw"] = (
        float(row["ppo_certificate"]) - float(row["raw_certificate"])
    )
    row["certificate_tie"] = float(
        abs(float(row["certificate_gap_ppo_minus_raw"]))
        <= ORACLE_TIE_TOLERANCE
    )
    row["mse_oracle"] = (
        "ppo" if float(row["ppo_mse"]) < float(row["raw_mse"]) else "raw"
    )
    row["certificate_oracle"] = (
        "ppo"
        if float(row["ppo_certificate"]) > float(row["raw_certificate"])
        else "raw"
    )
    best_certificate = max(
        float(row["raw_certificate"]),
        float(row["ppo_certificate"]),
    )
    row["safe_oracle"] = (
        str(row["certificate_oracle"]) if best_certificate > 0.0 else "noop"
    )
    row["ppo_lower_mse_raw_higher_certificate"] = float(
        float(row["ppo_mse"]) < float(row["raw_mse"])
        and float(row["raw_certificate"]) > float(row["ppo_certificate"])
    )
    return row


def label_runs(
    rows: list[dict[str, float | str]],
    key: str,
) -> list[dict[str, float | str | int]]:
    output: list[dict[str, float | str | int]] = []
    start = 0
    while start < len(rows):
        label = str(rows[start][key])
        end = start
        while end + 1 < len(rows) and str(rows[end + 1][key]) == label:
            end += 1
        output.append(
            {
                "label": label,
                "points": end - start + 1,
                "start_scale": float(rows[start]["path_scale"]),
                "end_scale": float(rows[end]["path_scale"]),
                "start_rho": float(rows[start]["population_rho"]),
                "end_rho": float(rows[end]["population_rho"]),
            }
        )
        start = end + 1
    return output


def predicate_runs(
    rows: list[dict[str, float | str]],
    key: str,
) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    start = 0
    while start < len(rows):
        if float(rows[start][key]) != 1.0:
            start += 1
            continue
        end = start
        while end + 1 < len(rows) and float(rows[end + 1][key]) == 1.0:
            end += 1
        output.append(
            {
                "points": end - start + 1,
                "start_scale": float(rows[start]["path_scale"]),
                "end_scale": float(rows[end]["path_scale"]),
                "start_rho": float(rows[start]["population_rho"]),
                "end_rho": float(rows[end]["population_rho"]),
                "min_raw_certificate": min(
                    float(row["raw_certificate"]) for row in rows[start : end + 1]
                ),
                "min_ppo_certificate": min(
                    float(row["ppo_certificate"]) for row in rows[start : end + 1]
                ),
            }
        )
        start = end + 1
    return output


def compact_state(
    row: dict[str, float | str] | None,
) -> dict[str, float | str] | None:
    if row is None:
        return None
    keys = (
        "path_scale",
        "population_rho",
        "population_value",
        "signal_sq",
        "raw_alignment",
        "ppo_alignment",
        "raw_second_moment",
        "ppo_second_moment",
        "raw_mse",
        "ppo_mse",
        "mse_reduction",
        "mse_tie",
        "certificate_requirement",
        "raw_certificate",
        "ppo_certificate",
        "certificate_gap_ppo_minus_raw",
        "certificate_tie",
        "mse_oracle",
        "certificate_oracle",
        "safe_oracle",
    )
    return {key: row[key] for key in keys}


def first_label(
    rows: Iterable[dict[str, float | str]],
    key: str,
    label: str,
) -> dict[str, float | str] | None:
    return next((row for row in rows if str(row[key]) == label), None)


def classification_metrics(
    rows: list[dict[str, float | str]],
    truth_key: str,
    predictor: Callable[[float], str],
    classes: tuple[str, ...],
) -> dict[str, object]:
    truth = [str(row[truth_key]) for row in rows]
    predicted = [predictor(float(row["population_rho"])) for row in rows]
    recalls: dict[str, float] = {}
    for label in classes:
        positives = sum(value == label for value in truth)
        if positives:
            recalls[label] = sum(
                target == estimate == label
                for target, estimate in zip(truth, predicted)
            ) / positives
    return {
        "states": len(rows),
        "accuracy": sum(
            target == estimate for target, estimate in zip(truth, predicted)
        )
        / len(rows),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "recall": recalls,
        "truth_counts": {label: truth.count(label) for label in classes},
        "prediction_counts": {
            label: predicted.count(label) for label in classes
        },
    }


def threshold_at_cut(rho: np.ndarray, cut: int) -> float:
    if cut == 0:
        return float(np.nextafter(rho[0], math.inf))
    if cut == len(rho):
        return float(np.nextafter(rho[-1], -math.inf))
    return float(0.5 * (rho[cut - 1] + rho[cut]))


def fit_binary_threshold(
    rows: list[dict[str, float | str]],
    key: str,
) -> float:
    ordered = sorted(
        rows,
        key=lambda row: float(row["population_rho"]),
        reverse=True,
    )
    best_correct = -1
    best_cut = 0
    for cut in range(len(ordered) + 1):
        correct = sum(
            str(row[key]) == ("raw" if index < cut else "ppo")
            for index, row in enumerate(ordered)
        )
        if correct > best_correct:
            best_correct = correct
            best_cut = cut
    rho = np.asarray([float(row["population_rho"]) for row in ordered])
    return threshold_at_cut(rho, best_cut)


def fit_safe_thresholds(
    rows: list[dict[str, float | str]],
) -> tuple[float, float]:
    ordered = sorted(
        rows,
        key=lambda row: float(row["population_rho"]),
        reverse=True,
    )
    size = len(ordered)
    prefix: dict[str, np.ndarray] = {}
    for label in ("raw", "ppo", "noop"):
        indicator = np.asarray(
            [str(row["safe_oracle"]) == label for row in ordered],
            dtype=int,
        )
        prefix[label] = np.concatenate(([0], np.cumsum(indicator)))

    best_correct = -1
    best_raw_cut = 0
    best_ppo_cut = 0
    for raw_cut in range(size + 1):
        for ppo_cut in range(raw_cut, size + 1):
            correct = int(
                prefix["raw"][raw_cut]
                + prefix["ppo"][ppo_cut]
                - prefix["ppo"][raw_cut]
                + prefix["noop"][size]
                - prefix["noop"][ppo_cut]
            )
            if correct > best_correct:
                best_correct = correct
                best_raw_cut = raw_cut
                best_ppo_cut = ppo_cut

    rho = np.asarray([float(row["population_rho"]) for row in ordered])
    return (
        threshold_at_cut(rho, best_raw_cut),
        threshold_at_cut(rho, best_ppo_cut),
    )


def ess_proxy_audit(
    rows: list[dict[str, float | str]],
    scope_name: str,
    selection: str,
) -> dict[str, object]:
    calibration = rows[::2]
    evaluation = rows[1::2]
    if not calibration or not evaluation:
        raise ValueError("ESS proxy audit requires at least two eligible states")

    binary_threshold = fit_binary_threshold(
        calibration,
        "certificate_oracle",
    )

    def binary_predictor(rho: float) -> str:
        return "ppo" if rho < binary_threshold else "raw"

    raw_threshold, noop_threshold = fit_safe_thresholds(calibration)

    def safe_predictor(rho: float) -> str:
        if rho >= raw_threshold:
            return "raw"
        if rho >= noop_threshold:
            return "ppo"
        return "noop"

    return {
        "scope": {
            "name": scope_name,
            "selection": selection,
            "states": len(rows),
            "path_scale_min": min(float(row["path_scale"]) for row in rows),
            "path_scale_max": max(float(row["path_scale"]) for row in rows),
            "split": (
                "alternating states within this scope; even positions "
                "calibrate and odd positions evaluate"
            ),
            "limitation": (
                "path-specific held-out interpolation, not validation on an "
                "independent trajectory"
            ),
        },
        "forced_raw_ppo": {
            "rule": "choose PPO when rho is below threshold; otherwise Raw",
            "rho_threshold": binary_threshold,
            "calibration": classification_metrics(
                calibration,
                "certificate_oracle",
                binary_predictor,
                ("raw", "ppo"),
            ),
            "evaluation": classification_metrics(
                evaluation,
                "certificate_oracle",
                binary_predictor,
                ("raw", "ppo"),
            ),
        },
        "safe_three_action": {
            "rule": (
                "Raw at high rho, PPO at intermediate rho, and no-op at low rho"
            ),
            "raw_if_rho_at_least": raw_threshold,
            "noop_if_rho_below": noop_threshold,
            "calibration": classification_metrics(
                calibration,
                "safe_oracle",
                safe_predictor,
                ("raw", "ppo", "noop"),
            ),
            "evaluation": classification_metrics(
                evaluation,
                "safe_oracle",
                safe_predictor,
                ("raw", "ppo", "noop"),
            ),
        },
    }


def add_safe_regions(
    axis: plt.Axes,
    rows: list[dict[str, float | str]],
) -> None:
    ordered = sorted(rows, key=lambda row: float(row["population_rho"]))
    rho = np.asarray([float(row["population_rho"]) for row in ordered])
    start = 0
    while start < len(ordered):
        label = str(ordered[start]["safe_oracle"])
        end = start
        while (
            end + 1 < len(ordered)
            and str(ordered[end + 1]["safe_oracle"]) == label
        ):
            end += 1
        left = rho[start] if start == 0 else 0.5 * (rho[start - 1] + rho[start])
        right = (
            rho[end]
            if end == len(ordered) - 1
            else 0.5 * (rho[end] + rho[end + 1])
        )
        axis.axvspan(
            left,
            right,
            color=REGION_COLORS[label],
            alpha=0.68,
            linewidth=0.0,
            zorder=0,
        )
        start = end + 1


def make_figure(
    rows: list[dict[str, float | str]],
    minimum_scale: float,
    output_stem: Path,
) -> None:
    visible = [
        row for row in rows if float(row["path_scale"]) >= minimum_scale
    ]
    ordered = sorted(visible, key=lambda row: float(row["population_rho"]))
    rho = np.asarray([float(row["population_rho"]) for row in ordered])
    raw_mse = np.asarray([float(row["raw_mse"]) for row in ordered])
    ppo_mse = np.asarray([float(row["ppo_mse"]) for row in ordered])
    raw_certificate = np.asarray(
        [float(row["raw_certificate"]) for row in ordered]
    )
    ppo_certificate = np.asarray(
        [float(row["ppo_certificate"]) for row in ordered]
    )
    mse_reduction = np.asarray(
        [float(row["mse_reduction"]) for row in ordered]
    )
    requirement = np.asarray(
        [float(row["certificate_requirement"]) for row in ordered]
    )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.65))
    for axis in axes:
        add_safe_regions(axis, visible)
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8, zorder=1)
        axis.set_xlabel(r"Population normalized ESS $\rho$")
        axis.set_xlim(float(np.min(rho)), float(np.max(rho)))

    axes[0].plot(rho, raw_mse, color=RAW_COLOR, linewidth=2.0, label="Raw")
    axes[0].plot(rho, ppo_mse, color=PPO_COLOR, linewidth=2.0, label="PPO")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Exact estimator MSE")
    axes[0].set_title("(a) Estimator-risk crossover")
    axes[0].legend(frameon=False)

    axes[1].plot(
        rho,
        raw_certificate,
        color=RAW_COLOR,
        linewidth=2.0,
        label=r"$B_{\mathrm{Raw}}$",
    )
    axes[1].plot(
        rho,
        ppo_certificate,
        color=PPO_COLOR,
        linewidth=2.0,
        label=r"$B_{\mathrm{PPO}}$",
    )
    axes[1].axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_ylabel(r"Full certificate $B(\eta)$")
    axes[1].set_title("(b) Fixed-step certificate")
    axes[1].legend(frameon=False)

    axes[2].plot(
        rho,
        mse_reduction,
        color=MSE_REDUCTION_COLOR,
        linewidth=2.0,
        label=r"MSE reduction $m_R-m_P$",
    )
    axes[2].plot(
        rho,
        requirement,
        color=REQUIREMENT_COLOR,
        linestyle="--",
        linewidth=2.0,
        label=r"Required $(1-L\eta)(s_R-s_P)$",
    )
    axes[2].axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    axes[2].set_yscale("symlog", linthresh=1e-6)
    axes[2].set_ylabel("Certificate-crossover terms")
    axes[2].set_title("(c) Why lower MSE can be insufficient")
    axes[2].legend(frameon=False, fontsize=7.8)

    figure.legend(
        handles=[
            Patch(facecolor=REGION_COLORS["raw"], label="Safe Raw"),
            Patch(facecolor=REGION_COLORS["ppo"], label="Safe PPO"),
            Patch(facecolor=REGION_COLORS["noop"], label="No certified update"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=2.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--max-scale", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--eta", type=float, default=0.17)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--rollout-scale", type=float, default=0.20)
    parser.add_argument("--target-scale", type=float, default=1.00)
    parser.add_argument("--proxy-min-scale", type=float, default=0.05)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "simulation" / "results",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("points must be at least two")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    numeric_arguments = (
        args.max_scale,
        args.eta,
        args.epsilon,
        args.rollout_scale,
        args.target_scale,
        args.proxy_min_scale,
    )
    if not all(math.isfinite(value) for value in numeric_arguments):
        raise ValueError("all numeric path and optimizer arguments must be finite")
    if args.max_scale <= 0.0:
        raise ValueError("max-scale must be positive")
    if not (0.0 <= args.epsilon < 1.0):
        raise ValueError("epsilon must lie in [0, 1)")
    if not (0.0 <= args.proxy_min_scale <= args.max_scale):
        raise ValueError("proxy-min-scale must lie on the controlled path")
    if args.rollout_scale == args.target_scale:
        raise ValueError("rollout-scale and target-scale must define a path")

    config = base.Config(ppo_epsilon=args.epsilon)
    features, labels = load_official_train(ROOT / "simulation" / "data")
    fitted_weights = base.fit_initial_policy(
        features,
        labels,
        replace(config, initialization_scale=1.0),
    )
    rollout_weights = args.rollout_scale * fitted_weights
    target_weights = args.target_scale * fitted_weights
    direction = target_weights - rollout_weights
    if not np.any(direction):
        raise ValueError("the controlled-state path direction is zero")
    lambda_max, smoothness, eta_max = global_smoothness_bound(features)
    if not (0.0 < args.eta <= eta_max + 1e-14):
        raise ValueError(
            f"eta={args.eta} violates the global limit 1/L={eta_max}"
        )

    rows = [
        evaluate_state(
            float(scale),
            rollout_weights + float(scale) * direction,
            rollout_weights,
            features,
            labels,
            args.batch_size,
            args.epsilon,
            args.eta,
            smoothness,
        )
        for scale in np.linspace(0.0, args.max_scale, args.points)
    ]
    finite_difference_scales = (0.0, 0.5 * args.max_scale, args.max_scale)
    finite_difference_errors = [
        finite_difference_directional_error(
            rollout_weights + scale * direction,
            features,
            labels,
        )
        for scale in finite_difference_scales
    ]
    focused = [
        row
        for row in rows
        if float(row["path_scale"]) >= args.proxy_min_scale
    ]
    if len(focused) < 2:
        raise ValueError("ESS proxy audit requires at least two focused states")
    disagreements = [
        row
        for row in rows
        if float(row["ppo_lower_mse_raw_higher_certificate"]) == 1.0
    ]
    high_ess_blip = [
        row
        for row in rows
        if float(row["path_scale"]) < args.proxy_min_scale
        and row["certificate_oracle"] == "ppo"
    ]
    blip_gaps = [
        float(row["ppo_certificate"]) - float(row["raw_certificate"])
        for row in high_ess_blip
    ]

    summary = {
        "protocol": {
            "population": "official Optdigits training split only",
            "population_size": len(features),
            "path": (
                "Q=rollout_scale*W_fit; theta(t)=Q+t*(target_scale*W_fit-Q)"
            ),
            "classifier_steps": config.classifier_steps,
            "classifier_learning_rate": config.classifier_learning_rate,
            "rollout_scale": args.rollout_scale,
            "target_scale": args.target_scale,
            "path_scale_min": 0.0,
            "path_scale_max": args.max_scale,
            "path_points": args.points,
            "batch_size": args.batch_size,
            "ppo_epsilon": args.epsilon,
            "eta": args.eta,
            "feature_covariance_lambda_max": lambda_max,
            "global_smoothness_L": smoothness,
            "eta_max_1_over_L": eta_max,
            "eta_times_L": args.eta * smoothness,
            "raw_ppo_tie_tolerance": ORACLE_TIE_TOLERANCE,
            "raw_ppo_tie_break": (
                "Raw is the deterministic stored label; explicit tie flags "
                "and strict-win counts are also reported"
            ),
            "softmax_clip_guard": (
                "abort if the inherited -50 shifted-logit clip is active"
            ),
            "moments": "exact enumeration of all context-action pairs",
            "random_seeds": "none; deterministic controlled-state path",
            "interpretation": (
                "statewise estimator and certificate comparison, not a "
                "cumulative-return oracle"
            ),
        },
        "all_path_counts": {
            "states": len(rows),
            "mse_ties": sum(float(row["mse_tie"]) == 1.0 for row in rows),
            "certificate_ties": sum(
                float(row["certificate_tie"]) == 1.0 for row in rows
            ),
            "strict_mse_raw": sum(
                float(row["raw_mse"]) < float(row["ppo_mse"])
                - ORACLE_TIE_TOLERANCE
                for row in rows
            ),
            "strict_mse_ppo": sum(
                float(row["ppo_mse"]) < float(row["raw_mse"])
                - ORACLE_TIE_TOLERANCE
                for row in rows
            ),
            "strict_certificate_raw": sum(
                float(row["raw_certificate"])
                > float(row["ppo_certificate"]) + ORACLE_TIE_TOLERANCE
                for row in rows
            ),
            "strict_certificate_ppo": sum(
                float(row["ppo_certificate"])
                > float(row["raw_certificate"]) + ORACLE_TIE_TOLERANCE
                for row in rows
            ),
            "safe_raw": sum(row["safe_oracle"] == "raw" for row in rows),
            "safe_ppo": sum(row["safe_oracle"] == "ppo" for row in rows),
            "safe_noop": sum(row["safe_oracle"] == "noop" for row in rows),
            "safe_raw_fraction": sum(
                row["safe_oracle"] == "raw" for row in rows
            )
            / len(rows),
            "safe_ppo_fraction": sum(
                row["safe_oracle"] == "ppo" for row in rows
            )
            / len(rows),
            "safe_noop_fraction": sum(
                row["safe_oracle"] == "noop" for row in rows
            )
            / len(rows),
        },
        "all_path_mse_oracle_runs": label_runs(rows, "mse_oracle"),
        "all_path_certificate_oracle_runs": label_runs(
            rows,
            "certificate_oracle",
        ),
        "all_path_safe_oracle_runs": label_runs(rows, "safe_oracle"),
        "high_ess_ppo_blip": {
            "definition": (
                "certificate-oracle PPO states below focused-path minimum; "
                "gap is B_PPO minus B_Raw"
            ),
            "states": len(high_ess_blip),
            "start": compact_state(high_ess_blip[0] if high_ess_blip else None),
            "end": compact_state(high_ess_blip[-1] if high_ess_blip else None),
            "certificate_gap_min": min(blip_gaps) if blip_gaps else None,
            "certificate_gap_max": max(blip_gaps) if blip_gaps else None,
        },
        "focused_path_minimum_scale": args.proxy_min_scale,
        "focused_path_counts": {
            "states": len(focused),
            "safe_raw": sum(row["safe_oracle"] == "raw" for row in focused),
            "safe_ppo": sum(row["safe_oracle"] == "ppo" for row in focused),
            "safe_noop": sum(row["safe_oracle"] == "noop" for row in focused),
            "safe_raw_fraction": sum(
                row["safe_oracle"] == "raw" for row in focused
            )
            / len(focused),
            "safe_ppo_fraction": sum(
                row["safe_oracle"] == "ppo" for row in focused
            )
            / len(focused),
            "safe_noop_fraction": sum(
                row["safe_oracle"] == "noop" for row in focused
            )
            / len(focused),
        },
        "focused_path_mse_oracle_runs": label_runs(focused, "mse_oracle"),
        "focused_path_certificate_oracle_runs": label_runs(
            focused,
            "certificate_oracle",
        ),
        "focused_path_safe_oracle_runs": label_runs(focused, "safe_oracle"),
        "ppo_lower_mse_raw_higher_certificate_runs": predicate_runs(
            focused,
            "ppo_lower_mse_raw_higher_certificate",
        ),
        "first_ppo_lower_mse_raw_higher_certificate": compact_state(
            disagreements[0] if disagreements else None
        ),
        "focused_first_mse_ppo": compact_state(
            first_label(focused, "mse_oracle", "ppo")
        ),
        "focused_first_certificate_ppo": compact_state(
            first_label(focused, "certificate_oracle", "ppo")
        ),
        "focused_first_safe_ppo": compact_state(
            first_label(focused, "safe_oracle", "ppo")
        ),
        "focused_first_noop": compact_state(
            first_label(focused, "safe_oracle", "noop")
        ),
        "numerical_checks": {
            "max_gradient_check_abs": max(
                abs(float(row["gradient_check_max_abs"])) for row in rows
            ),
            "max_mse_identity_abs": max(
                max(
                    abs(float(row["raw_mse_identity_error"])),
                    abs(float(row["ppo_mse_identity_error"])),
                )
                for row in rows
            ),
            "max_certificate_identity_abs": max(
                max(
                    abs(float(row["raw_certificate_identity_error"])),
                    abs(float(row["ppo_certificate_identity_error"])),
                )
                for row in rows
            ),
            "max_raw_bias_sq": max(float(row["raw_bias_sq"]) for row in rows),
            "finite_difference_scales": list(finite_difference_scales),
            "max_finite_difference_directional_error": max(
                finite_difference_errors
            ),
            "minimum_shifted_logit": min(
                min(
                    float(row["current_min_shifted_logit"]),
                    float(row["rollout_min_shifted_logit"]),
                )
                for row in rows
            ),
        },
    }
    proxy = {
        "full_path": ess_proxy_audit(
            rows,
            "full_path",
            "all controlled states, including the high-ESS PPO blip",
        ),
        "focused_main_branch": ess_proxy_audit(
            focused,
            "focused_main_branch",
            (
                "controlled states with path_scale >= "
                f"{args.proxy_min_scale:g}"
            ),
        ),
        "comparison_note": (
            "The two audits expose sensitivity to excluding the small "
            "high-ESS mask-boundary branch. Neither establishes transfer to "
            "an independent optimization trajectory."
        ),
    }

    result_stem = args.result_dir / "optdigits_full_certificate"
    atomic_write_csv(result_stem.with_name(result_stem.name + "_path.csv"), rows)
    atomic_write_json(
        result_stem.with_name(result_stem.name + "_summary.json"),
        summary,
    )
    atomic_write_json(
        result_stem.with_name(result_stem.name + "_ess_proxy.json"),
        proxy,
    )
    make_figure(
        rows,
        args.proxy_min_scale,
        args.figure_dir / "optdigits_full_certificate",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(proxy, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
