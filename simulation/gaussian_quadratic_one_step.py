"""Exact IS/TIS/PPO one-step comparison for a Gaussian bandit.

The rollout and current policies are unit-variance Gaussians separated by
``delta``. Estimator moments, crossover margins, and expected policy gains are
evaluated from truncated Gaussian polynomial moments. A fixed-seed Monte Carlo
experiment separately checks the finite-moment reliability event and its
adaptive-step policy-improvement consequence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "simulation" / "results"
FIGURES_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "figures" / "coding"))

import matplotlib.pyplot as plt
from paperstyle import FULL, format_sig, use_paper_style

G = 2.0
N = 32
EPSILON = 0.2
TIS_CAP = 3.0
ETA = 0.4
SMOOTHNESS = 1.0
MINIMUM_RHO = 0.005
MAXIMUM_RHO = 0.9
DELTA_CONFIDENCE = 0.1
MONTE_CARLO_POINTS = 81
MONTE_CARLO_BATCHES = 100_000
MONTE_CARLO_CHUNK_SIZE = 25_000
MONTE_CARLO_SEED = 20260901

IS_COLOR = "#3568B8"
PPO_COLOR = "#E07A3F"
TIS_COLOR = "#27966F"
NEUTRAL_COLOR = "#4B5563"


def normal_pdf(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return 1.0
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def truncated_standard_moments(
    lower: float,
    upper: float,
    order: int,
) -> list[float]:
    """Return integrals of z^j phi(z) over [lower, upper], j <= order."""
    moments = [0.0] * (order + 1)
    moments[0] = normal_cdf(upper) - normal_cdf(lower)
    if order == 0:
        return moments
    moments[1] = normal_pdf(lower) - normal_pdf(upper)
    for degree in range(2, order + 1):
        lower_boundary = (
            lower ** (degree - 1) * normal_pdf(lower)
            if math.isfinite(lower)
            else 0.0
        )
        upper_boundary = (
            upper ** (degree - 1) * normal_pdf(upper)
            if math.isfinite(upper)
            else 0.0
        )
        moments[degree] = (
            lower_boundary
            - upper_boundary
            + (degree - 1) * moments[degree - 2]
        )
    return moments


def polynomial_normal_integral(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
    mean: float = 0.0,
) -> float:
    """Integrate sum_j coefficients[j] z^j against N(mean, 1)."""
    shifted_lower = lower - mean
    shifted_upper = upper - mean
    moments = truncated_standard_moments(
        shifted_lower,
        shifted_upper,
        len(coefficients) - 1,
    )
    total = 0.0
    for degree, coefficient in enumerate(coefficients):
        for power in range(degree + 1):
            total += (
                float(coefficient)
                * math.comb(degree, power)
                * mean ** (degree - power)
                * moments[power]
            )
    return total


def advantage(z_value: float) -> float:
    return G * z_value + 0.5 * (1.0 - z_value * z_value)


F_COEFFICIENTS = np.asarray([0.0, 0.5, G, -0.5])
F_SQUARED_COEFFICIENTS = np.convolve(F_COEFFICIENTS, F_COEFFICIENTS)
ADVANTAGE_ROOTS = (
    G - math.sqrt(G * G + 1.0),
    G + math.sqrt(G * G + 1.0),
)


def kept_intervals(delta: float) -> list[tuple[float, float]]:
    if delta == 0.0:
        return [(-math.inf, math.inf)]
    upper_ratio_boundary = (
        math.log1p(EPSILON) - 0.5 * delta * delta
    ) / delta
    lower_ratio_boundary = (
        math.log1p(-EPSILON) - 0.5 * delta * delta
    ) / delta
    cuts = sorted(
        {
            -math.inf,
            ADVANTAGE_ROOTS[0],
            ADVANTAGE_ROOTS[1],
            lower_ratio_boundary,
            upper_ratio_boundary,
            math.inf,
        }
    )
    intervals: list[tuple[float, float]] = []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        if lower == -math.inf:
            midpoint = upper - 1.0
        elif upper == math.inf:
            midpoint = lower + 1.0
        else:
            midpoint = 0.5 * (lower + upper)
        log_ratio = delta * midpoint + 0.5 * delta * delta
        positive_branch = (
            advantage(midpoint) >= 0.0
            and log_ratio <= math.log1p(EPSILON)
        )
        negative_branch = (
            advantage(midpoint) < 0.0
            and log_ratio >= math.log1p(-EPSILON)
        )
        if positive_branch or negative_branch:
            intervals.append((lower, upper))
    return intervals


def estimator_moments(delta: float) -> dict[str, float]:
    intervals = kept_intervals(delta)
    ppo_mean = sum(
        polynomial_normal_integral(F_COEFFICIENTS, lower, upper)
        for lower, upper in intervals
    )
    tis_boundary = (
        (math.log(TIS_CAP) - 0.5 * delta * delta) / delta
        if delta > 0.0
        else math.inf
    )
    tis_mean = polynomial_normal_integral(
        F_COEFFICIENTS,
        -math.inf,
        tis_boundary,
    ) + TIS_CAP * polynomial_normal_integral(
        F_COEFFICIENTS,
        tis_boundary,
        math.inf,
        mean=-delta,
    )
    is_tilted_second_moment = polynomial_normal_integral(
        F_SQUARED_COEFFICIENTS,
        -math.inf,
        math.inf,
        mean=delta,
    )
    is_single_second = math.exp(delta * delta) * is_tilted_second_moment
    tis_single_second = (
        math.exp(delta * delta)
        * polynomial_normal_integral(
            F_SQUARED_COEFFICIENTS,
            -math.inf,
            tis_boundary,
            mean=delta,
        )
        + TIS_CAP * TIS_CAP
        * polynomial_normal_integral(
            F_SQUARED_COEFFICIENTS,
            tis_boundary,
            math.inf,
            mean=-delta,
        )
    )
    ppo_single_second = math.exp(delta * delta) * sum(
        polynomial_normal_integral(
            F_SQUARED_COEFFICIENTS,
            lower,
            upper,
            mean=delta,
        )
        for lower, upper in intervals
    )

    row: dict[str, float] = {
        "delta": delta,
        "rho": math.exp(-delta * delta),
        "is_mean": G,
        "tis_mean": tis_mean,
        "ppo_mean": ppo_mean,
        "is_tilted_second_moment_M2": is_tilted_second_moment,
        "is_single_second_moment": is_single_second,
        "tis_single_second_moment": tis_single_second,
        "ppo_single_second_moment": ppo_single_second,
    }
    for rule in ("is", "tis", "ppo"):
        mean = row[f"{rule}_mean"]
        single_second = row[f"{rule}_single_second_moment"]
        variance = max(single_second - mean * mean, 0.0) / N
        second_moment = mean * mean + variance
        mse = (mean - G) ** 2 + variance
        certificate = (
            ETA * G * mean
            - 0.5 * SMOOTHNESS * ETA * ETA * second_moment
        )
        row[f"{rule}_variance"] = variance
        row[f"{rule}_second_moment"] = second_moment
        row[f"{rule}_mse"] = mse
        row[f"{rule}_certificate"] = certificate
        # The smoothness certificate is exact for this quadratic objective.
        row[f"{rule}_expected_gain"] = certificate

    # This finite-moment specialization replaces a uniform bound on f by the
    # exact tilted second moment M2(delta). It yields
    #   MSE(IS) <= M2(delta) / (N rho(delta))
    # and the corresponding Markov error radius at confidence 1-delta.
    row["is_finite_moment_mse_bound"] = (
        is_tilted_second_moment / (N * row["rho"])
    )
    row["is_finite_moment_error_radius"] = math.sqrt(
        row["is_finite_moment_mse_bound"] / DELTA_CONFIDENCE
    )

    for rule in ("tis", "ppo"):
        mse_reduction = row["is_mse"] - row[f"{rule}_mse"]
        second_moment_reduction = (
            row["is_second_moment"] - row[f"{rule}_second_moment"]
        )
        discounted_reduction = (
            1.0 - SMOOTHNESS * ETA
        ) * second_moment_reduction
        crossover_margin = mse_reduction - discounted_reduction
        certificate_gap = (
            row[f"{rule}_certificate"] - row["is_certificate"]
        )
        row[f"{rule}_mse_reduction_vs_is"] = mse_reduction
        row[f"{rule}_second_moment_reduction_vs_is"] = (
            second_moment_reduction
        )
        row[
            f"{rule}_discounted_second_moment_reduction_vs_is"
        ] = discounted_reduction
        row[f"{rule}_crossover_margin"] = crossover_margin
        row[f"{rule}_certificate_gap_vs_is"] = certificate_gap
        row[f"{rule}_certificate_identity_error"] = certificate_gap - (
            0.5 * ETA * crossover_margin
        )
        row[f"{rule}_alignment_loss_vs_is"] = (
            G * G - G * row[f"{rule}_mean"]
        )
        row[f"{rule}_smoothness_penalty_reduction_vs_is"] = (
            0.5
            * SMOOTHNESS
            * ETA
            * second_moment_reduction
        )

    candidates = {
        "no update": 0.0,
        "IS": row["is_certificate"],
        f"TIS {TIS_CAP:g}": row["tis_certificate"],
        "PPO": row["ppo_certificate"],
    }
    row["oracle_expected_gain"] = max(candidates.values())
    row["oracle_rule"] = max(candidates, key=candidates.get)
    return row


def bisect_root(function, lower: float, upper: float) -> float:
    lower_value = function(lower)
    upper_value = function(upper)
    if lower_value * upper_value > 0.0:
        raise ValueError(f"root is not bracketed on [{lower}, {upper}]")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = function(midpoint)
        if abs(midpoint_value) < 1e-13 or upper - lower < 1e-12:
            return midpoint
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
            upper_value = midpoint_value
        else:
            lower = midpoint
            lower_value = midpoint_value
    return 0.5 * (lower + upper)


def crossover_summary() -> dict[str, float]:
    mse_delta = bisect_root(
        lambda delta: (
            estimator_moments(delta)["ppo_mse"]
            - estimator_moments(delta)["is_mse"]
        ),
        1.2,
        1.7,
    )
    certificate_delta = bisect_root(
        lambda delta: (
            estimator_moments(delta)["ppo_certificate"]
            - estimator_moments(delta)["is_certificate"]
        ),
        1.6,
        2.0,
    )
    tis_certificate_delta = bisect_root(
        lambda delta: estimator_moments(delta)["tis_crossover_margin"],
        1.3,
        1.7,
    )
    is_zero_delta = bisect_root(
        lambda delta: estimator_moments(delta)["is_certificate"],
        1.6,
        2.0,
    )
    ppo_zero_delta = bisect_root(
        lambda delta: estimator_moments(delta)["ppo_certificate"],
        2.0,
        2.3,
    )
    return {
        "mse_crossover_delta": mse_delta,
        "mse_crossover_rho": math.exp(-mse_delta * mse_delta),
        "certificate_crossover_delta": certificate_delta,
        "certificate_crossover_rho": math.exp(
            -certificate_delta * certificate_delta
        ),
        "tis_certificate_crossover_delta": tis_certificate_delta,
        "tis_certificate_crossover_rho": math.exp(
            -tis_certificate_delta * tis_certificate_delta
        ),
        "is_zero_delta": is_zero_delta,
        "is_zero_rho": math.exp(-is_zero_delta * is_zero_delta),
        "ppo_zero_delta": ppo_zero_delta,
        "ppo_zero_rho": math.exp(-ppo_zero_delta * ppo_zero_delta),
    }


def validate_crossovers(summary: dict[str, float]) -> None:
    mse_state = estimator_moments(summary["mse_crossover_delta"])
    certificate_state = estimator_moments(
        summary["certificate_crossover_delta"]
    )
    tis_state = estimator_moments(
        summary["tis_certificate_crossover_delta"]
    )
    stop_state = estimator_moments(summary["ppo_zero_delta"])
    residuals = {
        "IS-PPO MSE crossover": abs(
            mse_state["is_mse"] - mse_state["ppo_mse"]
        ),
        "IS-PPO certificate crossover": abs(
            certificate_state["is_certificate"]
            - certificate_state["ppo_certificate"]
        ),
        "IS-TIS certificate crossover": abs(
            tis_state["is_certificate"] - tis_state["tis_certificate"]
        ),
        "IS-TIS corollary crossover": abs(
            tis_state["tis_crossover_margin"]
        ),
        "PPO zero certificate": abs(stop_state["ppo_certificate"]),
    }
    for name, residual in residuals.items():
        if residual > 1e-10:
            raise FloatingPointError(f"{name} residual is {residual}")

    intermediate = estimator_moments(1.6)
    if not (
        intermediate["ppo_mse"] < intermediate["is_mse"]
        and intermediate["is_certificate"]
        > intermediate["ppo_certificate"]
    ):
        raise AssertionError("the intermediate oracle-separation regime failed")
    low_support = estimator_moments(2.1)
    if not (
        low_support["ppo_certificate"] > low_support["is_certificate"]
        and low_support["ppo_certificate"] > 0.0
    ):
        raise AssertionError("the low-support PPO regime failed")

    for state in (mse_state, certificate_state, tis_state, stop_state):
        for rule in ("tis", "ppo"):
            if abs(state[f"{rule}_certificate_identity_error"]) > 1e-10:
                raise FloatingPointError(
                    f"{rule.upper()} certificate identity failed"
                )


def monte_carlo_reliability(
    delta: float,
    error_radius: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Check the finite-moment reliability event and adaptive-step guarantee."""
    error_event_count = 0
    theorem_inequality_count = 0
    positive_certificate_count = 0
    actual_improvement_count = 0
    error_event_violations = 0
    batches_seen = 0

    while batches_seen < MONTE_CARLO_BATCHES:
        chunk_size = min(
            MONTE_CARLO_CHUNK_SIZE,
            MONTE_CARLO_BATCHES - batches_seen,
        )
        z_values = rng.normal(
            loc=-delta,
            scale=1.0,
            size=(chunk_size, N),
        )
        weights = np.exp(delta * z_values + 0.5 * delta * delta)
        contributions = weights * (
            G * z_values * z_values
            + 0.5 * (z_values - z_values * z_values * z_values)
        )
        estimates = np.mean(contributions, axis=1)
        absolute_estimates = np.abs(estimates)
        errors = np.abs(estimates - G)
        error_events = errors <= error_radius

        positive_parts = np.maximum(absolute_estimates - error_radius, 0.0)
        gammas = np.zeros_like(estimates)
        nonzero = absolute_estimates > error_radius
        gammas[nonzero] = (
            1.0 - error_radius / absolute_estimates[nonzero]
        ) / SMOOTHNESS
        actual_gains = (
            G * gammas * estimates
            - 0.5 * SMOOTHNESS * gammas * gammas * estimates * estimates
        )
        theorem_bounds = 0.5 * positive_parts * positive_parts / SMOOTHNESS
        tolerance = 1e-12 * (
            1.0 + np.abs(actual_gains) + np.abs(theorem_bounds)
        )
        theorem_events = actual_gains + tolerance >= theorem_bounds

        error_event_count += int(np.count_nonzero(error_events))
        theorem_inequality_count += int(np.count_nonzero(theorem_events))
        positive_certificate_count += int(np.count_nonzero(nonzero))
        actual_improvement_count += int(np.count_nonzero(actual_gains > 0.0))
        error_event_violations += int(
            np.count_nonzero(error_events & ~theorem_events)
        )
        batches_seen += chunk_size

    if error_event_violations:
        raise AssertionError(
            "the adaptive-step inequality failed on the reliability event"
        )

    return {
        "mc_error_event_coverage": (
            error_event_count / MONTE_CARLO_BATCHES
        ),
        "mc_theorem_inequality_coverage": (
            theorem_inequality_count / MONTE_CARLO_BATCHES
        ),
        "mc_positive_certificate_fraction": (
            positive_certificate_count / MONTE_CARLO_BATCHES
        ),
        "mc_actual_improvement_fraction": (
            actual_improvement_count / MONTE_CARLO_BATCHES
        ),
        "mc_error_event_theorem_violations": float(error_event_violations),
    }


def validate_rows(rows: list[dict[str, float | str]]) -> None:
    tolerance = 2e-10
    for row in rows:
        finite_moment_gap = (
            row["is_finite_moment_mse_bound"] - row["is_mse"]
        )
        if abs(finite_moment_gap - G * G / N) > tolerance:
            raise FloatingPointError("the finite-moment MSE identity failed")
        for rule in ("tis", "ppo"):
            if abs(row[f"{rule}_certificate_identity_error"]) > tolerance:
                raise FloatingPointError(
                    f"{rule.upper()} certificate identity failed on the grid"
                )
            if abs(
                row[f"{rule}_expected_gain"]
                - row[f"{rule}_certificate"]
            ) > tolerance:
                raise FloatingPointError(
                    f"{rule.upper()} exact-gain identity failed"
                )
        if abs(row["is_expected_gain"] - row["is_certificate"]) > tolerance:
            raise FloatingPointError("IS exact-gain identity failed")
        if row["mc_error_event_coverage"] < 0.895:
            raise AssertionError(
                "Monte Carlo reliability-event coverage fell below tolerance"
            )
        if (
            row["mc_theorem_inequality_coverage"] + 1e-12
            < row["mc_error_event_coverage"]
        ):
            raise AssertionError(
                "theorem coverage fell below reliability-event coverage"
            )
        if (
            row["mc_actual_improvement_fraction"]
            > row["mc_positive_certificate_fraction"] + 1e-12
        ):
            raise AssertionError(
                "positive improvement occurred without a positive certificate"
            )
        if (
            row["mc_positive_certificate_fraction"]
            - row["mc_actual_improvement_fraction"]
            > DELTA_CONFIDENCE + 0.005
        ):
            raise AssertionError(
                "the empirical certificate-to-improvement gap exceeds delta"
            )

    if not all(
        row["tis_mse"] <= min(row["is_mse"], row["ppo_mse"])
        for row in rows
    ):
        raise AssertionError("TIS 3 is not the MSE oracle on the displayed path")
    if not all(
        row["tis_certificate"] >= row["ppo_certificate"]
        for row in rows
    ):
        raise AssertionError("TIS 3 does not dominate PPO on the displayed path")
    observed_oracles = {str(row["oracle_rule"]) for row in rows}
    if observed_oracles != {"IS", f"TIS {TIS_CAP:g}"}:
        raise AssertionError(f"unexpected oracle rules: {observed_oracles}")


def make_figure(
    rows: list[dict[str, float | str]],
    summary: dict[str, float],
) -> None:
    use_paper_style()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.0,
            "legend.handlelength": 2.35,
            "legend.labelspacing": 0.38,
            "legend.borderaxespad": 0.55,
            "axes.linewidth": 0.85,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.34,
            "lines.linewidth": 2.2,
            "lines.markersize": 4.6,
            "xtick.major.size": 3.4,
            "ytick.major.size": 3.4,
        }
    )
    ordered = sorted(rows, key=lambda item: item["rho"])
    rho = np.asarray([float(row["rho"]) for row in ordered])
    is_mse = np.asarray([float(row["is_mse"]) for row in ordered])
    is_mse_bound = np.asarray(
        [float(row["is_finite_moment_mse_bound"]) for row in ordered]
    )
    positive_fraction = np.asarray(
        [float(row["mc_positive_certificate_fraction"]) for row in ordered]
    )
    improvement_fraction = np.asarray(
        [float(row["mc_actual_improvement_fraction"]) for row in ordered]
    )
    tis_gain_gap = np.asarray(
        [float(row["tis_certificate_gap_vs_is"]) for row in ordered]
    )
    ppo_gain_gap = np.asarray(
        [float(row["ppo_certificate_gap_vs_is"]) for row in ordered]
    )
    is_gain = np.asarray(
        [float(row["is_expected_gain"]) for row in ordered]
    )
    tis_gain = np.asarray(
        [float(row["tis_expected_gain"]) for row in ordered]
    )
    ppo_gain = np.asarray(
        [float(row["ppo_expected_gain"]) for row in ordered]
    )

    rho_tis = summary["tis_certificate_crossover_rho"]
    rho_ppo = summary["certificate_crossover_rho"]
    tis_root = estimator_moments(summary["tis_certificate_crossover_delta"])
    ppo_root = estimator_moments(summary["certificate_crossover_delta"])

    reliability_figure, (risk_axis, improvement_axis) = plt.subplots(
        1, 2, figsize=(FULL, 2.45)
    )
    risk_axis.plot(
        rho,
        is_mse,
        color=IS_COLOR,
        marker="o",
        markevery=(0, 9),
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Exact IS MSE",
    )
    risk_axis.plot(
        rho,
        is_mse_bound,
        color=NEUTRAL_COLOR,
        linestyle=(0, (5.0, 2.2)),
        linewidth=2.0,
        label="Finite-moment upper bound",
    )
    risk_axis.fill_between(
        rho,
        is_mse,
        is_mse_bound,
        color=IS_COLOR,
        alpha=0.13,
        linewidth=0.0,
    )
    risk_axis.set_xscale("log")
    risk_axis.set_yscale("log")
    risk_axis.set_xlim(MINIMUM_RHO, MAXIMUM_RHO)
    risk_axis.set_xlabel(r"ESS $\rho$")
    risk_axis.set_ylabel("IS gradient MSE")
    risk_axis.set_title("(a)", loc="left", fontweight="bold", pad=7)
    risk_axis.legend(loc="lower left", frameon=False)

    improvement_axis.plot(
        rho,
        improvement_fraction,
        color=IS_COLOR,
        marker="o",
        markevery=(0, 9),
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Actual improvement",
    )
    improvement_axis.plot(
        rho,
        positive_fraction,
        color=PPO_COLOR,
        linestyle=(0, (5.0, 2.2)),
        marker="D",
        markevery=(4, 9),
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Positive certificate",
    )
    improvement_axis.fill_between(
        rho,
        improvement_fraction,
        positive_fraction,
        where=positive_fraction >= improvement_fraction,
        color=NEUTRAL_COLOR,
        alpha=0.10,
        linewidth=0.0,
    )
    improvement_axis.set_xscale("log")
    improvement_axis.set_xlim(MINIMUM_RHO, MAXIMUM_RHO)
    improvement_axis.set_ylim(0.0, 0.67)
    improvement_axis.set_yticks([0.0, 0.2, 0.4, 0.6])
    improvement_axis.yaxis.set_major_formatter(
        PercentFormatter(1.0, decimals=0)
    )
    improvement_axis.set_xlabel(r"ESS $\rho$")
    improvement_axis.set_ylabel("Fraction of batches")
    improvement_axis.set_title("(b)", loc="left", fontweight="bold", pad=7)
    improvement_axis.legend(loc="upper left", frameon=False)

    for axis in (risk_axis, improvement_axis):
        axis.grid(True, which="major", color="#C9D1D9", alpha=0.48)
        axis.grid(False, which="minor")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reliability_base = FIGURES_DIR / "gaussian_is_reliability"
    reliability_figure.savefig(reliability_base.with_suffix(".pdf"))
    reliability_figure.savefig(reliability_base.with_suffix(".png"), dpi=300)
    plt.close(reliability_figure)

    choice_figure, (crossover_axis, gain_axis) = plt.subplots(
        1, 2, figsize=(FULL, 2.55)
    )
    crossover_axis.axhline(
        0.0,
        color=NEUTRAL_COLOR,
        linewidth=1.25,
        zorder=1,
    )
    crossover_axis.plot(
        rho,
        tis_gain_gap,
        color=TIS_COLOR,
        marker="D",
        markevery=(0, 9),
        markerfacecolor="white",
        markeredgewidth=1.1,
        label=f"TIS {TIS_CAP:g} vs. IS",
    )
    crossover_axis.plot(
        rho,
        ppo_gain_gap,
        color=PPO_COLOR,
        linestyle=(0, (5.0, 2.2)),
        marker="s",
        markevery=(4, 9),
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="PPO vs. IS",
    )
    for crossover_rho, color, label, x_offset in (
        (rho_tis, TIS_COLOR, "TIS 3 = IS", 5),
        (rho_ppo, PPO_COLOR, "PPO = IS", -5),
    ):
        crossover_axis.axvline(
            crossover_rho,
            color=color,
            linestyle=(0, (1.2, 2.0)),
            linewidth=1.55,
            alpha=0.9,
        )
        crossover_axis.annotate(
            f"{label}\n{format_sig(crossover_rho)}",
            xy=(crossover_rho, 0.96),
            xycoords=("data", "axes fraction"),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="top",
            rotation=90,
            fontsize=8.8,
            fontweight="semibold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )
    crossover_axis.set_xscale("log")
    crossover_axis.set_yscale("symlog", linthresh=0.1, linscale=1.45)
    crossover_axis.set_yticks([-1.0, -0.1, 0.0, 0.1, 1.0, 10.0])
    crossover_axis.set_xlim(MINIMUM_RHO, MAXIMUM_RHO)
    crossover_axis.set_xlabel(r"ESS $\rho$")
    crossover_axis.set_ylabel("Expected gain relative to IS")
    crossover_axis.set_title("(a)", loc="left", fontweight="bold", pad=7)
    crossover_axis.legend(loc="lower left", frameon=False)

    gain_axis.axhline(
        0.0,
        color=NEUTRAL_COLOR,
        linewidth=1.15,
        zorder=1,
    )
    gain_axis.plot(
        rho,
        is_gain,
        color=IS_COLOR,
        marker="o",
        markevery=(0, 8),
        markerfacecolor="white",
        markeredgewidth=1.15,
        label="IS",
    )
    gain_axis.plot(
        rho,
        tis_gain,
        color=TIS_COLOR,
        marker="D",
        markevery=(3, 8),
        markerfacecolor="white",
        markeredgewidth=1.05,
        label=f"TIS {TIS_CAP:g}",
    )
    gain_axis.plot(
        rho,
        ppo_gain,
        color=PPO_COLOR,
        linestyle=(0, (5.0, 2.2)),
        marker="s",
        markevery=(6, 8),
        markerfacecolor="white",
        markeredgewidth=1.05,
        label="PPO",
    )
    for crossover_rho, color in (
        (rho_ppo, PPO_COLOR),
        (rho_tis, TIS_COLOR),
    ):
        gain_axis.axvline(
            crossover_rho,
            color=color,
            linestyle=(0, (1.2, 2.0)),
            linewidth=1.55,
            alpha=0.9,
        )
    gain_axis.scatter(
        [rho_ppo],
        [ppo_root["is_expected_gain"]],
        s=58,
        facecolor="white",
        edgecolor=PPO_COLOR,
        linewidth=1.7,
        zorder=6,
    )
    gain_axis.scatter(
        [rho_tis],
        [tis_root["is_expected_gain"]],
        s=58,
        facecolor="white",
        edgecolor=TIS_COLOR,
        linewidth=1.7,
        zorder=6,
    )
    gain_axis.set_xlim(0.025, 0.16)
    gain_axis.set_ylim(-0.35, 1.12)
    gain_axis.set_xticks([0.04, 0.08, 0.12, 0.16])
    gain_axis.set_xlabel(r"ESS $\rho$")
    gain_axis.set_ylabel("Exact expected gain")
    gain_axis.set_title("(b)", loc="left", fontweight="bold", pad=7)
    gain_axis.legend(
        loc="lower right",
        frameon=False,
        ncol=1,
    )

    for axis in (crossover_axis, gain_axis):
        axis.grid(True, which="major", color="#C9D1D9", alpha=0.48)
        axis.grid(False, which="minor")

    choice_base = FIGURES_DIR / "gaussian_estimator_choice"
    choice_figure.savefig(choice_base.with_suffix(".pdf"))
    choice_figure.savefig(choice_base.with_suffix(".png"), dpi=300)
    plt.close(choice_figure)


def main() -> None:
    summary = crossover_summary()
    validate_crossovers(summary)
    minimum_delta = math.sqrt(-math.log(MAXIMUM_RHO))
    maximum_delta = math.sqrt(-math.log(MINIMUM_RHO))
    displayed_rhos = np.geomspace(
        MINIMUM_RHO,
        MAXIMUM_RHO,
        MONTE_CARLO_POINTS,
    )
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    for rho in displayed_rhos:
        delta = math.sqrt(-math.log(float(rho)))
        row = estimator_moments(delta)
        row.update(
            monte_carlo_reliability(
                delta,
                float(row["is_finite_moment_error_radius"]),
                rng,
            )
        )
        rows.append(row)
    validate_rows(rows)

    representative_rhos = (0.9, 0.1, 0.01)
    representative = [
        estimator_moments(math.sqrt(-math.log(rho)))
        for rho in representative_rhos
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "gaussian_quadratic_one_step.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "gaussian_quadratic_one_step_summary.json"
    payload = {
        "parameters": {
            "gradient": G,
            "batch_size": N,
            "ppo_epsilon": EPSILON,
            "tis_cap": TIS_CAP,
            "step_size": ETA,
            "smoothness": SMOOTHNESS,
        },
        "finite_moment_reliability": {
            "delta_confidence": DELTA_CONFIDENCE,
            "target_coverage": 1.0 - DELTA_CONFIDENCE,
            "monte_carlo_points": MONTE_CARLO_POINTS,
            "batches_per_point": MONTE_CARLO_BATCHES,
            "batch_size": N,
            "seed": MONTE_CARLO_SEED,
            "minimum_error_event_coverage": min(
                float(row["mc_error_event_coverage"]) for row in rows
            ),
            "minimum_theorem_inequality_coverage": min(
                float(row["mc_theorem_inequality_coverage"])
                for row in rows
            ),
            "maximum_positive_certificate_fraction": max(
                float(row["mc_positive_certificate_fraction"])
                for row in rows
            ),
            "maximum_actual_improvement_fraction": max(
                float(row["mc_actual_improvement_fraction"])
                for row in rows
            ),
            "maximum_certificate_improvement_gap": max(
                float(row["mc_positive_certificate_fraction"])
                - float(row["mc_actual_improvement_fraction"])
                for row in rows
            ),
        },
        "displayed_branch": {
            "minimum_delta": minimum_delta,
            "maximum_delta": maximum_delta,
            "minimum_rho": MINIMUM_RHO,
            "maximum_rho": MAXIMUM_RHO,
        },
        "crossovers": summary,
        "oracle": {
            "mse_oracle_on_displayed_branch": f"TIS {TIS_CAP:g}",
            "certificate_oracle_high_ess": "IS",
            "certificate_oracle_low_ess": f"TIS {TIS_CAP:g}",
            "certificate_crossover_rho": summary[
                "tis_certificate_crossover_rho"
            ],
            "ppo_is_never_global": True,
            "no_update_is_never_global": True,
        },
        "representative_states": representative,
        "calculation": (
            "closed-form truncated Gaussian polynomial moments plus "
            "fixed-seed Monte Carlo reliability diagnostics"
        ),
        "monte_carlo_replications_per_rho": MONTE_CARLO_BATCHES,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(rows, summary)

    for name, value in summary.items():
        print(f"{name}: {value:.12g}")
    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print("wrote figures/gaussian_is_reliability.{pdf,png}")
    print("wrote figures/gaussian_estimator_choice.{pdf,png}")


if __name__ == "__main__":
    main()
