"""Exactly solvable one-step Raw/PPO comparison for a Gaussian bandit.

The rollout and current policies are unit-variance Gaussians separated by
``delta``.  All estimator moments are evaluated from truncated Gaussian
polynomial moments, so the reported curves contain no Monte Carlo error.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "simulation" / "results"
FIGURES_DIR = ROOT / "figures"

G = 2.0
N = 32
EPSILON = 0.2
ETA = 0.4
SMOOTHNESS = 1.0
MINIMUM_DELTA = 0.2
MINIMUM_RHO = 0.005

RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
DIFFERENCE_COLOR = "#7A5195"
NEUTRAL_COLOR = "#59636E"
GRID_COLOR = "#D9DEE8"


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
    raw_single_second = math.exp(delta * delta) * polynomial_normal_integral(
        F_SQUARED_COEFFICIENTS,
        -math.inf,
        math.inf,
        mean=delta,
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
        "raw_mean": G,
        "ppo_mean": ppo_mean,
        "raw_single_second_moment": raw_single_second,
        "ppo_single_second_moment": ppo_single_second,
    }
    for rule in ("raw", "ppo"):
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
            - estimator_moments(delta)["raw_mse"]
        ),
        1.2,
        1.7,
    )
    certificate_delta = bisect_root(
        lambda delta: (
            estimator_moments(delta)["ppo_certificate"]
            - estimator_moments(delta)["raw_certificate"]
        ),
        1.6,
        2.0,
    )
    raw_zero_delta = bisect_root(
        lambda delta: estimator_moments(delta)["raw_certificate"],
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
        "raw_zero_delta": raw_zero_delta,
        "raw_zero_rho": math.exp(-raw_zero_delta * raw_zero_delta),
        "ppo_zero_delta": ppo_zero_delta,
        "ppo_zero_rho": math.exp(-ppo_zero_delta * ppo_zero_delta),
    }


def validate_crossovers(summary: dict[str, float]) -> None:
    mse_state = estimator_moments(summary["mse_crossover_delta"])
    certificate_state = estimator_moments(
        summary["certificate_crossover_delta"]
    )
    stop_state = estimator_moments(summary["ppo_zero_delta"])
    residuals = {
        "MSE crossover": abs(mse_state["raw_mse"] - mse_state["ppo_mse"]),
        "certificate crossover": abs(
            certificate_state["raw_certificate"]
            - certificate_state["ppo_certificate"]
        ),
        "PPO zero certificate": abs(stop_state["ppo_certificate"]),
    }
    for name, residual in residuals.items():
        if residual > 1e-10:
            raise FloatingPointError(f"{name} residual is {residual}")

    intermediate = estimator_moments(1.6)
    if not (
        intermediate["ppo_mse"] < intermediate["raw_mse"]
        and intermediate["raw_certificate"]
        > intermediate["ppo_certificate"]
    ):
        raise AssertionError("the intermediate oracle-separation regime failed")
    low_support = estimator_moments(2.1)
    if not (
        low_support["ppo_certificate"] > low_support["raw_certificate"]
        and low_support["ppo_certificate"] > 0.0
    ):
        raise AssertionError("the low-support PPO regime failed")


def make_figure(rows: list[dict[str, float]], summary: dict[str, float]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    ordered = sorted(rows, key=lambda item: item["rho"])
    rho = np.asarray([row["rho"] for row in ordered])
    raw_mse = np.asarray([row["raw_mse"] for row in ordered])
    ppo_mse = np.asarray([row["ppo_mse"] for row in ordered])
    raw_gain = np.asarray([row["raw_expected_gain"] for row in ordered])
    ppo_gain = np.asarray([row["ppo_expected_gain"] for row in ordered])

    rho_mse = summary["mse_crossover_rho"]
    rho_certificate = summary["certificate_crossover_rho"]
    rho_stop = summary["ppo_zero_rho"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.75, 2.7),
        constrained_layout=True,
    )
    risk_axis, gain_axis = axes

    risk_axis.axvspan(
        rho_certificate,
        rho_mse,
        color=DIFFERENCE_COLOR,
        alpha=0.11,
        linewidth=0.0,
    )
    risk_axis.plot(rho, raw_mse, color=RAW_COLOR, linewidth=1.8, label="Raw")
    risk_axis.plot(rho, ppo_mse, color=PPO_COLOR, linewidth=1.8, label="PPO")
    risk_axis.axvline(rho_mse, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    risk_axis.axvline(
        rho_certificate,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.0,
    )
    risk_axis.set_xscale("log")
    risk_axis.set_yscale("log")
    maximum_rho = math.exp(-MINIMUM_DELTA * MINIMUM_DELTA)
    risk_axis.set_xlim(MINIMUM_RHO, maximum_rho)
    risk_axis.set_xlabel(r"Normalized ESS $\rho$ (log scale)")
    risk_axis.set_ylabel("Gradient MSE")
    risk_axis.set_title("(a) Estimator MSE")
    risk_axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.6)
    risk_axis.legend(frameon=False, loc="upper right")
    risk_axis.text(
        math.sqrt(rho_certificate * rho_mse),
        0.28,
        "PPO lower MSE,\nRaw larger gain",
        color=DIFFERENCE_COLOR,
        ha="center",
        va="bottom",
        fontsize=7,
    )

    gain_axis.axvspan(MINIMUM_RHO, rho_stop, color=NEUTRAL_COLOR, alpha=0.09, linewidth=0.0)
    gain_axis.axvspan(rho_stop, rho_certificate, color=PPO_COLOR, alpha=0.09, linewidth=0.0)
    gain_axis.axvspan(rho_certificate, maximum_rho, color=RAW_COLOR, alpha=0.06, linewidth=0.0)
    gain_axis.plot(rho, raw_gain, color=RAW_COLOR, linewidth=1.8, label="Raw")
    gain_axis.plot(rho, ppo_gain, color=PPO_COLOR, linewidth=1.8, label="PPO")
    gain_axis.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    gain_axis.axvline(rho_mse, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    gain_axis.axvline(
        rho_certificate,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.0,
    )
    gain_axis.set_xscale("log")
    gain_axis.set_xlim(MINIMUM_RHO, maximum_rho)
    gain_axis.set_ylim(-4.3, 1.45)
    gain_axis.set_xlabel(r"Normalized ESS $\rho$ (log scale)")
    gain_axis.set_ylabel("Exact expected one-step gain")
    gain_axis.set_title("(b) Exact one-step improvement")
    gain_axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.6)
    gain_axis.legend(frameon=False, loc="lower right")
    gain_axis.text(0.23, 1.18, "Raw", color=RAW_COLOR, ha="center", fontsize=7)
    gain_axis.text(0.021, 1.18, "PPO", color=PPO_COLOR, ha="center", fontsize=7)
    gain_axis.text(0.0073, 1.18, "No update", color=NEUTRAL_COLOR, ha="center", fontsize=7)

    for axis in axes:
        axis.annotate(
            r"$\rho_{\rm MSE}=0.123$",
            xy=(rho_mse, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=6.5,
            color=NEUTRAL_COLOR,
        )
        axis.annotate(
            r"$\rho_{B}=0.039$",
            xy=(rho_certificate, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=6.5,
            color=NEUTRAL_COLOR,
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_base = FIGURES_DIR / "gaussian_quadratic_one_step"
    fig.savefig(figure_base.with_suffix(".pdf"))
    fig.savefig(figure_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    summary = crossover_summary()
    validate_crossovers(summary)
    maximum_delta = math.sqrt(-math.log(MINIMUM_RHO))
    rows = [
        estimator_moments(float(delta))
        for delta in np.linspace(MINIMUM_DELTA, maximum_delta, 801)
    ]

    representative_rhos = (0.9607894392, 0.0773047404, 0.0121551783)
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
            "step_size": ETA,
            "smoothness": SMOOTHNESS,
        },
        "displayed_branch": {
            "minimum_delta": MINIMUM_DELTA,
            "maximum_delta": maximum_delta,
            "minimum_rho": MINIMUM_RHO,
            "maximum_rho": math.exp(-MINIMUM_DELTA * MINIMUM_DELTA),
        },
        "crossovers": summary,
        "representative_states": representative,
        "calculation": "closed-form truncated Gaussian polynomial moments",
        "monte_carlo_replications": 0,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(rows, summary)

    for name, value in summary.items():
        print(f"{name}: {value:.12g}")
    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print("wrote figures/gaussian_quadratic_one_step.{pdf,png}")


if __name__ == "__main__":
    main()
