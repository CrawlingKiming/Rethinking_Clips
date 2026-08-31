"""Exactly solvable one-step IS/TIS/PPO comparison for a Gaussian bandit.

The rollout and current policies are unit-variance Gaussians separated by
``delta``.  All estimator moments are evaluated from truncated Gaussian
polynomial moments, so the reported curves contain no Monte Carlo error.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "simulation" / "results"
FIGURES_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "figures" / "coding"))

import matplotlib.pyplot as plt
from paperstyle import FAM, FULL, use_paper_style

G = 2.0
N = 32
EPSILON = 0.2
TIS_CAP = 1.0 + EPSILON
ETA = 0.4
SMOOTHNESS = 1.0
MINIMUM_DELTA = 0.2
MINIMUM_RHO = 0.005

IS_COLOR = FAM[0]
TIS_COLOR = FAM[2]
PPO_COLOR = FAM[1]
NEUTRAL_COLOR = "#59636E"


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
    raw_single_second = math.exp(delta * delta) * polynomial_normal_integral(
        F_SQUARED_COEFFICIENTS,
        -math.inf,
        math.inf,
        mean=delta,
    )
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
        "raw_mean": G,
        "tis_mean": tis_mean,
        "ppo_mean": ppo_mean,
        "raw_single_second_moment": raw_single_second,
        "tis_single_second_moment": tis_single_second,
        "ppo_single_second_moment": ppo_single_second,
    }
    for rule in ("raw", "tis", "ppo"):
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
    tis_certificate_delta = bisect_root(
        lambda delta: (
            estimator_moments(delta)["tis_certificate"]
            - estimator_moments(delta)["raw_certificate"]
        ),
        1.3,
        1.7,
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
        "tis_certificate_crossover_delta": tis_certificate_delta,
        "tis_certificate_crossover_rho": math.exp(
            -tis_certificate_delta * tis_certificate_delta
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
    tis_state = estimator_moments(
        summary["tis_certificate_crossover_delta"]
    )
    residuals = {
        "MSE crossover": abs(mse_state["raw_mse"] - mse_state["ppo_mse"]),
        "certificate crossover": abs(
            certificate_state["raw_certificate"]
            - certificate_state["ppo_certificate"]
        ),
        "PPO zero certificate": abs(stop_state["ppo_certificate"]),
        "IS-TIS certificate crossover": abs(
            tis_state["raw_certificate"] - tis_state["tis_certificate"]
        ),
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
    use_paper_style()
    ordered = sorted(rows, key=lambda item: item["rho"])
    rho = np.asarray([row["rho"] for row in ordered])
    raw_mse = np.asarray([row["raw_mse"] for row in ordered])
    tis_mse = np.asarray([row["tis_mse"] for row in ordered])
    ppo_mse = np.asarray([row["ppo_mse"] for row in ordered])
    raw_alignment = np.asarray([row["raw_mean"] / G for row in ordered])
    tis_alignment = np.asarray([row["tis_mean"] / G for row in ordered])
    ppo_alignment = np.asarray([row["ppo_mean"] / G for row in ordered])
    raw_gain = np.asarray([row["raw_expected_gain"] for row in ordered])
    tis_gain = np.asarray([row["tis_expected_gain"] for row in ordered])
    ppo_gain = np.asarray([row["ppo_expected_gain"] for row in ordered])

    rho_mse = summary["mse_crossover_rho"]
    rho_certificate = summary["certificate_crossover_rho"]
    rho_tis = summary["tis_certificate_crossover_rho"]
    rho_stop = summary["ppo_zero_rho"]
    maximum_rho = math.exp(-MINIMUM_DELTA * MINIMUM_DELTA)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL, 2.55),
    )
    risk_axis, alignment_axis, gain_axis = axes

    lines = [
        ("IS", IS_COLOR, raw_mse, raw_alignment, raw_gain),
        (rf"TIS ($c={TIS_CAP:g}$)", TIS_COLOR, tis_mse, tis_alignment, tis_gain),
        ("PPO", PPO_COLOR, ppo_mse, ppo_alignment, ppo_gain),
    ]
    for label, color, risk, alignment, gain in lines:
        risk_axis.plot(rho, risk, color=color, label=label)
        alignment_axis.plot(rho, alignment, color=color, label=label)
        gain_axis.plot(rho, gain, color=color, label=label)

    risk_axis.axvline(rho_mse, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.9)
    risk_axis.set_xscale("log")
    risk_axis.set_yscale("log")
    risk_axis.set_xlim(MINIMUM_RHO, maximum_rho)
    risk_axis.set_xlabel(r"normalized ESS $\rho$")
    risk_axis.set_ylabel("Gradient MSE")
    risk_axis.set_title("(a) Estimation error", loc="left")
    risk_axis.annotate(
        r"$\rho_{\rm MSE}=0.123$",
        xy=(rho_mse, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        rotation=90,
        fontsize=6.2,
        color=NEUTRAL_COLOR,
    )

    alignment_axis.set_xscale("log")
    alignment_axis.set_xlim(MINIMUM_RHO, maximum_rho)
    alignment_axis.set_ylim(-0.02, 1.05)
    alignment_axis.set_xlabel(r"normalized ESS $\rho$")
    alignment_axis.set_ylabel(r"Retained alignment $\mu_u/g$")
    alignment_axis.set_title("(b) Signal retained", loc="left")

    gain_axis.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    gain_axis.axvline(
        rho_certificate,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=0.9,
    )
    gain_axis.axvline(
        rho_stop,
        color=NEUTRAL_COLOR,
        linestyle="-.",
        linewidth=0.9,
    )
    gain_axis.set_xscale("log")
    gain_axis.set_xlim(MINIMUM_RHO, maximum_rho)
    gain_axis.set_ylim(-4.3, 1.45)
    gain_axis.set_xlabel(r"normalized ESS $\rho$")
    gain_axis.set_ylabel(r"Exact gain $B_u(\eta)$")
    gain_axis.set_title("(c) One-step improvement", loc="left")
    tis_state = estimator_moments(summary["tis_certificate_crossover_delta"])
    gain_axis.scatter(
        [rho_tis],
        [tis_state["tis_certificate"]],
        s=14,
        color=TIS_COLOR,
        zorder=5,
    )
    gain_axis.annotate(
        rf"$\rho_B={rho_certificate:.3f}$",
        xy=(rho_certificate, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        rotation=90,
        fontsize=6.2,
        color=NEUTRAL_COLOR,
    )
    gain_axis.annotate(
        rf"$\rho_{{\rm stop}}={rho_stop:.3f}$",
        xy=(rho_stop, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(-2, -3),
        textcoords="offset points",
        ha="right",
        va="top",
        rotation=90,
        fontsize=6.5,
        color=NEUTRAL_COLOR,
    )

    handles, labels = risk_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
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
    if not all(
        row["tis_mse"] <= min(row["raw_mse"], row["ppo_mse"])
        for row in rows
    ):
        raise AssertionError("TIS does not have the lowest displayed MSE")
    if not all(
        row["tis_certificate"] >= row["ppo_certificate"]
        for row in rows
    ):
        raise AssertionError("TIS does not dominate the displayed PPO certificate")

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
            "tis_cap": TIS_CAP,
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
