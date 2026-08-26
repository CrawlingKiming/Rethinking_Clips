from pathlib import Path

root = Path(__file__).resolve().parents[1]

simulation_path = root / "simulation" / "optdigits_categorical_theory.py"
text = simulation_path.read_text(encoding="utf-8")

start = text.index("def quantile_bin_rows(")
end = text.index("def relative_error_bin_rows(")
text = text[:start] + r'''def quantile_bin_rows(state_rows: list[dict[str, float]], bins: int = 6) -> list[dict[str, float]]:
    rho = np.asarray([row["population_rho"] for row in state_rows], dtype=float)
    edges = np.unique(np.quantile(rho, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(float(np.min(rho)), float(np.max(rho)) + 1e-8, 3)
    assignments = np.digitize(rho, edges[1:-1], right=True)
    output: list[dict[str, float]] = []
    for bin_index in range(len(edges) - 1):
        indices = np.where(assignments == bin_index)[0]
        if not len(indices):
            continue
        selected = [state_rows[index] for index in indices]
        row: dict[str, float] = {
            "bin": float(bin_index),
            "rho_left": float(edges[bin_index]),
            "rho_right": float(edges[bin_index + 1]),
            "rho_median": float(np.median(rho[indices])),
            "states": float(len(indices)),
        }
        oracle = np.asarray([item["oracle_change"] for item in selected])
        row["oracle_mean_change"] = float(np.mean(oracle))
        row["oracle_change_se"] = standard_error(oracle)
        for name in ("raw", "truncation", "ppo"):
            values = np.asarray([item[f"{name}_mse"] for item in selected])
            row[f"{name}_mse"] = float(np.mean(values))
            row[f"{name}_mse_se"] = standard_error(values)
            changes = np.asarray([item[f"{name}_mean_change"] for item in selected])
            harms = np.asarray([item[f"{name}_harm_rate"] for item in selected])
            row[f"{name}_mean_change"] = float(np.mean(changes))
            row[f"{name}_change_se"] = standard_error(changes)
            row[f"{name}_harm_rate"] = float(np.mean(harms))
        output.append(row)
    return output


''' + text[end:]

start = text.index("def make_figure(")
end = text.index("def aggregate_final_values(")
text = text[:start] + r'''def make_figure(
    state_rows: list[dict[str, float]],
    bin_rows: list[dict[str, float]],
    error_rows: list[dict[str, float]],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    rho = np.asarray([row["population_rho"] for row in state_rows])
    raw_mse = np.asarray([row["raw_mse"] for row in state_rows])
    ax.scatter(rho, raw_mse, s=22, alpha=0.45)
    x = np.asarray([row["rho_median"] for row in bin_rows])
    y = np.asarray([row["raw_mse"] for row in bin_rows])
    yerr = np.asarray([row["raw_mse_se"] for row in bin_rows])
    order = np.argsort(x)
    ax.errorbar(x[order], y[order], yerr=yerr[order], marker="o", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("Unmodified gradient MSE")
    ax.set_title("ESS and estimator error")

    ax = axes[1]
    x = np.asarray([row["relative_error_median"] for row in error_rows])
    harm = np.asarray([row["harm_rate"] for row in error_rows])
    ax.plot(x, harm, marker="o")
    ax.axvline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(-0.01, max(0.2, float(np.max(harm)) + 0.04))
    ax.set_xlabel(r"Realized squared error / $\|g\|_2^2$")
    ax.set_ylabel("Probability of negative change")
    ax.set_title("Error relative to gradient signal")

    ax = axes[2]
    raw_change = np.asarray([row["raw_mean_change"] for row in bin_rows])
    raw_se = np.asarray([row["raw_change_se"] for row in bin_rows])
    oracle_change = np.asarray([row["oracle_mean_change"] for row in bin_rows])
    oracle_se = np.asarray([row["oracle_change_se"] for row in bin_rows])
    ax.errorbar(x=np.asarray([row["rho_median"] for row in bin_rows])[order],
                y=raw_change[order], yerr=raw_se[order], marker="o",
                capsize=3, label="Sampled gradient")
    ax.errorbar(x=np.asarray([row["rho_median"] for row in bin_rows])[order],
                y=oracle_change[order], yerr=oracle_se[order], marker="s",
                capsize=3, label="Population gradient")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Population normalized ESS")
    ax.set_ylabel("One-step population change")
    ax.set_title("Sampled and oracle updates")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


''' + text[end:]

start = text.index("def correlation_summary(")
end = text.index("def parse_args(")
text = text[:start] + r'''def correlation_summary(
    state_rows: list[dict[str, float]],
    draw_rows: list[dict[str, float]],
    final_rows: list[dict[str, float]],
) -> str:
    rho = np.asarray([row["population_rho"] for row in state_rows])
    raw_mse = np.asarray([row["raw_mse"] for row in state_rows])
    positive = rho > 0.0
    log_corr = float(
        np.corrcoef(np.log(rho[positive]), np.log(raw_mse[positive]))[0, 1]
    )
    raw_draws = [
        row
        for row in draw_rows
        if row["estimator"] == "raw" and np.isfinite(row["reward_change"])
    ]
    below = [row for row in raw_draws if row["relative_error"] < 1.0]
    above = [row for row in raw_draws if row["relative_error"] >= 1.0]
    below_harm = float(np.mean([row["reward_change"] < 0.0 for row in below]))
    above_harm = float(np.mean([row["reward_change"] < 0.0 for row in above]))
    oracle_negative = float(np.mean([row["oracle_change"] < 0.0 for row in state_rows]))

    lowest = min(state_rows, key=lambda row: row["population_rho"])
    highest = max(state_rows, key=lambda row: row["population_rho"])
    lines = [
        f"states={len(state_rows)}",
        f"population_rho_min={np.min(rho):.10f}",
        f"population_rho_median={np.median(rho):.6f}",
        f"population_rho_max={np.max(rho):.6f}",
        f"corr_log_rho_log_raw_mse={log_corr:.6f}",
        f"lowest_rho_raw_mse={lowest['raw_mse']:.8f}",
        f"highest_rho_raw_mse={highest['raw_mse']:.8f}",
        f"low_to_high_mse_ratio={lowest['raw_mse']/highest['raw_mse']:.6f}",
        f"relative_error_below_one_count={len(below)}",
        f"relative_error_below_one_harm_rate={below_harm:.6f}",
        f"relative_error_above_one_count={len(above)}",
        f"relative_error_above_one_harm_rate={above_harm:.6f}",
        f"oracle_negative_rate={oracle_negative:.6f}",
    ]
    for row in final_rows:
        lines.append(
            f"final_value_{row['trajectory']}_mean={row['mean_final_value']:.8f}"
        )
    return "\n".join(lines) + "\n"


''' + text[end:]

simulation_path.write_text(text, encoding="utf-8")

main_path = root / "main.tex"
main = main_path.read_text(encoding="utf-8")
section_start = main.index(r"\section{Policy optimization on OptDigits}")
section_end = main.index(r"\section{Language-model evidence for delayed failure and recovery}")
new_section = r'''\section{Policy optimization on OptDigits}
\label{sec:simulation}

The theory gives two implications that can be tested without introducing an ESS-based decision rule. First, lower effective support should increase the error of the importance-weighted gradient estimate. Second, the sampled update should become vulnerable when its realized gradient error is comparable to or larger than the population gradient signal. The experiment below tests these two links directly. No ESS threshold is used to choose an update.

\paragraph{Categorical contextual bandit.}
We convert the Optdigits classification dataset into a one-step contextual bandit \citep{alpaydin1998optdigits}. Each handwritten-digit image is a context $x$, the action is a digit $a\in\{0,\ldots,9\}$, and the reward is $R(x,a)=\mathbf 1\{a=y(x)\}$. The policy is a linear softmax classifier. The complete dataset is treated as a finite context population, so its population value and gradient can be evaluated exactly.

At each policy iteration, we sample 600 images without replacement and draw one action per image from a frozen rollout policy. We use the rollout policy's exact success probability as a detached context-dependent baseline. The 600 observations are shuffled into 12 minibatches, and the policy is updated sequentially for one epoch. The rollout is then discarded and a new batch is collected from the updated policy. We run static unmodified and static PPO trajectories to generate policy states with different levels of mismatch. Neither trajectory switches its update rule according to ESS.

\paragraph{Oracle diagnostics.}
To measure estimator reliability independently of a particular realized minibatch, we select 30 pre-update policy states spanning the observed population ESS range. At each state, we draw 80 fresh minibatches of 128 independent context-action pairs from the frozen rollout policy. The exact finite-population gradient gives the squared error of every sampled gradient. For the first 12 redraws at each state, we also evaluate the exact population reward change produced by a common diagnostic step. These redraws are used only for evaluation and never affect training.

\paragraph{Results.}
Figure~\ref{fig:optdigits-theory} verifies both links in the theory. In the lowest-ESS bin, whose median population ESS is $0.007$, the mean squared error of the unmodified gradient is $0.198$. In the on-policy bin, where the median ESS is $1.000$, the corresponding error is $0.0065$, a reduction by more than a factor of thirty. Thus the same nominal minibatch size can provide very different gradient accuracy as effective support changes.

The second panel compares realized squared gradient error with the population gradient signal. Across 209 redraws for which $\|\widehat g-g\|_2^2<\|g\|_2^2$, none produces a negative population change. Once the realized error exceeds the gradient signal, 21 of 151 updates, or $13.9\%$, are harmful. The population-gradient step improves the objective at all 30 frozen states. The harmful sampled updates therefore arise from estimation error rather than from the absence of a locally improving direction.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/optdigits_categorical_theory.pdf}
  \caption{Theory validation in a ten-action Optdigits contextual bandit. (a) The mean-squared error of the unmodified gradient increases as population normalized ESS decreases. Points denote frozen policy states, and the connected values are ESS-bin averages. (b) Harmful updates appear only after the realized squared gradient error reaches the scale of the population gradient signal. The dotted line marks equality of these two quantities. (c) The exact population-gradient step remains improving across the ESS range, while sampled-gradient steps become less reliable as effective support deteriorates. No ESS-conditioned update rule is used in this experiment.}
  \label{fig:optdigits-theory}
\end{figure}

The categorical experiment isolates the two statistical links with exact oracle quantities. It is not intended to validate a fixed ESS threshold. The next section examines whether the same loss of effective support accompanies delayed failure in language-model training, where the population gradient is unavailable.

'''
main = main[:section_start] + new_section + main[section_end:]
main_path.write_text(main, encoding="utf-8")

readme = r'''# Optdigits policy-optimization experiments

## Main theory experiment

`optdigits_categorical_theory.py` is the experiment used in the main paper. It is a one-step contextual bandit with the ten digit labels as the action space.

- The complete Optdigits dataset is the finite context population.
- At each policy iteration, 600 images are sampled without replacement.
- One action in `{0, ..., 9}` is drawn for each image from a frozen rollout policy.
- The reward is one when the sampled action equals the image label and zero otherwise.
- The rollout is shuffled into 12 minibatches and used for one optimization epoch.
- A new rollout is collected after the epoch.
- No ESS threshold or ESS-gated update is used.

The script runs static unmodified and static PPO trajectories only to produce policy states with different levels of mismatch. It then freezes 30 states across the observed ESS range and uses independent redraws to evaluate the theoretical links:

1. population normalized ESS versus gradient mean-squared error;
2. realized gradient error relative to the population gradient signal versus one-step population improvement.

The action distribution is a linear softmax policy. Because the context population is finite and the action space contains only ten classes, the population value, population gradient, and population ESS are all computed exactly.

Run from the repository root:

```bash
python -m pip install numpy matplotlib
python simulation/optdigits_categorical_theory.py --replications 12
```

Main outputs:

- `simulation/results/optdigits_categorical_frozen_states.csv`
- `simulation/results/optdigits_categorical_redraws.csv`
- `simulation/results/optdigits_categorical_ess_bins.csv`
- `simulation/results/optdigits_categorical_error_bins.csv`
- `simulation/results/optdigits_categorical_summary.txt`
- `figures/optdigits_categorical_theory.pdf`
- `figures/optdigits_categorical_theory.png`

## Legacy structured-action experiment

`ess_policy_optimization.py` contains the earlier 16-bit structured-action and ESS-gating study. It is retained for reference, but it is not the main Optdigits theory validation and is not used to support the fixed-threshold claim in the main text.
'''
(root / "simulation" / "README.md").write_text(readme, encoding="utf-8")
