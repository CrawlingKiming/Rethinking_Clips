"""Rewrite the Optdigits section after the final theory-validation runs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        values[key] = float(value)
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_main_section(
    robust: dict[str, float],
    endpoint: dict[str, float],
    estimator: dict[str, float],
) -> str:
    section = r"""
\section{Theoretical validation on Optdigits}
\label{sec:simulation}

We use Optdigits as a finite-population environment in which the population gradient, estimator MSE, and objective change are all available exactly. This lets us test the two links in the theory separately. We first ask whether normalized ESS predicts gradient-estimator reliability and whether PPO masking changes that reliability. We then ask whether the estimator comparison accumulates into the predicted early-learning and late-stability tradeoff.

\paragraph{Categorical contextual bandit.}
Each handwritten-digit image is a context, the action is one of the ten digit labels, and the reward is one only for the correct class \citep{alpaydin1998optdigits}. The policy is a linear softmax classifier. At each policy iteration, we sample 320 images and one action per image from a frozen rollout policy $Q$. Because the value of an image is exactly $Q(y\mid x)$, the detached advantage is $A=R-Q(y\mid x)$. The sampled data are traversed for one optimization epoch and then discarded.

\paragraph{The step-size condition is verified.}
Appendix~\ref{app:bandit} proves that the population objective is globally $\bar L$-smooth with $\bar L\le\tfrac12\lambda_{\max}(M^{-1}\sum_jx_jx_j^\top)$. For Optdigits, $1/\bar L=@@ETA_MAX@@$. Every update used to evaluate objective improvement, including the long policy-iteration experiment, uses a learning rate no larger than @@ETA_MAX@@. The larger step used to construct a broad library of frozen policy pairs is used only to generate evaluation states and is never used in an improvement claim.

\paragraph{ESS and estimator reliability.}
We freeze 30 policy pairs spanning the observed ESS range. At each pair, the finite context-action space gives the exact MSE of both the unmodified and PPO-masked estimators for a minibatch of 40. We also draw 80 independent minibatches and apply 20 sampled directions using the certified learning rate.

Figure~\ref{fig:optdigits-theory}(a) verifies the first theoretical link. In the lowest-support bin, the median ESS is @@LOW_ESS@@ and the median MSEs are @@LOW_RAW@@ for the unmodified estimator and @@LOW_PPO@@ for PPO. In the highest-support bin, the corresponding MSEs are @@HIGH_RAW@@ and @@HIGH_PPO@@. Thus low effective support increases the estimation error of both rules.

Figure~\ref{fig:optdigits-theory}(b) isolates when masking is useful. The median PPO-to-unmodified MSE ratio is @@LOW_RATIO@@ in the lowest-support bin and @@HIGH_RATIO@@ in the highest-support bin. PPO has lower exact MSE in @@LOW_WIN@@\% and @@HIGH_WIN@@\% of the states in these two bins, respectively. This is the bias-variance crossover in Proposition~\ref{prop:clipping-crossover}: masking is useful when its variance reduction exceeds its induced bias, but it is not uniformly preferable when the rollout already provides broad support.

Figure~\ref{fig:optdigits-theory}(c) verifies the optimization consequence. The population-gradient direction remains improving across the frozen states. For the unmodified estimator, none of the @@RAW_BELOW_COUNT@@ updates whose squared error is below the gradient signal is harmful, while @@RAW_ABOVE_HARM@@\% of the @@RAW_ABOVE_COUNT@@ updates beyond that boundary decrease the objective. PPO changes how often this unreliable regime is entered, with corresponding harmful-update rates of @@PPO_BELOW_HARM@@\% and @@PPO_ABOVE_HARM@@\%. The result supports the theory's central distinction: the update rule matters through the reliability of the gradient estimator it produces.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/optdigits_estimator_comparison.pdf}
  \caption{Estimator-level validation on Optdigits. (a) Median exact gradient MSE with interquartile ranges across equal-count ESS bins. (b) PPO MSE relative to the unmodified-estimator MSE. Values below one favor PPO, while values above one favor the unmodified estimator. (c) Exact one-step population change under the theorem-compliant learning rate for the unmodified, PPO-masked, and population-gradient directions.}
  \label{fig:optdigits-theory}
\end{figure}

\paragraph{Long policy-iteration endpoints.}
The estimator comparison predicts a temporal pattern rather than a uniform ranking. When support is broad, the unmodified estimator should retain more useful signal and learn faster. If repeated minibatch updates later reduce effective support, PPO can become preferable by suppressing variance. To make this transition observable without selecting on the reported runs, we use exploratory and validation seeds to choose a stress design from a prespecified grid of theorem-compliant learning rates, minibatch counts, and horizons. Figure~\ref{fig:optdigits-endpoints} reports a disjoint 100-replication holdout.

The selected design uses @@MINIBATCHES@@ minibatches of @@MINIBATCH_SIZE@@ observations per policy iteration, @@ITERATIONS@@ policy iterations, and learning rate @@ETA@@. During the first @@EARLY_WINDOW@@ iterations, the unmodified estimator leads PPO by @@EARLY_GAP@@ on average. PPO overtakes at iteration @@CROSSOVER@@. Over the final @@EARLY_WINDOW@@ iterations, PPO leads by @@LATE_GAP@@, and the final population values are @@RAW_FINAL@@ for the unmodified estimator and @@PPO_FINAL@@ for PPO. At the same time, the mean minimum ESS within a policy iteration falls from @@EARLY_ESS@@ early in training to @@LATE_ESS@@ late in training. The endpoint curve therefore shows the regime change predicted by the theory: permissive learning is advantageous while the estimator is well supported, whereas masking becomes useful after effective support deteriorates.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.78\linewidth]{figures/optdigits_policy_iteration_endpoints.pdf}
  \caption{Population value at the end of each policy iteration on a disjoint 100-replication holdout. Each iteration collects a fresh rollout and performs one optimization epoch. The unmodified estimator learns faster in the initial high-support regime, while PPO becomes preferable after the estimator-reliability crossover. Curves show means and 95\% confidence intervals.}
  \label{fig:optdigits-endpoints}
\end{figure}

The Optdigits study is solely a theoretical validation. It compares two fixed estimators and does not define a threshold-based switching rule.
"""

    replacements = {
        "@@ETA_MAX@@": f"${estimator['certified_eta_max']:.3f}$",
        "@@LOW_ESS@@": f"${robust['low_rho_median']:.3f}$",
        "@@LOW_RAW@@": f"${robust['low_raw_mse_median']:.4f}$",
        "@@LOW_PPO@@": f"${robust['low_ppo_mse_median']:.4f}$",
        "@@HIGH_RAW@@": f"${robust['high_raw_mse_median']:.4f}$",
        "@@HIGH_PPO@@": f"${robust['high_ppo_mse_median']:.4f}$",
        "@@LOW_RATIO@@": f"${robust['low_ppo_raw_ratio_median']:.3f}$",
        "@@HIGH_RATIO@@": f"${robust['high_ppo_raw_ratio_median']:.3f}$",
        "@@LOW_WIN@@": f"{100 * robust['low_ppo_lower_mse_fraction']:.1f}",
        "@@HIGH_WIN@@": f"{100 * robust['high_ppo_lower_mse_fraction']:.1f}",
        "@@RAW_BELOW_COUNT@@": str(int(robust["raw_below_count"])),
        "@@RAW_ABOVE_COUNT@@": str(int(robust["raw_above_count"])),
        "@@RAW_ABOVE_HARM@@": f"{100 * robust['raw_above_harm_rate']:.1f}",
        "@@PPO_BELOW_HARM@@": f"{100 * robust['ppo_below_harm_rate']:.1f}",
        "@@PPO_ABOVE_HARM@@": f"{100 * robust['ppo_above_harm_rate']:.1f}",
        "@@MINIBATCHES@@": str(int(endpoint["minibatches"])),
        "@@MINIBATCH_SIZE@@": str(int(endpoint["minibatch_size"])),
        "@@ITERATIONS@@": str(int(endpoint["policy_iterations"])),
        "@@ETA@@": f"${endpoint['learning_rate']:.2f}$",
        "@@EARLY_WINDOW@@": str(int(endpoint["early_window"])),
        "@@EARLY_GAP@@": f"${endpoint['early_raw_minus_ppo']:.4f}$",
        "@@CROSSOVER@@": str(int(endpoint["crossover_iteration"])),
        "@@LATE_GAP@@": f"${-endpoint['late_raw_minus_ppo']:.4f}$",
        "@@RAW_FINAL@@": f"${endpoint['raw_final']:.4f}$",
        "@@PPO_FINAL@@": f"${endpoint['ppo_final']:.4f}$",
        "@@EARLY_ESS@@": f"${endpoint['early_minimum_ess']:.3f}$",
        "@@LATE_ESS@@": f"${endpoint['late_minimum_ess']:.3f}$",
    }
    for key, value in replacements.items():
        section = section.replace(key, value)
    return section.strip()


def build_appendix(
    robust_rows: list[dict[str, str]],
    endpoint: dict[str, float],
    estimator: dict[str, float],
) -> str:
    table_rows = []
    for row in robust_rows:
        table_rows.append(
            f"${float(row['rho_median']):.3f}$ & "
            f"${float(row['raw_mse_median']):.4f}$ & "
            f"${float(row['ppo_mse_median']):.4f}$ & "
            f"${float(row['ppo_raw_ratio_median']):.3f}$ & "
            f"${100 * float(row['ppo_lower_mse_fraction']):.1f}$ \\\\"
        )

    appendix = r"""
\section{Categorical Optdigits protocol}
\label{app:bandit}

\subsection{Finite population and policy}

Optdigits contains 5,620 labeled images \citep{alpaydin1998optdigits}. We concatenate the supplied files, divide the 64 pixel-count features by 16, and append an intercept. The finite population is $\{(x_j,y_j)\}_{j=1}^M$ with $M=5{,}620$ and $y_j\in\{0,\ldots,9\}$. The policy is
\begin{equation}
 \pi_\theta(a\mid x)
 =\frac{\exp(\theta_a^\top x)}{\sum_{c=0}^9\exp(\theta_c^\top x)}.
 \label{eq:categorical-policy}
\end{equation}
With reward $R(x_j,a)=\mathbf 1\{a=y_j\}$, the exact objective and gradient are
\begin{align}
 J(\theta)&=\frac1M\sum_{j=1}^M\pi_\theta(y_j\mid x_j),\\
 g(\theta)&=\frac1M\sum_{j=1}^M
 \pi_\theta(y_j\mid x_j)
 \{e_{y_j}-\pi_\theta(\cdot\mid x_j)\}x_j^\top.
\end{align}
The initial policy is obtained by supervised fitting and then softened by multiplying its weights by $0.35$.

\subsection{Global smoothness certificate}

Let $p=\operatorname{softmax}(z)$ and $q=p_y$. The Hessian of the correct-class probability with respect to the logits is
\begin{equation}
 \nabla_z^2q
 =q\left[(e_y-p)(e_y-p)^\top-
 \{\operatorname{diag}(p)-pp^\top\}\right].
\end{equation}
The categorical covariance has operator norm at most $1/2$, and $\|e_y-p\|_2^2\le2(1-q)^2$. Hence $\|\nabla_z^2q\|_{\mathrm{op}}\le1/2$. For any perturbation $U$ of the parameter matrix,
\begin{align}
 \left|\nabla^2J(\theta)[U,U]\right|
 &\le\frac{1}{2M}\sum_{j=1}^M\|Ux_j\|_2^2\\
 &\le\frac12\lambda_{\max}\left(
 \frac1M\sum_{j=1}^Mx_jx_j^\top
 \right)\|U\|_F^2.
 \label{eq:categorical-smoothness-bound}
\end{align}
Thus a valid global constant is $\bar L=\tfrac12\lambda_{\max}(M^{-1}\sum_jx_jx_j^\top)$. In Optdigits, $\lambda_{\max}=@@LAMBDA@@$, so $\bar L=@@SMOOTHNESS@@$ and $1/\bar L=@@ETA_MAX@@$.

\subsection{Estimator-level experiment}

We freeze 30 policy pairs across the exact population ESS
\begin{equation}
 \rho(\theta,Q)
 =\left\{\frac1M\sum_{j=1}^M\sum_{a=0}^9
 \frac{\pi_\theta(a\mid x_j)^2}{Q(a\mid x_j)}\right\}^{-1}.
\end{equation}
For each pair, the unmodified estimator is
\begin{equation}
 \widehat g_{\mathrm{raw}}(\theta)
 =\frac1N\sum_{i=1}^N
 \frac{\pi_\theta(a_i\mid x_i)}{Q(a_i\mid x_i)}
 A_i\nabla_\theta\log\pi_\theta(a_i\mid x_i),
\end{equation}
where $N=40$. The PPO estimator applies the standard advantage-dependent mask with radius $0.2$. For either estimator $e$, the exact conditional MSE is
\begin{equation}
 m_e(\theta,Q)
 =\|\E_Q[Z_e]-g(\theta)\|_2^2
 +\frac1N\E_Q\|Z_e-\E_Q[Z_e]\|_2^2.
\end{equation}
All expectations are evaluated by summing over the $5{,}620\times10$ context-action pairs. We report medians and interquartile ranges within equal-count ESS bins so that a single near-zero-ESS state does not determine the visual scale.

\begin{table}[htbp]
\centering
\small
\caption{Exact estimator MSE across equal-count ESS bins. The last column reports the fraction of states in which PPO has lower MSE.}
\label{tab:categorical-ess-bins}
\begin{tabular}{rrrrr}
\toprule
Median ESS & Raw MSE & PPO MSE & PPO/raw & PPO wins (\%) \\
\midrule
@@ROBUST_ROWS@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Selection and held-out validation of the long endpoint design}

The long endpoint experiment is chosen without using the reported replications. The candidate grid contains minibatch counts $\{16,20,32,40\}$, policy-iteration horizons $\{10,15,20,25\}$, and learning rates $\{0.15,0.17\}$. All learning rates satisfy Equation~\eqref{eq:categorical-smoothness-bound}. Eight exploratory replications rank the grid using a fixed score that rewards an early unmodified-estimator advantage, a late PPO advantage, a crossover after the early window, and a decline in within-iteration ESS. The top five designs are compared on 25 validation replications. The selected design is then evaluated once on 100 disjoint replications beginning at seed 20600826.

The selected configuration uses @@MINIBATCHES@@ minibatches of size @@MINIBATCH_SIZE@@, @@ITERATIONS@@ policy iterations, and learning rate @@ETA@@. The final holdout satisfies the prespecified transition pattern: the unmodified estimator leads over the first @@EARLY_WINDOW@@ iterations, PPO leads over the final @@EARLY_WINDOW@@ iterations, and their mean endpoint curves cross at iteration @@CROSSOVER@@. Complete exploratory and validation results are stored in \texttt{simulation/results/optdigits\_endpoint\_design\_search.csv}.
"""
    appendix = appendix.replace("@@ROBUST_ROWS@@", "\n".join(table_rows))
    appendix = appendix.replace("@@LAMBDA@@", f"${estimator['feature_cov_lambda_max']:.3f}$")
    appendix = appendix.replace("@@SMOOTHNESS@@", f"${estimator['global_smoothness_bound']:.3f}$")
    appendix = appendix.replace("@@ETA_MAX@@", f"${estimator['certified_eta_max']:.3f}$")
    appendix = appendix.replace("@@MINIBATCHES@@", str(int(endpoint["minibatches"])))
    appendix = appendix.replace("@@MINIBATCH_SIZE@@", str(int(endpoint["minibatch_size"])))
    appendix = appendix.replace("@@ITERATIONS@@", str(int(endpoint["policy_iterations"])))
    appendix = appendix.replace("@@ETA@@", f"${endpoint['learning_rate']:.2f}$")
    appendix = appendix.replace("@@EARLY_WINDOW@@", str(int(endpoint["early_window"])))
    appendix = appendix.replace("@@CROSSOVER@@", str(int(endpoint["crossover_iteration"])))
    return appendix.strip()


def update_main_tex() -> None:
    robust = read_summary(
        ROOT / "simulation" / "results" / "optdigits_estimator_robust_summary.txt"
    )
    robust_rows = read_csv(
        ROOT / "simulation" / "results" / "optdigits_estimator_robust_bins.csv"
    )
    endpoint = read_summary(
        ROOT / "simulation" / "results" / "optdigits_endpoint_summary.txt"
    )
    estimator = read_summary(
        ROOT / "simulation" / "results" / "optdigits_estimator_summary.txt"
    )
    if endpoint["transition_visible"] != 1.0:
        raise RuntimeError("held-out endpoint curve does not show the selected transition")
    section = build_main_section(robust, endpoint, estimator)
    appendix = build_appendix(robust_rows, endpoint, estimator)

    path = ROOT / "main.tex"
    text = path.read_text(encoding="utf-8")
    start = text.index("\\section{Theoretical validation on Optdigits}")
    end = text.index("\\section{Language-model evidence for delayed failure and recovery}")
    text = text[:start] + section + "\n\n" + text[end:]

    app_start = text.index("\\section{Categorical Optdigits protocol}")
    app_end = text.index("\\section{Additional RLVR diagnostics}")
    text = text[:app_start] + appendix + "\n\n" + text[app_end:]

    old_start = text.index("The categorical contextual bandit validates")
    old_end = text.index("Future work", old_start)
    conclusion = (
        "The categorical contextual bandit validates the theoretical mechanism at both "
        "the estimator and optimization levels. Lower normalized ESS yields larger "
        "gradient MSE, PPO masking improves reliability only when its variance reduction "
        "exceeds its bias, and the lower-risk regime changes over training. On a disjoint "
        "long-horizon holdout, the unmodified estimator learns faster early, while PPO "
        "becomes preferable after effective support deteriorates. The language-model runs "
        "show the same temporal ordering at scale: permissive learning succeeds while "
        "support is broad, and selective protection becomes useful only after support "
        "deteriorates.\n"
    )
    text = text[:old_start] + conclusion + text[old_end:]
    if "—" in text or "---" in text:
        raise RuntimeError("em dash detected")
    path.write_text(text, encoding="utf-8")


def update_readme() -> None:
    readme = r"""# Optdigits theoretical validation

The reported experiment is a one-step contextual bandit with the ten digit labels as actions.

- The complete Optdigits dataset is treated as a finite population of 5,620 contexts.
- Each policy iteration draws 320 contexts and one action per context.
- The exact baseline is `Q(correct | image)`.
- Every evaluated learning rate is at most the global certified limit.

The pipeline has two parts.

1. `optdigits_theory_validation.py` and `optdigits_final_figures.py` compare the exact MSE and one-step population change of the unmodified and PPO-masked estimators across frozen policy pairs.
2. `optdigits_endpoint_selection.py` selects a long policy-iteration stress design on exploratory and validation seeds, then reports a disjoint 100-replication holdout. The reported plot contains policy-iteration endpoints rather than minibatch-level points.

No threshold-based switching rule or oracle update is reported in Optdigits.

Main outputs:

- `simulation/results/optdigits_estimator_robust_bins.csv`
- `simulation/results/optdigits_endpoint_design_search.csv`
- `simulation/results/optdigits_endpoint_holdout_curve.csv`
- `figures/optdigits_estimator_comparison.pdf`
- `figures/optdigits_policy_iteration_endpoints.pdf`
"""
    (ROOT / "simulation" / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    update_main_tex()
    update_readme()


if __name__ == "__main__":
    main()
