"""Patch the paper after the final Optdigits estimator comparison."""

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


def pm(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}\\pm{se:.{digits}f}"


def build_main_section(summary: dict[str, float]) -> str:
    low_ess = summary["low_ess_median"]
    low_raw = summary["low_ess_raw_mse"]
    low_ppo = summary["low_ess_ppo_mse"]
    low_ppo_fraction = 100 * summary["low_ess_ppo_lower_mse_fraction"]
    high_ess = summary["high_ess_median"]
    high_raw = summary["high_ess_raw_mse"]
    high_ppo = summary["high_ess_ppo_mse"]
    high_ppo_fraction = 100 * summary["high_ess_ppo_lower_mse_fraction"]
    raw_below = int(summary["raw_relative_error_below_one_count"])
    raw_below_harm = 100 * summary["raw_relative_error_below_one_harm_rate"]
    raw_above = int(summary["raw_relative_error_above_one_count"])
    raw_above_harm = 100 * summary["raw_relative_error_above_one_harm_rate"]
    ppo_below = int(summary["ppo_relative_error_below_one_count"])
    ppo_below_harm = 100 * summary["ppo_relative_error_below_one_harm_rate"]
    ppo_above = int(summary["ppo_relative_error_above_one_count"])
    ppo_above_harm = 100 * summary["ppo_relative_error_above_one_harm_rate"]
    lambda_max = summary["feature_cov_lambda_max"]
    smoothness = summary["global_smoothness_bound"]
    eta_max = summary["certified_eta_max"]
    eta = summary["used_learning_rate"]
    raw_final = summary["final_raw"]
    raw_final_se = summary["final_raw_se"]
    ppo_final = summary["final_ppo"]
    ppo_final_se = summary["final_ppo_se"]
    raw_minus_ppo = summary["raw_minus_ppo"]
    raw_minus_ppo_se = summary["raw_minus_ppo_se"]

    section = r"""
\section{Theoretical validation on Optdigits}
\label{sec:simulation}

Optdigits is used only as a finite-population test of the theoretical claims. The section compares the unmodified and PPO-masked gradient estimators as two estimators of the same population gradient.

\paragraph{Categorical contextual bandit.}
We convert Optdigits into a one-step contextual bandit \citep{alpaydin1998optdigits}. Each handwritten-digit image is a context, the action is a digit in $\{0,\ldots,9\}$, and the reward is one only for the correct class. The policy is a linear softmax classifier. At each policy iteration, we draw 320 contexts from the finite population and sample one action per context from a frozen rollout policy. Since the value of an image under the rollout policy is exactly its probability of the correct class, the advantage is $A=R-Q(y\mid x)$. The observations are divided into eight minibatches of 40 and used for one optimization epoch.

\paragraph{Step-size condition.}
The smoothness condition is checked rather than assumed numerically. Appendix~\ref{app:bandit} shows that the Optdigits objective is globally $\bar L$-smooth with $\bar L\le\tfrac12\lambda_{\max}(M^{-1}\sum_jx_jx_j^\top)$. In this population, $\lambda_{\max}=@@LAMBDA@@$, giving $\bar L=@@SMOOTHNESS@@$ and $1/\bar L=@@ETA_MAX@@$. Every one-step evaluation in Figure~\ref{fig:optdigits-theory} and every learning update in Figure~\ref{fig:optdigits-learning} uses $\eta=@@ETA@@$, so the condition $\eta\le1/\bar L$ holds. A separate stress trajectory is used only to generate a broad collection of frozen policy pairs and is not used to evaluate the smoothness guarantee.

\paragraph{Comparing the two estimators.}
We freeze 30 policy pairs across the observed population-ESS range. At each pair, the finite context-action space gives the exact MSE of the unmodified and PPO estimators for a minibatch of 40. We also draw 80 independent minibatches and apply the first 20 sampled directions with the certified step size.

Figure~\ref{fig:optdigits-theory} shows how PPO changes each link in the theory. In the lowest-support bin, the median ESS is @@LOW_ESS@@, the unmodified MSE is @@LOW_RAW@@, and the PPO MSE is @@LOW_PPO@@. PPO has lower exact MSE at @@LOW_PPO_FRAC@@\% of the states in this bin. In the highest-support bin, the corresponding MSEs are @@HIGH_RAW@@ and @@HIGH_PPO@@, and PPO has lower MSE at only @@HIGH_PPO_FRAC@@\% of states. Thus masking reduces estimator risk when support is poor, but adds unnecessary distortion in much of the high-support regime.

For the unmodified estimator, none of the @@RAW_BELOW@@ updates with realized squared error below the gradient signal is harmful, while @@RAW_ABOVE_HARM@@\% of the @@RAW_ABOVE@@ updates above that boundary decrease the population objective. The corresponding rates for PPO are @@PPO_BELOW_HARM@@\% among @@PPO_BELOW@@ updates below the boundary and @@PPO_ABOVE_HARM@@\% among @@PPO_ABOVE@@ updates above it. These comparisons show that the same error-to-signal boundary governs both estimators, while PPO changes how frequently each estimator enters the unreliable regime.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/optdigits_estimator_comparison.pdf}
  \caption{Theoretical validation with the unmodified and PPO-masked estimators. (a) Exact gradient MSE across population normalized ESS. PPO lowers MSE in the low-support region, while the unmodified estimator is more reliable in much of the high-support region. (b) Harmful-update rate as realized squared gradient error crosses the population-gradient signal. (c) One-step population change under the certified step size. The population-gradient direction remains improving, while the two sampled estimators exhibit different bias and variance profiles.}
  \label{fig:optdigits-theory}
\end{figure}

\paragraph{Learning across all minibatch updates.}
Figure~\ref{fig:optdigits-learning} restores the full optimization trajectory rather than showing only policy-iteration endpoints. One minibatch update uses 40 images. Five policy iterations with eight minibatches each therefore produce 40 updates, and a fresh rollout is collected after updates 8, 16, 24, and 32. Across 100 paired replications, the final population value is $@@RAW_FINAL@@$ for the unmodified estimator and $@@PPO_FINAL@@$ for PPO, with paired difference $@@RAW_MINUS_PPO@@$. Under the theorem-certified step size, the unmodified estimator retains a small advantage because the rollouts remain sufficiently well supported for PPO masking to remove more signal than variance over much of the trajectory.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/optdigits_certified_learning.pdf}
  \caption{Population value across all 40 minibatch updates. Curves show means and 95\% confidence intervals over 100 paired replications. Dotted vertical lines mark collection of a fresh rollout. Every update uses $\eta=@@ETA@@$, which satisfies the global smoothness condition derived in Appendix~\ref{app:bandit}.}
  \label{fig:optdigits-learning}
\end{figure}

Together, the two figures validate the theoretical interpretation. ESS controls the reliability of the unmodified estimator, PPO changes the bias-variance tradeoff rather than uniformly improving it, and aggressive masking can be unnecessarily restrictive while effective support remains broad. No threshold-based update rule is considered in Optdigits.
"""

    replacements = {
        "@@LAMBDA@@": f"{lambda_max:.3f}",
        "@@SMOOTHNESS@@": f"{smoothness:.3f}",
        "@@ETA_MAX@@": f"{eta_max:.3f}",
        "@@ETA@@": f"{eta:.2f}",
        "@@LOW_ESS@@": f"${low_ess:.3f}$",
        "@@LOW_RAW@@": f"${low_raw:.4f}$",
        "@@LOW_PPO@@": f"${low_ppo:.4f}$",
        "@@LOW_PPO_FRAC@@": f"{low_ppo_fraction:.1f}",
        "@@HIGH_RAW@@": f"${high_raw:.4f}$",
        "@@HIGH_PPO@@": f"${high_ppo:.4f}$",
        "@@HIGH_PPO_FRAC@@": f"{high_ppo_fraction:.1f}",
        "@@RAW_BELOW@@": str(raw_below),
        "@@RAW_ABOVE@@": str(raw_above),
        "@@RAW_ABOVE_HARM@@": f"{raw_above_harm:.1f}",
        "@@PPO_BELOW@@": str(ppo_below),
        "@@PPO_ABOVE@@": str(ppo_above),
        "@@PPO_BELOW_HARM@@": f"{ppo_below_harm:.1f}",
        "@@PPO_ABOVE_HARM@@": f"{ppo_above_harm:.1f}",
        "@@RAW_FINAL@@": pm(raw_final, raw_final_se),
        "@@PPO_FINAL@@": pm(ppo_final, ppo_final_se),
        "@@RAW_MINUS_PPO@@": pm(raw_minus_ppo, raw_minus_ppo_se),
    }
    for key, value in replacements.items():
        section = section.replace(key, value)
    return section.strip()


def build_appendix(
    summary: dict[str, float],
    ess_bins: list[dict[str, str]],
) -> str:
    rows = []
    for row in ess_bins:
        rows.append(
            f"${float(row['rho_median']):.3f}$ & "
            f"${float(row['raw_exact_mse']):.4f}$ & "
            f"${float(row['ppo_exact_mse']):.4f}$ & "
            f"${100 * float(row['raw_harm_rate']):.1f}$ & "
            f"${100 * float(row['ppo_harm_rate']):.1f}$ \\\\"
        )
    eta = summary["used_learning_rate"]
    lambda_max = summary["feature_cov_lambda_max"]
    smoothness = summary["global_smoothness_bound"]
    eta_max = summary["certified_eta_max"]

    appendix = r"""
\section{Categorical Optdigits protocol}
\label{app:bandit}

This appendix gives the full construction used in Section~\ref{sec:simulation}. The experiment is a one-step contextual bandit with ten categorical actions. It is designed only to validate the ESS-dependent reliability theorem and to compare the unmodified and PPO-masked estimators.

\subsection{Finite population and policy}

Optdigits contains 5,620 labeled handwritten-digit images \citep{alpaydin1998optdigits}. We concatenate the supplied training and test files, divide each of the 64 pixel-count features by 16, and append an intercept. The resulting finite population is $\{(x_j,y_j)\}_{j=1}^M$ with $M=5{,}620$, $x_j\in\mathbb R^{65}$, and $y_j\in\{0,\ldots,9\}$. The original split is not used because the target is finite-population policy optimization rather than classification generalization.

The policy is a linear softmax model,
\begin{equation}
 \pi_\theta(a\mid x)
 =\frac{\exp(\theta_a^\top x)}{\sum_{c=0}^9\exp(\theta_c^\top x)},
 \qquad a\in\{0,\ldots,9\}.
 \label{eq:categorical-policy}
\end{equation}
The reward is $R(x_j,a)=\mathbf 1\{a=y_j\}$, so the exact population objective and gradient are
\begin{align}
 J(\theta)&=\frac1M\sum_{j=1}^M\pi_\theta(y_j\mid x_j),
 \label{eq:categorical-value}\\
 g(\theta)&=\frac1M\sum_{j=1}^M
 \pi_\theta(y_j\mid x_j)
 \{e_{y_j}-\pi_\theta(\cdot\mid x_j)\}x_j^\top.
 \label{eq:categorical-gradient}
\end{align}
Both quantities are evaluated by summing over all images. The policy is initialized by 400 full-population cross-entropy steps with step size $0.5$, after which the fitted weights are multiplied by $0.35$.

\subsection{A global smoothness certificate}

Let $p=\operatorname{softmax}(z)$ and $q=p_y$. The Hessian of one correct-class probability with respect to the logits is
\begin{equation}
 \nabla_z^2q
 =q\left[
 (e_y-p)(e_y-p)^\top-\{\operatorname{diag}(p)-pp^\top\}
 \right].
 \label{eq:categorical-logit-hessian}
\end{equation}
The categorical covariance satisfies $\|\operatorname{diag}(p)-pp^\top\|_{\mathrm{op}}\le1/2$, and $\|e_y-p\|_2^2\le2(1-q)^2$. Hence
\begin{equation}
 \|\nabla_z^2q\|_{\mathrm{op}}
 \le q\{2(1-q)^2+1/2\}
 \le1/2.
 \label{eq:categorical-logit-bound}
\end{equation}
For a perturbation $U$ of the softmax parameter matrix, Equation~\eqref{eq:categorical-logit-bound} gives
\begin{align}
 \left|\nabla^2J(\theta)[U,U]\right|
 &\le\frac{1}{2M}\sum_{j=1}^M\|Ux_j\|_2^2
 \nonumber\\
 &\le\frac12\lambda_{\max}\left(
 \frac1M\sum_{j=1}^Mx_jx_j^\top
 \right)\|U\|_F^2.
 \label{eq:categorical-smoothness-bound}
\end{align}
Thus a valid global smoothness constant is $\bar L=\tfrac12\lambda_{\max}(M^{-1}\sum_jx_jx_j^\top)$. For Optdigits, $\lambda_{\max}=@@LAMBDA@@$, so $\bar L=@@SMOOTHNESS@@$ and $1/\bar L=@@ETA_MAX@@$. All evaluated updates use $\eta=@@ETA@@\le1/\bar L$.

\subsection{Sampling and gradient estimators}

At each policy iteration, we freeze the current policy as $Q$, draw 320 contexts independently and uniformly from the finite population, and sample one action from $Q(\cdot\mid x)$ for each context. Since the reward is exact class match, the state value under $Q$ is $Q(y\mid x)$. We use the detached exact baseline and set $A=R-Q(y\mid x)$. The 320 observations are randomly partitioned into eight minibatches of size $N=40$ and traversed once. The rollout is then discarded.

The unmodified estimator is
\begin{equation}
 \widehat g_{\mathrm{raw}}(\theta)
 =\frac1N\sum_{i=1}^N
 \frac{\pi_\theta(a_i\mid x_i)}{Q(a_i\mid x_i)}
 A_i\nabla_\theta\log\pi_\theta(a_i\mid x_i).
 \label{eq:categorical-raw-gradient}
\end{equation}
The PPO estimator multiplies each sampled contribution by the standard advantage-dependent mask with radius $0.2$.

For $e\in\{\mathrm{raw},\mathrm{PPO}\}$, let $Z_e$ denote one gradient contribution. Since the context population and action space are finite, the exact conditional MSE is
\begin{equation}
 m_e(\theta,Q)
 =\left\|\E_Q[Z_e]-g(\theta)\right\|_2^2
 +\frac1N\E_Q\left\|Z_e-\E_Q[Z_e]\right\|_2^2.
 \label{eq:categorical-exact-risk}
\end{equation}
We calculate Equation~\eqref{eq:categorical-exact-risk} by summing over all $5{,}620\times10$ context-action pairs.

\subsection{Frozen-state estimator comparison}

To obtain policy pairs spanning a wide ESS range, we generate a separate state library from six policy iterations of the static unmodified and PPO updates under a stress step size. This trajectory is used only to define fixed pairs $(P_\theta,Q)$. No population-change result is computed using the stress step. We order the resulting pre-update states by exact population ESS,
\begin{equation}
 \rho(\theta,Q)
 =\left\{
 \frac1M\sum_{j=1}^M\sum_{a=0}^9
 \frac{\pi_\theta(a\mid x_j)^2}{Q(a\mid x_j)}
 \right\}^{-1},
 \label{eq:categorical-population-ess}
\end{equation}
and retain 30 approximately equally spaced states. At each state, the exact MSE uses minibatch size 40, and 80 independent minibatches estimate realized gradient error. The first 20 redraws receive the certified step $\eta=@@ETA@@$, after which the exact objective is recomputed.

\begin{table}[htbp]
\centering
\small
\caption{Optdigits estimator comparison across equal-count population-ESS bins. Exact MSE includes both squared bias and variance divided by 40. Harmful-update rates use the certified step size.}
\label{tab:categorical-ess-bins}
\begin{tabular}{rrrrr}
\toprule
Median ESS & Raw MSE & PPO MSE & Raw harmful (\%) & PPO harmful (\%) \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Full certified learning curve}

The learning comparison uses five policy iterations and 100 paired replications. Each iteration consists of eight minibatch updates followed by collection of a fresh rollout, giving 40 updates in total. The unmodified and PPO trajectories share context draws, action-sampling uniforms, and minibatch order within every replication. Both use $\eta=@@ETA@@$, which satisfies Equation~\eqref{eq:categorical-smoothness-bound}. No threshold rule or oracle update is used.
"""
    appendix = appendix.replace("@@ROWS@@", "\n".join(rows))
    appendix = appendix.replace("@@LAMBDA@@", f"{lambda_max:.3f}")
    appendix = appendix.replace("@@SMOOTHNESS@@", f"{smoothness:.3f}")
    appendix = appendix.replace("@@ETA_MAX@@", f"{eta_max:.3f}")
    appendix = appendix.replace("@@ETA@@", f"{eta:.2f}")
    return appendix.strip()


def update_main_tex() -> None:
    summary = read_summary(
        ROOT / "simulation" / "results" / "optdigits_estimator_summary.txt"
    )
    ess_bins = read_csv(
        ROOT / "simulation" / "results" / "optdigits_estimator_ess_bins.csv"
    )
    section = build_main_section(summary)
    appendix = build_appendix(summary, ess_bins)

    path = ROOT / "main.tex"
    text = path.read_text(encoding="utf-8")
    section_start = text.index("\\section{Theoretical validation on Optdigits}")
    section_end = text.index(
        "\\section{Language-model evidence for delayed failure and recovery}"
    )
    text = text[:section_start] + section + "\n\n" + text[section_end:]

    appendix_start = text.index("\\section{Categorical Optdigits protocol}")
    appendix_end = text.index("\\section{Additional RLVR diagnostics}")
    text = text[:appendix_start] + appendix + "\n\n" + text[appendix_end:]

    conclusion_start = text.index("The categorical contextual bandit validates")
    conclusion_end = text.index("Future work", conclusion_start)
    conclusion = (
        "The categorical contextual bandit validates the theoretical links for both the "
        "unmodified and PPO-masked estimators. Lower normalized ESS coincides with larger "
        "gradient MSE, and harmful updates appear after the realized error reaches the "
        "population-gradient signal. PPO reduces MSE in the low-support regime but is more "
        "distorting in much of the high-support regime. A separate 40-update learning curve "
        "uses a globally certified step size and shows the corresponding loss of learning "
        "speed from static masking while support remains broad. No practical threshold-based "
        "algorithm is defined in Optdigits. The language-model runs show a related temporal "
        "ordering at scale: permissive learning succeeds while effective support is broad, "
        "and selective protection becomes useful only after support deteriorates.\n"
    )
    text = text[:conclusion_start] + conclusion + text[conclusion_end:]
    path.write_text(text, encoding="utf-8")


def update_readme() -> None:
    readme = r"""# Optdigits theoretical validation

`optdigits_theory_validation.py` is the Optdigits experiment used in the paper. The task is a one-step contextual bandit with the ten digit labels as actions.

- The complete Optdigits dataset is treated as a finite population of 5,620 contexts.
- Each policy iteration draws 320 contexts.
- One digit action is sampled per context from a frozen rollout policy.
- The exact baseline is the rollout probability of the correct class, so the advantage is `reward - Q(correct | image)`.
- Each rollout is divided into 8 minibatches of 40 and used for one optimization epoch.
- A fresh rollout is collected at the next policy iteration.
- All evaluated policy updates use learning rate `0.17`, below the global certified limit computed from the feature covariance.

The script performs two theory-validation studies.

1. It freezes policy pairs across a wide population-ESS range and compares the exact MSE, harmful-update rate, and one-step population change of the unmodified and PPO-masked estimators.
2. It reports the full 40-update learning curves for the two estimators under the certified step size.

No threshold-based update rule or oracle update is evaluated in the reported Optdigits study.

Run from the repository root:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/optdigits_theory_validation.py --replications 100 --diagnostic-replications 40
```

Main outputs:

- `simulation/results/optdigits_estimator_states.csv`
- `simulation/results/optdigits_estimator_redraws.csv`
- `simulation/results/optdigits_estimator_ess_bins.csv`
- `simulation/results/optdigits_estimator_error_bins.csv`
- `simulation/results/optdigits_certified_learning_paths.csv`
- `simulation/results/optdigits_certified_learning_final.csv`
- `figures/optdigits_estimator_comparison.pdf`
- `figures/optdigits_certified_learning.pdf`

`optdigits_categorical_theory.py` and `ess_policy_optimization.py` are retained as exploratory and legacy scripts. They are not used for the reported Optdigits figures.
"""
    (ROOT / "simulation" / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    update_main_tex()
    update_readme()


if __name__ == "__main__":
    main()
