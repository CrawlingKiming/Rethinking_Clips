"""Rewrite the Optdigits section after running the final regime validation."""

from __future__ import annotations

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


def pm(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}\\pm{se:.{digits}f}"


def main_section(values: dict[str, float]) -> str:
    raw_loss = values["raw_certificate_loss_rho"]
    ppo_loss = values["ppo_certificate_loss_rho"]
    ppo_only_min = values["ppo_only_rho_min"]
    ppo_only_max = values["ppo_only_rho_max"]
    high_relative = values["high_rho_raw_relative_risk"]
    crossover = int(values["persistent_crossover_iteration"])
    early_iteration = int(values["max_early_raw_advantage_iteration"])
    early_advantage_pp = 100.0 * values["max_early_raw_advantage"]
    raw_final = values["raw_final"]
    raw_final_se = values["raw_final_se"]
    ppo_final = values["ppo_final"]
    ppo_final_se = values["ppo_final_se"]
    ppo_advantage_pp = -100.0 * values["final_raw_minus_ppo"]
    ppo_advantage_se_pp = 100.0 * values["final_gap_se"]
    eta = values["learning_rate"]
    eta_max = values["certified_eta_max"]

    section = r"""
\section{Theoretical validation on Optdigits}
\label{sec:simulation}

The theory makes two predictions that can be separated in a finite model. First, an unmodified update is reliable while its gradient-estimation error remains smaller than the population-gradient signal. Second, after this certificate is lost, a conservative estimator is useful only if its variance reduction compensates for the bias it introduces. Optdigits allows the objective, the population gradient, normalized ESS, and the MSE of both estimators to be evaluated exactly.

\paragraph{Categorical contextual bandit.}
We convert Optdigits into a one-step contextual bandit \citep{alpaydin1998optdigits}. Each image is a context, the action is a digit in $\{0,\ldots,9\}$, and the reward is one only for the correct class. The policy is a linear softmax classifier. One action is sampled per image, and the exact rollout value $Q(y\mid x)$ gives the advantage $A=R-Q(y\mid x)$. All evaluated updates use $\eta=@@ETA@@$. Appendix~\ref{app:bandit} gives a global smoothness certificate with $1/\bar L=@@ETA_MAX@@$, so the step-size condition in Lemma~\ref{lem:gradient-error-progress} is satisfied.

\paragraph{An exact transition across support regimes.}
We first isolate policy mismatch from the optimization trajectory. A softened fitted classifier is fixed as the rollout policy $Q$. The current policy is then moved along a deterministic path toward the fully fitted classifier. As the two policies separate, normalized ESS decreases. At every point on this path, we compute the exact population gradient and the exact MSE of an unmodified estimator and a PPO-masked estimator with minibatch size $N=320$.

Figure~\ref{fig:optdigits-transition} follows the complete theoretical chain. At the on-policy endpoint, both estimators have relative MSE @@HIGH_RELATIVE@@, so both have a positive expected-improvement certificate. The unmodified certificate becomes nonpositive when normalized ESS reaches @@RAW_LOSS@@. PPO reduces the estimator variance enough to retain a positive certificate over the interval @@PPO_ONLY_MIN@@ to @@PPO_ONLY_MAX@@, and its certificate becomes nonpositive at @@PPO_LOSS@@. These numerical boundaries are specific to this finite population, batch size, and policy path. They are not universal ESS thresholds.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/optdigits_reliability_transition.pdf}
  \caption{Exact validation of the estimator-reliability framework. The current policy moves away from a fixed rollout policy along a controlled path. (a) The exact MSE of both gradient estimators increases as normalized ESS decreases. (b) The unmodified estimator loses its reliability certificate when MSE reaches the population-gradient signal. PPO delays this transition by reducing variance. (c) The corresponding smoothness lower bound produces three regimes: both estimators are certified, only PPO is certified, and neither estimator is certified. All quantities are evaluated exactly over the finite Optdigits population.}
  \label{fig:optdigits-transition}
\end{figure}

\paragraph{Cumulative effect over policy iterations.}
The controlled path identifies the mechanism but does not show how the two estimators accumulate progress during training. We therefore report a longer on-policy experiment that contains both phases. Each policy iteration draws 160 fresh images, samples one action per image, and performs one epoch of 40 sequential minibatch updates. The minibatch size is four, which intentionally makes finite-sample reliability consequential. The learning rate remains @@ETA@@ and therefore satisfies the same global smoothness condition.

The setting was selected using a pilot grid designed to expose both the early signal-preservation regime and the later variance-control regime. Figure~\ref{fig:optdigits-iterations} reports a disjoint final evaluation with 100 paired seeds. The unmodified estimator learns faster initially and leads by as much as @@EARLY_ADV@@ percentage points at policy iteration @@EARLY_ITER@@. PPO overtakes persistently at iteration @@CROSSOVER@@. At iteration 25, the population value is $@@RAW_FINAL@@$ for the unmodified estimator and $@@PPO_FINAL@@$ for PPO, a paired PPO advantage of $@@PPO_ADV@@$ percentage points.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.78\linewidth]{figures/optdigits_policy_iterations.pdf}
  \caption{Population value at the end of each policy iteration. The unmodified estimator learns faster in the early phase, when preserving the full gradient signal is useful. PPO becomes better in the later phase, when variance control is worth its masking bias. Curves show means and 95\% confidence intervals over 100 paired replications. The dotted line marks the first iteration after which PPO remains ahead.}
  \label{fig:optdigits-iterations}
\end{figure}

The two experiments play different roles. Figure~\ref{fig:optdigits-transition} validates the theorem with exact population quantities, while Figure~\ref{fig:optdigits-iterations} shows the cumulative consequence of committing to either estimator throughout training. No ESS threshold, oracle update, or adaptive switching rule is used in Optdigits.
"""

    replacements = {
        "@@ETA@@": f"{eta:.2f}",
        "@@ETA_MAX@@": f"{eta_max:.3f}",
        "@@HIGH_RELATIVE@@": f"${high_relative:.3f}$",
        "@@RAW_LOSS@@": f"${raw_loss:.3f}$",
        "@@PPO_LOSS@@": f"${ppo_loss:.3f}$",
        "@@PPO_ONLY_MIN@@": f"${ppo_only_min:.3f}$",
        "@@PPO_ONLY_MAX@@": f"${ppo_only_max:.3f}$",
        "@@EARLY_ADV@@": f"{early_advantage_pp:.2f}",
        "@@EARLY_ITER@@": str(early_iteration),
        "@@CROSSOVER@@": str(crossover),
        "@@RAW_FINAL@@": pm(raw_final, raw_final_se),
        "@@PPO_FINAL@@": pm(ppo_final, ppo_final_se),
        "@@PPO_ADV@@": pm(ppo_advantage_pp, ppo_advantage_se_pp, digits=2),
    }
    for key, value in replacements.items():
        section = section.replace(key, value)
    return section.strip()


def appendix_section(values: dict[str, float]) -> str:
    eta = values["learning_rate"]
    eta_max = values["certified_eta_max"]
    lambda_max = values["feature_cov_lambda_max"]
    smoothness = values["global_smoothness_bound"]

    appendix = r"""
\section{Categorical Optdigits protocol}
\label{app:bandit}

This appendix gives the complete construction used in Section~\ref{sec:simulation}. The goal is not classification generalization. The supplied training and test files are concatenated into a finite population so that the target objective and all estimator moments can be evaluated exactly.

\subsection{Finite population and policy}

Optdigits contains 5,620 labeled handwritten-digit images \citep{alpaydin1998optdigits}. We divide each of the 64 pixel-count features by 16 and append an intercept. The finite population is $\{(x_j,y_j)\}_{j=1}^M$ with $M=5{,}620$, $x_j\in\mathbb R^{65}$, and $y_j\in\{0,\ldots,9\}$. The policy is
\begin{equation}
 \pi_\theta(a\mid x)
 =\frac{\exp(\theta_a^\top x)}{\sum_{c=0}^9\exp(\theta_c^\top x)},
 \qquad a\in\{0,\ldots,9\}.
 \label{eq:categorical-policy}
\end{equation}
With reward $R(x_j,a)=\mathbf 1\{a=y_j\}$, the exact population objective and gradient are
\begin{align}
 J(\theta)&=\frac1M\sum_{j=1}^M\pi_\theta(y_j\mid x_j),
 \label{eq:categorical-value}\\
 g(\theta)&=\frac1M\sum_{j=1}^M
 \pi_\theta(y_j\mid x_j)
 \{e_{y_j}-\pi_\theta(\cdot\mid x_j)\}x_j^\top.
 \label{eq:categorical-gradient}
\end{align}
A full-population cross-entropy fit provides a reference classifier. Multiplying its weights by a scale smaller than one produces the stochastic initial policies used below.

\subsection{Global smoothness certificate}

Let $p=\operatorname{softmax}(z)$ and $q=p_y$. The Hessian of one correct-class probability with respect to the logits is
\begin{equation}
 \nabla_z^2q
 =q\left[(e_y-p)(e_y-p)^\top-\{\operatorname{diag}(p)-pp^\top\}\right].
 \label{eq:categorical-logit-hessian}
\end{equation}
The categorical covariance has operator norm at most $1/2$, and $\|e_y-p\|_2^2\le2(1-q)^2$. Therefore $\|\nabla_z^2q\|_{\mathrm{op}}\le1/2$. For a perturbation $U$ of the softmax parameter matrix,
\begin{align}
 \left|\nabla^2J(\theta)[U,U]\right|
 &\le\frac{1}{2M}\sum_{j=1}^M\|Ux_j\|_2^2
 \nonumber\\
 &\le\frac12\lambda_{\max}\left(\frac1M\sum_{j=1}^Mx_jx_j^\top\right)\|U\|_F^2.
 \label{eq:categorical-smoothness-bound}
\end{align}
Thus $\bar L=\tfrac12\lambda_{\max}(M^{-1}\sum_jx_jx_j^\top)$ is a valid global smoothness constant. The Optdigits population gives $\lambda_{\max}=@@LAMBDA@@$, $\bar L=@@SMOOTHNESS@@$, and $1/\bar L=@@ETA_MAX@@$. Every evaluated update uses $\eta=@@ETA@@\le1/\bar L$.

\subsection{Gradient estimators}

A rollout policy $Q$ samples one action for each context. Since the reward is exact class match, the rollout value is $Q(y\mid x)$ and the detached advantage is $A=R-Q(y\mid x)$. For a minibatch of size $N$, the unmodified estimator is
\begin{equation}
 \widehat g_{\mathrm{raw}}(\theta)
 =\frac1N\sum_{i=1}^N
 \frac{\pi_\theta(a_i\mid x_i)}{Q(a_i\mid x_i)}
 A_i\nabla_\theta\log\pi_\theta(a_i\mid x_i).
 \label{eq:categorical-raw-gradient}
\end{equation}
The PPO estimator applies the standard advantage-dependent mask with radius $0.2$ to each sampled contribution. For $e\in\{\mathrm{raw},\mathrm{PPO}\}$, let $Z_e$ be one contribution. Its exact conditional MSE is
\begin{equation}
 m_e(\theta,Q)
 =\left\|\E_Q[Z_e]-g(\theta)\right\|_2^2
 +\frac1N\E_Q\left\|Z_e-\E_Q[Z_e]\right\|_2^2.
 \label{eq:categorical-exact-risk}
\end{equation}
The finite context-action space allows Equation~\eqref{eq:categorical-exact-risk} to be evaluated by summing over all $5{,}620\times10$ pairs.

\subsection{Controlled reliability path}

Let $\theta_0$ be the fitted classifier scaled by $0.2$, and fix $Q=\pi_{\theta_0}$. Let $\theta_\star$ denote the unscaled fitted classifier. We evaluate $\theta(s)=\theta_0+s(\theta_\star-\theta_0)$ for 201 equally spaced values of $s$ between 0 and 2.5. For each policy, we compute the exact normalized ESS, population gradient, and the MSE of both estimators for $N=320$. The expected-improvement certificate plotted in Figure~\ref{fig:optdigits-transition} is
\begin{equation}
 C_e(\theta,Q)
 =\frac{\eta}{2}\left\{\|g(\theta)\|_2^2-m_e(\theta,Q)\right\},
 \qquad e\in\{\mathrm{raw},\mathrm{PPO}\}.
 \label{eq:optdigits-certificate}
\end{equation}
This is the expected version of Lemma~\ref{lem:gradient-error-progress}. The controlled path is used only to expose the three reliability regimes. Its numerical transition points are not proposed as thresholds.

\subsection{Longitudinal policy-iteration experiment}

The longitudinal setting uses 160 fresh contexts per policy iteration, 40 sequential minibatches of size four, one optimization epoch, learning rate $0.17$, PPO radius $0.2$, and an initialization scale of $0.2$. The small minibatch is intentional because a large $N$ would keep both estimators in the reliable regime and conceal the finite-sample transition.

We selected this stress setting transparently. A pilot grid varied rollout size in $\{160,320\}$, minibatch count in $\{8,16,40\}$, learning rate in $\{0.14,0.17\}$, PPO radius in $\{0.05,0.1,0.2\}$, and initialization scale in $\{0.2,0.35,0.5\}$. Three pilot seeds and 15 policy iterations were used to identify settings containing both an early unmodified advantage and a later PPO advantage. Four candidates were then checked on 30 new seeds. The reported condition was the only shortlisted setting with a persistent crossover. Figure~\ref{fig:optdigits-iterations} uses a further disjoint set of 100 paired seeds and 25 policy iterations. Both estimators receive the same context draws, action-sampling uniforms, and minibatch order within each replication.
"""

    replacements = {
        "@@LAMBDA@@": f"{lambda_max:.3f}",
        "@@SMOOTHNESS@@": f"{smoothness:.3f}",
        "@@ETA_MAX@@": f"{eta_max:.3f}",
        "@@ETA@@": f"{eta:.2f}",
    }
    for key, value in replacements.items():
        appendix = appendix.replace(key, value)
    return appendix.strip()


def readme_text() -> str:
    return r"""# Optdigits theoretical validation

`optdigits_regime_validation.py` produces the Optdigits results reported in the paper. The task is a one-step contextual bandit with the ten digit labels as actions.

The script performs two experiments.

1. It moves a current policy away from a fixed rollout policy along a controlled path. Because the population is finite, normalized ESS, the population gradient, the exact MSE of the unmodified and PPO estimators, and their expected-improvement certificates are evaluated exactly. The plotted certificate uses batch size 320.
2. It runs a 25-iteration on-policy stress test. Each policy iteration samples 160 fresh images and traverses 40 minibatches of size four once. The figure reports population value only at policy-iteration endpoints. The final curve uses 100 seeds disjoint from the pilot search.

All evaluated updates use learning rate 0.17, below the global smoothness limit 0.173. No ESS threshold, oracle update, or adaptive switching rule is used.

Run from the repository root:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/optdigits_regime_validation.py --replications 100
```

Main outputs:

- `simulation/results/optdigits_controlled_certificate_path.csv`
- `simulation/results/optdigits_policy_iteration_curve.csv`
- `simulation/results/optdigits_policy_iteration_runs.csv`
- `simulation/results/optdigits_regime_summary.txt`
- `figures/optdigits_reliability_transition.pdf`
- `figures/optdigits_policy_iterations.pdf`

`optdigits_categorical_theory.py` contains shared finite-population utilities. Other Optdigits CSV files in `simulation/results` document exploratory searches and are not used to report the final curves.
"""


def update_main_tex(values: dict[str, float]) -> None:
    path = ROOT / "main.tex"
    text = path.read_text(encoding="utf-8")

    start = text.index("\\section{Theoretical validation on Optdigits}")
    end = text.index("\\section{Language-model evidence for delayed failure and recovery}")
    text = text[:start] + main_section(values) + "\n\n" + text[end:]

    appendix_start = text.index("\\section{Categorical Optdigits protocol}")
    appendix_end = text.index("\\section{Additional RLVR diagnostics}")
    text = (
        text[:appendix_start]
        + appendix_section(values)
        + "\n\n"
        + text[appendix_end:]
    )

    conclusion_start = text.index("The categorical contextual bandit validates")
    conclusion_end = text.index("Future work", conclusion_start)
    conclusion = (
        "The categorical contextual bandit validates the estimator-reliability "
        "framework in two complementary ways. Along a controlled policy path, "
        "the unmodified update first loses its positive certificate as effective "
        "support deteriorates, PPO temporarily recovers that certificate through "
        "variance reduction, and both estimators eventually become unsupported. "
        "In a separate held-out policy-iteration experiment, the unmodified "
        "estimator learns faster early, while PPO overtakes later and finishes "
        "higher. The language-model runs show the same qualitative ordering at "
        "scale: permissive learning succeeds while effective support is broad, "
        "and selective protection becomes useful only after support deteriorates.\n"
    )
    text = text[:conclusion_start] + conclusion + text[conclusion_end:]

    if "—" in text or "---" in text:
        raise RuntimeError("em dash detected")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    values = read_summary(
        ROOT / "simulation" / "results" / "optdigits_regime_summary.txt"
    )
    update_main_tex(values)
    (ROOT / "simulation" / "README.md").write_text(
        readme_text(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
