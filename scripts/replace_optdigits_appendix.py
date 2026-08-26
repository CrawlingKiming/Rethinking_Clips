from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "main.tex"
text = path.read_text(encoding="utf-8")

old_conclusion = """The contextual bandit confirms the predicted relation among normalized ESS,
gradient error, harmful updates, and estimator crossover. The language-model
runs show the same temporal ordering at scale: permissive learning succeeds
while effective support is broad, and selective protection becomes useful only
after support deteriorates. Future work should develop tighter online estimates
of the contribution scale and test whether the resulting reliability boundary
transfers across models, rollout sizes, and verifier tasks.
"""
new_conclusion = """The categorical contextual bandit confirms the two links used by the theory.
Lower normalized ESS coincides with larger gradient-estimation error, and
harmful updates appear after the realized error reaches the scale of the
population gradient signal. The language-model runs show a related temporal
ordering at scale: permissive learning succeeds while effective support is
broad, and selective protection becomes useful only after support deteriorates.
Future work should develop tighter online estimates of the contribution scale
and test whether the resulting reliability boundary transfers across models,
rollout sizes, and verifier tasks.
"""
if old_conclusion not in text:
    raise RuntimeError("Conclusion paragraph not found")
text = text.replace(old_conclusion, new_conclusion, 1)

start_marker = r"\section{Compact contextual-bandit protocol and supporting results}"
end_marker = r"\section{Additional RLVR diagnostics}"
start = text.index(start_marker)
end = text.index(end_marker)

appendix = r'''\section{Categorical Optdigits protocol}
\label{app:bandit}

This appendix gives the full construction used in Section~\ref{sec:simulation}. The experiment is a one-step contextual bandit with a ten-class action space. It contains no ESS-conditioned update rule.

\subsection{Finite population and policy}

Optdigits contains 5,620 labeled handwritten-digit images \citep{alpaydin1998optdigits}. We concatenate the supplied training and test files, divide each of the 64 pixel-count features by 16, and append an intercept. The resulting finite population is $\{(x_j,y_j)\}_{j=1}^{M}$ with $M=5{,}620$, $x_j\in\mathbb R^{65}$, and $y_j\in\{0,\ldots,9\}$. The original split is not used because the target is finite-population policy optimization rather than classification generalization.

The policy is a linear softmax model,
\begin{equation}
 \pi_\theta(a\mid x)
 =\frac{\exp(\theta_a^\top x)}{\sum_{c=0}^{9}\exp(\theta_c^\top x)},
 \qquad a\in\{0,\ldots,9\}.
 \label{eq:categorical-policy}
\end{equation}
The reward is $R(x_j,a)=\mathbf 1\{a=y_j\}$, so the exact population objective is
\begin{equation}
 J(\theta)=\frac1M\sum_{j=1}^{M}\pi_\theta(y_j\mid x_j).
 \label{eq:categorical-value}
\end{equation}
Writing $e_y$ for the one-hot vector of class $y$, the exact population gradient is
\begin{equation}
 g(\theta)
 =\frac1M\sum_{j=1}^{M}
 \pi_\theta(y_j\mid x_j)
 \{e_{y_j}-\pi_\theta(\cdot\mid x_j)\}x_j^\top.
 \label{eq:categorical-gradient}
\end{equation}
Both Equations~\eqref{eq:categorical-value} and \eqref{eq:categorical-gradient} are evaluated by summing over all images. The policy is initialized by 400 full-population cross-entropy steps with step size $0.5$, after which the fitted weights are multiplied by $0.35$. This produces an initial population value of $0.4382$ while retaining substantial action randomness.

\subsection{One-epoch rollout and update}

Each policy iteration uses the following procedure.
\begin{enumerate}
 \item Freeze the current parameters as the rollout policy $Q$.
 \item Sample 600 images without replacement and draw one action from $Q(\cdot\mid x)$ for each image.
 \item Set $A=R-Q(y\mid x)$, where $Q(y\mid x)$ is used as a detached context-dependent baseline.
 \item Shuffle the 600 observations into 12 minibatches of size 50 and traverse them once.
 \item Discard the rollout and collect a new batch from the updated policy.
\end{enumerate}
Thus each rollout is used for one optimization epoch. At a minibatch update, the unmodified estimator is
\begin{equation}
 \widehat g(\theta)
 =\frac1n\sum_{i=1}^{n}
 \frac{\pi_\theta(a_i\mid x_i)}{Q(a_i\mid x_i)}
 A_i\nabla_\theta\log\pi_\theta(a_i\mid x_i).
 \label{eq:categorical-raw-gradient}
\end{equation}
The static PPO trajectory uses the usual advantage-dependent mask with radius $0.2$. Both static trajectories use learning rate $3.0$ and six rollout cycles. They are used only to generate a broad collection of pre-update policy states. No trajectory changes its update rule according to ESS.

\subsection{Population ESS and frozen-state redraws}

For a current policy $P_\theta$ and rollout policy $Q$, the normalized population ESS is exact in this categorical model:
\begin{equation}
 \rho(\theta,Q)
 =\left\{
 \frac1M\sum_{j=1}^{M}\sum_{a=0}^{9}
 \frac{\pi_\theta(a\mid x_j)^2}{Q(a\mid x_j)}
 \right\}^{-1}.
 \label{eq:categorical-population-ess}
\end{equation}
We order all pre-update states by Equation~\eqref{eq:categorical-population-ess} and select 30 approximately equally spaced states over the observed range. At each state, 80 independent minibatches of 128 context-action pairs are redrawn from the frozen rollout policy. These redraws estimate the conditional gradient MSE. The first 12 redraws also receive a common diagnostic step of size $0.25$, after which Equation~\eqref{eq:categorical-value} is recomputed exactly. None of these diagnostic redraws changes the training trajectory.

Table~\ref{tab:categorical-ess-bins} reports the six equal-count ESS bins used in Figure~\ref{fig:optdigits-theory}. Each bin contains five frozen states. The MSE is averaged over 80 redraws per state, and the harmful-update rate is computed from the 12 diagnostic steps per state.

\begin{table}[htbp]
\centering
\small
\caption{Categorical Optdigits diagnostics across population-ESS bins.}
\label{tab:categorical-ess-bins}
\begin{tabular}{rrr}
\toprule
Median population ESS & Unmodified gradient MSE & Harmful updates (\%) \\
\midrule
$0.007$ & $0.1981$ & $30.0$ \\
$0.293$ & $0.1980$ & $1.7$ \\
$0.554$ & $0.0325$ & $1.7$ \\
$0.720$ & $0.0216$ & $1.7$ \\
$0.843$ & $0.0168$ & $0.0$ \\
$1.000$ & $0.0065$ & $0.0$ \\
\bottomrule
\end{tabular}
\end{table}

The second diagnostic uses the realized ratio $\|\widehat g-g\|_2^2/\|g\|_2^2$. Among 209 redraws below one, no update decreases the population objective. Among 151 redraws at or above one, 21 updates decrease it. The exact population-gradient step is positive at every frozen state. This separates estimator failure from the absence of an improving population direction.

'''
text = text[:start] + appendix + text[end:]
path.write_text(text, encoding="utf-8")
