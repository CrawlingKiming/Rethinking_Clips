from pathlib import Path

path = Path("main.tex")
text = path.read_text(encoding="utf-8")

start = r"\subsection{When does clipping becomes necessary?}"
end = r"\section{Contextual-bandit validation of the framework}"
if text.count(start) != 1 or text.count(end) != 1:
    raise RuntimeError("Could not identify the main-text replacement block uniquely.")

new_block = r'''\subsection{When does clipping become necessary?}
\label{sec:clipping-recovery}

The preceding theorem evaluates the unmodified sequence-level estimator. It does not determine how a token-level coefficient rule should be chosen. Classical truncated importance sampling replaces the complete weight by a bounded weight and studies the resulting bias and variance as functions of the truncation level and $d_2$ \citep{metelli2020is}. Our setting is different. We modify each token coefficient, not the complete-response weight, and detached coefficient truncation keeps the score gradient active after the coefficient reaches its cap. The one-sided concentration results for nonnegative scalar estimands therefore do not transfer directly. The useful principle is instead to compare token-level rules by the reliability of the gradient estimator they induce.

Let $r_{k,t}$ denote the token likelihood ratio, let $s_{k,t}:=\nabla_\theta\log P_{\theta_k}(Y_t\mid X,Y_{<t})$ denote the token score, and let $A_k:=R-b(X)$. For any coefficient rule $u:\mathbb R_+\times\mathbb R\to\mathbb R$, define the response contribution and its estimator by
\begin{equation}
 \psi_k(u;Z):=A_k\sum_{t=1}^{T}u(r_{k,t},A_k)s_{k,t},
 \qquad
 \widehat g_k(u):=\frac1N\sum_{i=1}^N\psi_k(u;Z_i).
 \label{eq:token-rule-estimator}
\end{equation}
The target remains the population gradient $g_k$. Thus any discrepancy caused by using a token-local coefficient rather than the complete-response weight is included in the estimator bias. Define
\begin{equation}
 b_k(u):=\E_Q[\psi_k(u;Z)]-g_k,
 \qquad
 v_k(u):=\E_Q\left\|\psi_k(u;Z)-\E_Q[\psi_k(u;Z)]\right\|_2^2.
 \label{eq:token-rule-bias-variance}
\end{equation}
Independence gives the total gradient risk
\begin{equation}
 m_k(u):=\E\|\widehat g_k(u)-g_k\|_2^2
 =\|b_k(u)\|_2^2+\frac{v_k(u)}{N}.
 \label{eq:clipping-risk-decomposition}
\end{equation}

\begin{remark}[Token weight truncation and PPO masking]
\label{rem:token-truncation-vs-clipping}
A detached upper truncation uses $u_{\mathrm T}(r,A):=\min\{r,c\}$. Its coefficient remains nonzero when $r>c$, and
\begin{equation}
 \|\psi_k(u_{\mathrm T};Z)\|_2
 \le c|A_k|\sum_{t=1}^{T}\|s_{k,t}\|_2.
 \label{eq:token-truncation-envelope}
\end{equation}
Hence token weight truncation gives a direct second-moment envelope whenever the score contribution on the right has a finite second moment.

PPO defines a different rule. Away from the clipping boundary, its gradient is represented by $u_{\mathrm P}(r,A):=rM_\epsilon(r,A)$, where $M_\epsilon(r,A)=\mathbf 1\{A\ge0,\ r\le1+\epsilon\}+\mathbf 1\{A<0,\ r\ge1-\epsilon\}$. For $A\ge0$ and $c=1+\epsilon$,
\begin{equation}
 0\le r-u_{\mathrm T}(r,A)=(r-c)_+
 \le r\mathbf 1\{r>c\}=r-u_{\mathrm P}(r,A).
 \label{eq:truncation-mask-distortion}
\end{equation}
Thus truncation perturbs the positive-advantage upper branch less and preserves a bounded gradient after the cap, while PPO removes that gradient. On the negative-advantage lower branch, PPO masks small ratios whereas upper truncation does not. Neither rule therefore dominates on every sample. They are also distinct from truncating the sequence weight $W_k=\prod_t r_{k,t}$.
\end{remark}

For a common step size $0<\eta_k\le1/L_k$, the fixed-step smoothness argument gives
\begin{equation}
 \E\left[J(\theta_k+\eta_k\widehat g_k(u))-J(\theta_k)\right]
 \ge\frac{\eta_k}{2}\left\{\|g_k\|_2^2-m_k(u)\right\}.
 \label{eq:token-rule-progress}
\end{equation}
A more reliable coefficient rule therefore gives a stronger expected-improvement certificate. Comparing two rules reduces the question to one bias and variance inequality.

\begin{proposition}[Estimator reliability under coefficient control]
\label{prop:clipping-crossover}
For any two coefficient rules $u_1$ and $u_2$, the estimator $\widehat g_k(u_2)$ has lower total gradient risk than $\widehat g_k(u_1)$ if and only if
\begin{equation}
 v_k(u_1)-v_k(u_2)
 >N\left\{\|b_k(u_2)\|_2^2-\|b_k(u_1)\|_2^2\right\}.
 \label{eq:crossover}
\end{equation}
\end{proposition}

Proposition~\ref{prop:clipping-crossover} states the estimator-level tradeoff directly. A stronger rule is preferable only when its additional variance reduction exceeds its additional squared bias. This comparison is more general than a distinction between clipped and unclipped objectives because it applies to any two token coefficient rules.

\begin{corollary}[When a stronger rule recovers reliability]
\label{cor:clipping-needed}
For a common step size $0<\eta_k\le1/L_k$, rule $u_2$ has a positive expected-improvement certificate while rule $u_1$ does not exactly when
\begin{equation}
 m_k(u_2)<\|g_k\|_2^2\le m_k(u_1).
 \label{eq:clipping-needed}
\end{equation}
\end{corollary}

\begin{remark}[ESS and the strength of coefficient control]
\label{rem:high-ess-clipping}
The role of $\rho_k$ is to identify the reliability regime of the fixed rollout. When $\rho_k$ is high, Equation~\eqref{eq:rho-gradient-mse-bound} gives a small error bound for the unmodified sequence estimator. In this regime, a loose nonzero token cap provides bounded influence while preserving most of the outbound signal, so aggressive PPO-type masking can be unnecessarily restrictive. When $\rho_k$ is low, the sequence-level reliability certificate deteriorates. Stronger masking can then become preferable if its additional variance reduction satisfies Proposition~\ref{prop:clipping-crossover}. A low value of $\rho_k$ does not prove that masking is better, and a high value does not prove that truncation is optimal. ESS identifies when stronger control can become worth its bias, while the risk comparison determines whether a particular rule is reliable.
\end{remark}

The theoretical quantity $\rho_k$ is a population object. During training, we use the standard normalized plug-in estimate
\begin{equation}
 \widehat\rho_k
 :=\frac{(\sum_{i=1}^N W_{k,i})^2}{N\sum_{i=1}^N W_{k,i}^2}.
 \label{eq:sample-ess}
\end{equation}
It is computed from the complete-response ratios before any token coefficient is modified. A finite batch can miss rare, large ratios and can therefore overstate population support, so $\widehat\rho_k$ is an observable diagnostic rather than an exact replacement for $\rho_k$ in the theorem. The numerical value $0.1$ used below is a prespecified operational threshold, not a universal constant derived from the theory.

'''

before, rest = text.split(start, 1)
_, after = rest.split(end, 1)
text = before + new_block + end + after

proof_start = r"\subsection{Risk comparison for a general modification}"
proof_end = r"\subsection{Intact prompt groups}"
if text.count(proof_start) != 1 or text.count(proof_end) != 1:
    raise RuntimeError("Could not identify the appendix replacement block uniquely.")

new_proof = r'''\subsection{Risk comparison for token coefficient rules}

For a coefficient rule $u$, let $\psi_{k,i}(u):=\psi_k(u;Z_i)$. Add and subtract its expectation:
\begin{equation}
 \widehat g_k(u)-g_k
 =\left\{\widehat g_k(u)-\E_Q[\psi_k(u;Z)]\right\}+b_k(u).
 \label{eq:modified-bias-split}
\end{equation}
Squaring and taking expectation gives
\begin{align}
 m_k(u)
 &=\E\left\|\widehat g_k(u)-\E_Q[\psi_k(u;Z)]\right\|_2^2
 +\|b_k(u)\|_2^2
 \nonumber\\
 &\quad+2b_k(u)^\top\E\left[\widehat g_k(u)-\E_Q[\psi_k(u;Z)]\right]
 \nonumber\\
 &=\frac{v_k(u)}{N}+\|b_k(u)\|_2^2,
 \label{eq:modified-risk-proof}
\end{align}
where the cross term is zero and independence gives the factor $1/N$. Therefore, for two rules $u_1$ and $u_2$,
\begin{align}
 m_k(u_2)<m_k(u_1)
 &\iff
 \|b_k(u_2)\|_2^2+\frac{v_k(u_2)}{N}
 <\|b_k(u_1)\|_2^2+\frac{v_k(u_1)}{N}
 \nonumber\\
 &\iff
 v_k(u_1)-v_k(u_2)
 >N\left\{\|b_k(u_2)\|_2^2-\|b_k(u_1)\|_2^2\right\},
 \label{eq:crossover-proof}
\end{align}
which proves Proposition~\ref{prop:clipping-crossover}.

It remains to connect total risk to the expected policy-improvement certificate. For any random estimate $\widehat g$ of $g_k$ and any $0<\eta_k\le1/L_k$, smoothness gives
\begin{equation}
 J(\theta_k+\eta_k\widehat g)-J(\theta_k)
 \ge
 \eta_k g_k^\top\widehat g
 -\frac{L_k\eta_k^2}{2}\|\widehat g\|_2^2.
 \label{eq:fixed-step-smoothness}
\end{equation}
The polarization identity gives
\begin{equation}
 2g_k^\top\widehat g
 =\|g_k\|_2^2+\|\widehat g\|_2^2-\|\widehat g-g_k\|_2^2.
 \label{eq:polarization}
\end{equation}
Substituting this identity and using $\eta_k\le1/L_k$ yields
\begin{align}
 J(\theta_k+\eta_k\widehat g)-J(\theta_k)
 &\ge
 \frac{\eta_k}{2}\left\{\|g_k\|_2^2-\|\widehat g-g_k\|_2^2\right\}
 \nonumber\\
 &\quad+\frac{\eta_k}{2}(1-L_k\eta_k)\|\widehat g\|_2^2
 \nonumber\\
 &\ge
 \frac{\eta_k}{2}\left\{\|g_k\|_2^2-\|\widehat g-g_k\|_2^2\right\}.
 \label{eq:fixed-step-polarized}
\end{align}
Taking expectation and setting $\widehat g=\widehat g_k(u)$ proves Equation~\eqref{eq:token-rule-progress}. The lower bound is positive exactly when $m_k(u)<\|g_k\|_2^2$. Applying this statement to $u_1$ and $u_2$ proves Corollary~\ref{cor:clipping-needed}.

'''

before, rest = text.split(proof_start, 1)
_, after = rest.split(proof_end, 1)
text = before + new_proof + proof_end + after

text = text.replace(
    "This is the batch-size effect\npredicted by the factor $N\\|b_C\\|_2^2$ in\nEquation~\\eqref{eq:crossover}, applied to the block-level estimator described\nabove.",
    "This is the batch-size effect predicted by Equation~\\eqref{eq:crossover}, applied to the block-level estimator described above."
)

for forbidden in [
    "Consider the clipping strategy on the ..",
    r"\widehat g_{\phi,k}",
    r"b_{\phi,k}",
    r"v_{\phi,k}",
    r"m_{\phi,k}",
    "\u2014",
]:
    if forbidden in text:
        raise RuntimeError(f"Forbidden stale text remains: {forbidden}")

if "---" in new_block or "  " in new_block:
    raise RuntimeError("New subsection contains an em-dash command or double spacing.")

path.write_text(text, encoding="utf-8")
print("Rewrote the estimator-reliability subsection and its appendix proof.")
