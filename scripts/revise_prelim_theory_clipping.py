from pathlib import Path
import re

PATH = Path("main.tex")

PRELIMINARIES = r'''\section{Preliminaries}
\label{sec:setting}

\subsection{Policy-gradient estimation from a fixed rollout}

Let $X\sim\nu$ denote a prompt, let $Q(\cdot\mid X)$ be the rollout policy, and let $P_\theta(\cdot\mid X)$ be the current policy. A complete response is denoted by $Y=(Y_1,\ldots,Y_T)$, the joint sample is $Z=(X,Y)\in\mathcal Z$, and $R(Z)$ is its verifier reward. The population objective is $J(\theta):=\E_{P_\theta}[R]$.

In RLVR, responses are generated under $Q$ and then reused while the learner changes from $Q$ to $P_\theta$. The objective and the available data are therefore defined under different policies. Assuming $P_\theta(\cdot\mid X)\ll Q(\cdot\mid X)$, this mismatch is corrected by the complete-response likelihood ratio
\begin{equation}
 W_\theta(Z):=\frac{P_\theta(Y\mid X)}{Q(Y\mid X)}.
 \label{eq:sequence-ratio}
\end{equation}
Let $b(X)$ be a detached prompt-level baseline and define the policy-gradient contribution
\begin{equation}
 f_\theta(Z):=\{R(Z)-b(X)\}\nabla_\theta\log P_\theta(Y\mid X).
 \label{eq:gradient-contribution}
\end{equation}
The score identity and change of measure then give
\begin{equation}
 g(\theta):=\nabla J(\theta)=\E_Q[W_\theta f_\theta],
 \qquad
 \widehat g_N(\theta):=\frac1N\sum_{i=1}^N W_{\theta,i}f_{\theta,i}.
 \label{eq:true-gradient}
\end{equation}
Thus the policy gradient is itself a vector-valued importance-weighted mean, and $\widehat g_N(\theta)$ is its finite-rollout estimate.

This representation makes the statistical difficulty explicit. For a fixed policy, $\widehat g_N(\theta)$ targets the correct gradient, but its finite-sample accuracy depends on how unevenly the likelihood ratios distribute mass across responses. When a few responses carry most of the weight, the nominal batch size $N$ can greatly overstate the amount of information supporting the current update.

We analyze one optimization phase. The rollout batch is randomly partitioned into disjoint minibatches, and each minibatch is used at most once in the theoretical traversal. At update $k$, we condition on all previous updates and suppress this conditioning in the notation. The current policy is then fixed and the unused minibatch consists of independent draws from $Q$. When several responses share a prompt or a group-relative baseline, the intact prompt group is treated as one independent sampling unit. Appendix~\ref{app:proofs} gives this extension.

The remaining question is how to quantify the effective support that the rollout provides to the current policy. Since the estimation error is driven by moments of $W_\theta$, we next introduce a divergence that records these moments and the corresponding normalized effective sample size.

\subsection{R\'enyi divergence and normalized effective sample size}

R\'enyi divergence is a family of measures indexed by a moment order $\alpha>1$. For the current and rollout policies, define
\begin{equation}
 D_\alpha(P_\theta\|Q)
 :=\frac{1}{\alpha-1}\log\E_Q[W_\theta^\alpha],
 \qquad
 d_\alpha(P_\theta\|Q):=\exp\{D_\alpha(P_\theta\|Q)\}.
 \label{eq:renyi-divergence}
\end{equation}
Larger values of $\alpha$ place progressively more emphasis on rare, large likelihood ratios. The order-two case is especially relevant here because the mean-squared error of an importance-weighted mean is governed by the second moment of its weight.

At $\alpha=2$, $d_2(P_\theta\|Q)=\E_Q[W_\theta^2]$. Following importance-sampling analyses for policy optimization \citep{metelli2018pois,metelli2020is}, we define the normalized sequence effective sample size by
\begin{equation}
 \rho_\theta
 :=\frac{1}{d_2(P_\theta\|Q)}
 =\frac{1}{\E_Q[W_\theta^2]}
 =\exp\{-D_2(P_\theta\|Q)\}.
 \label{eq:population-ess}
\end{equation}
The quantity $\rho_\theta$ lies in $(0,1]$, and $N\rho_\theta$ is the corresponding effective sequence count. When $P_\theta=Q$, every sequence is fully effective and $\rho_\theta=1$. As the second moment of the likelihood ratio grows, $\rho_\theta$ decreases and the effective count contracts. Since $\E_Q[W_\theta]=1$, the identity $\E_Q[(W_\theta-1)^2]=\rho_\theta^{-1}-1$ shows that $\rho_\theta$ is exactly the inverse second-moment measure of ratio dispersion.

Importance-sampling theory makes this quantity operational. For a bounded integrand, the estimation error scales with $d_2(P_\theta\|Q)/N$, or equivalently with $1/(N\rho_\theta)$. Normalized ESS therefore measures the part of estimator reliability that is lost through policy mismatch. The next section formalizes this relation for the policy-gradient estimator and then connects it to policy improvement.'''

THEORY = r'''\section{Theoretical analysis}
\label{sec:theory}

The preliminaries identify normalized sequence ESS as the effective-support coordinate of the fixed rollout. Figure~\ref{fig:llm-motivation} asks when a decline in this support makes a permissive update unreliable. We answer the question in two steps. First, we convert $\rho_k$ into a high-probability bound on the error of the sampled policy gradient. Second, we show how that error controls the improvement produced by the next update.

At update $k$, write $W_k:=W_{\theta_k}$, $f_k:=f_{\theta_k}$, $g_k:=g(\theta_k)$, $\widehat g_k:=\widehat g_N(\theta_k)$, and $\rho_k:=\rho_{\theta_k}$. As stated above, the current policy is fixed after conditioning on the preceding updates, and the $N$ unused sampling units are independent draws from $Q$. We use the standard bounded-integrand condition $\|f_k\|_\infty:=\sup_{z\in\mathcal Z}\|f_k(z)\|_2<\infty$. This condition fixes the scale of one response contribution, while $\rho_k$ records the additional loss of reliability caused by policy mismatch.

\subsection{From normalized ESS to a gradient-error radius}
\label{sec:raw}

Change of measure gives $\E[\widehat g_k]=g_k$. A standard sample-mean calculation then gives $\E\|\widehat g_k-g_k\|_2^2=N^{-1}\{\E_Q[W_k^2\|f_k\|_2^2]-\|g_k\|_2^2\}$. Using $\|f_k(z)\|_2\le\|f_k\|_\infty$ and $\rho_k^{-1}=\E_Q[W_k^2]$ yields
\begin{equation}
 \E\|\widehat g_k-g_k\|_2^2
 \le
 \frac{\|f_k\|_\infty^2}{N\rho_k}.
 \label{eq:rho-gradient-mse-bound}
\end{equation}
Equation~\eqref{eq:rho-gradient-mse-bound} describes the average size of the estimation error. The actual update, however, is computed from one realized minibatch. We therefore convert the mean-square bound into an event-level guarantee.

\begin{lemma}[Normalized ESS controls the gradient-error radius]
\label{lem:rho-gradient-reliability}
Let $P_{\theta_k}$ and $Q$ be probability measures on $\mathcal Z$ such that $P_{\theta_k}\ll Q$ and $d_2(P_{\theta_k}\|Q)<\infty$. Let $Z_1,\ldots,Z_N$ be independent samples from $Q$, and let $f_k:\mathcal Z\to\mathbb R^d$ satisfy $\|f_k\|_\infty<\infty$. Define $\widehat g_k=N^{-1}\sum_{i=1}^N W_k(Z_i)f_k(Z_i)$, $g_k=\E_Q[W_kf_k]$, and $\rho_k=d_2(P_{\theta_k}\|Q)^{-1}$. Then, for any $0<\delta\le1$, with probability at least $1-\delta$, it holds that
\begin{equation}
 \|\widehat g_k-g_k\|_2
 \le
 \frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}.
 \label{eq:rho-gradient-concentration}
\end{equation}
\end{lemma}

Lemma~\ref{lem:rho-gradient-reliability} turns normalized ESS into a directly interpretable uncertainty radius. For fixed $N$ and $\|f_k\|_\infty$, the radius grows as $\rho_k^{-1/2}$. Equivalently, policy mismatch replaces the nominal batch size $N$ by the effective sequence count $N\rho_k$.

\subsection{From gradient accuracy to policy improvement}
\label{sec:mse-improvement}

The first lemma tells us when the sampled gradient is close to the population gradient. To determine whether that accuracy is sufficient, we must translate an error radius into a change in the population objective. The next lemma performs this step using smoothness.

\begin{lemma}[A reliable gradient estimate yields policy improvement]
\label{lem:gradient-error-progress}
Let $\widehat g$ be an estimate of $g=\nabla J(\theta)$ satisfying $\|\widehat g-g\|_2\le\varepsilon$. Suppose $J$ is $L$-smooth along the update segment, and define $\alpha=L^{-1}(1-\varepsilon/\|\widehat g\|_2)_+$, with $\alpha=0$ when $\widehat g=0$. Then
\begin{equation}
 J(\theta+\alpha\widehat g)-J(\theta)
 \ge
 \frac{1}{2L}
 \left(\|\widehat g\|_2-\varepsilon\right)_+^2.
 \label{eq:gradient-error-progress}
\end{equation}
\end{lemma}

The lower bound increases as the error radius decreases. Estimator reliability therefore determines both whether a nonzero step can be certified and how large the certificate can be. This is the same smoothness mechanism used in safe policy-gradient analyses \citep{pirotta2013adaptive,papini2022smoothing}.

\subsection{When does a permissive update remain reliable?}

Lemma~\ref{lem:rho-gradient-reliability} supplies an ESS-dependent error radius for the current minibatch. Lemma~\ref{lem:gradient-error-progress} converts any valid radius into population improvement. Applying the second lemma on the event supplied by the first gives the central result.

\begin{theorem}[When a permissive update remains reliable]
\label{thm:permissive-reliability}
Under the conditions of Lemma~\ref{lem:rho-gradient-reliability}, suppose that $J$ is $L_k$-smooth along the update segment. For any $0<\delta\le1$, define $\alpha_k=L_k^{-1}\{1-\|f_k\|_\infty/(\|\widehat g_k\|_2\sqrt{\delta N\rho_k})\}_+$, with $\alpha_k=0$ when $\widehat g_k=0$. Then, with probability at least $1-\delta$, it holds that
\begin{equation}
 J(\theta_k+\alpha_k\widehat g_k)-J(\theta_k)
 \ge
 \frac{1}{2L_k}
 \left(
 \|\widehat g_k\|_2
 -\frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
 \right)_+^2.
 \label{eq:permissive-reliability}
\end{equation}
\end{theorem}

The certificate is positive exactly when the observed gradient norm exceeds its ESS-dependent uncertainty radius. This comparison explains why failure can be delayed. The same rollout can support many useful updates while $N\rho_k$ remains large, and the certificate becomes inconclusive only after effective support has sufficiently deteriorated.

\begin{remark}[Interpretation of the reliability boundary]
\label{rem:rho-interpretation}
Theorem~\ref{thm:permissive-reliability} identifies a loss of certification, not a deterministic collapse point. A zero lower bound means that the rollout no longer certifies the sampled direction at the chosen confidence level. The realized update may still improve.

The result complements policy-deviation guarantees. Those guarantees study population-level consequences of changing the policy. The present result studies whether a fixed finite rollout still estimates the current policy gradient accurately enough. Both effects may matter, but they answer different questions.

For fixed $N$, contribution scale, confidence level, and observed gradient signal, all mismatch-dependent degradation enters through $\rho_k$. No universal numerical threshold follows because the required value also depends on those other quantities.
\end{remark}

The theorem diagnoses when the unmodified estimator loses a useful guarantee, but it does not imply that clipping will repair the problem. A modification can help only if the reduction in sampling error is large enough to compensate for the bias it introduces. We now state this comparison without committing to one particular clipping implementation.

\subsection{When can clipping recover reliability?}
\label{sec:clipping-recovery}

Let $\phi_k:\mathcal Z\to\mathbb R^d$ be any integrable transformation of one rollout sample, and define the corresponding estimator by
\begin{equation}
 \widehat g_{\phi,k}:=\frac1N\sum_{i=1}^N\phi_k(Z_i).
 \label{eq:general-modified-estimator}
\end{equation}
The unmodified importance-weighted estimator is recovered by setting $\phi_k(z)=W_k(z)f_k(z)$. For a general modification, define its bias $b_{\phi,k}:=\E_Q[\phi_k]-g_k$ and per-sample variance $v_{\phi,k}:=\E_Q\|\phi_k-\E_Q[\phi_k]\|_2^2$. Let $v_k:=\E_Q\|W_kf_k-g_k\|_2^2$ denote the corresponding per-sample variance of the unmodified estimator. Independence gives
\begin{equation}
 m_k:=\E\|\widehat g_k-g_k\|_2^2=\frac{v_k}{N},
 \qquad
 m_{\phi,k}:=\E\|\widehat g_{\phi,k}-g_k\|_2^2
 =\|b_{\phi,k}\|_2^2+\frac{v_{\phi,k}}{N}.
 \label{eq:clipping-risk-decomposition}
\end{equation}

\begin{remark}[Per-token coefficient truncation and PPO clipping]
\label{rem:token-truncation-vs-clipping}
Write $r_{k,t}$ for the token likelihood ratio and $s_{k,t}:=\nabla_\theta\log P_{\theta_k}(Y_t\mid X,Y_{<t})$ for the token score. A detached per-token coefficient truncation with a bounded function $u$ uses the response contribution $A_k\sum_t u(r_{k,t})s_{k,t}$. For example, an upper truncation uses $u(r)=\min\{r,c\}$.

PPO clipping defines a different estimator. Away from the clipping boundary, its response contribution is $A_k\sum_t r_{k,t}M_\epsilon(r_{k,t},A_k)s_{k,t}$, where $M_\epsilon(r,A)=\mathbf 1\{A\ge0,\ r\le1+\epsilon\}+\mathbf 1\{A<0,\ r\ge1-\epsilon\}$. Thus coefficient truncation caps the magnitude of a token contribution while generally retaining a nonzero gradient, whereas PPO clipping removes the gradient in an advantage-dependent region. Neither operation is the same as truncating the complete-response weight $W_k=\prod_t r_{k,t}$. They define different choices of $\phi_k$ and therefore different biases and variances.
\end{remark}

For any estimator with total risk $m$, the fixed-step smoothness argument gives the expected-improvement lower bound $\eta_k(\|g_k\|_2^2-m)/2$ for every $0<\eta_k\le1/L_k$. Hence a modified estimator has a stronger lower bound exactly when its total risk is smaller. Substituting Equation~\eqref{eq:clipping-risk-decomposition} reduces this comparison to one condition.

\begin{proposition}[When clipping lowers gradient risk]
\label{prop:clipping-crossover}
For a common sample size $N$, the modified estimator $\widehat g_{\phi,k}$ has lower total gradient risk than $\widehat g_k$ if and only if
\begin{equation}
 v_k-v_{\phi,k}
 >N\|b_{\phi,k}\|_2^2.
 \label{eq:crossover}
\end{equation}
\end{proposition}

Proposition~\ref{prop:clipping-crossover} says that variance reduction alone is insufficient. It must exceed the squared bias introduced by the modification. Lower risk is still weaker than recovery because both estimators may have errors larger than the population-gradient signal. This observation leads to the narrower recovery condition.

\begin{corollary}[When clipping recovers the certificate]
\label{cor:clipping-needed}
For a common step size $0<\eta_k\le1/L_k$, the modified estimator has a positive expected-improvement certificate while the unmodified estimator does not exactly when
\begin{equation}
 m_{\phi,k}<\|g_k\|_2^2\le m_k.
 \label{eq:clipping-needed}
\end{equation}
\end{corollary}

\begin{remark}[Why high normalized ESS favors the unmodified estimator]
\label{rem:high-ess-clipping}
Equation~\eqref{eq:rho-gradient-mse-bound} shows that the risk bound of the unmodified estimator decreases as $\rho_k$ increases. In the high-ESS regime, there is therefore less sampling variance available for a clipping rule to remove, while any clipping bias remains. A high value of $\rho_k$ does not prove that every unmodified update is optimal, but it weakens the statistical case for clipping. Conversely, a low value of $\rho_k$ only identifies a vulnerable regime. A particular clipping rule is justified only if it satisfies Proposition~\ref{prop:clipping-crossover} and, when recovery is required, Corollary~\ref{cor:clipping-needed}.
\end{remark}

The theoretical quantity $\rho_k$ is a population object. During training, we use the standard normalized plug-in estimate
\begin{equation}
 \widehat\rho_k
 :=\frac{(\sum_{i=1}^N W_{k,i})^2}{N\sum_{i=1}^N W_{k,i}^2}.
 \label{eq:sample-ess}
\end{equation}
It is computed from the complete-response ratios before any modification is applied. A finite batch can miss rare, large ratios and can therefore overstate population support, so $\widehat\rho_k$ is an observable diagnostic rather than an exact replacement for $\rho_k$ in the theorem. The numerical value $0.1$ used below is a prespecified operational threshold, not a universal constant derived from the theory.'''

PROOFS = r'''\section{Proofs for the theoretical results}
\label{app:proofs}

At update $k$, all expectations in this appendix are taken after conditioning on the preceding updates. The current policy is therefore fixed and the unused sampling units are independent draws from $Q$. We suppress this conditioning throughout.

\subsection{Policy-gradient identity and mean-square bound}

By definition, $f_k=(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)$ and $W_k=dP_{\theta_k}/dQ$. Change of measure gives
\begin{align}
 \E_Q[W_kf_k]
 &=\E_{P_{\theta_k}}[(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)]
 \nonumber\\
 &=\E_{P_{\theta_k}}[R\nabla_\theta\log P_{\theta_k}(Y\mid X)]
 \nonumber\\
 &\quad-
 \E_{P_{\theta_k}}[b(X)\nabla_\theta\log P_{\theta_k}(Y\mid X)].
 \label{eq:proof-baseline-split}
\end{align}
For the baseline term, condition further on $X$:
\begin{align}
 &\E_{P_{\theta_k}}[b(X)\nabla_\theta\log P_{\theta_k}(Y\mid X)\mid X]
 \nonumber\\
 &\qquad=b(X)\sum_yP_{\theta_k}(y\mid X)\nabla_\theta\log P_{\theta_k}(y\mid X)
 \nonumber\\
 &\qquad=b(X)\sum_y\nabla_\theta P_{\theta_k}(y\mid X)
 \nonumber\\
 &\qquad=b(X)\nabla_\theta1
 =0.
 \label{eq:proof-baseline-zero}
\end{align}
The remaining term is the score-function identity. Hence
\begin{equation}
 \E_Q[W_kf_k]=\nabla J(\theta_k)=g_k.
 \label{eq:conditional-gradient-identity}
\end{equation}
Averaging $N$ independent copies gives $\E[\widehat g_k]=g_k$.

For the mean-square calculation, define $\xi_{k,i}:=W_{k,i}f_{k,i}-g_k$. Equation~\eqref{eq:conditional-gradient-identity} implies $\E[\xi_{k,i}]=0$. Since $\widehat g_k-g_k=N^{-1}\sum_{i=1}^N\xi_{k,i}$,
\begin{align}
 \E\|\widehat g_k-g_k\|_2^2
 &=\frac1{N^2}\sum_{i=1}^N\E\|\xi_{k,i}\|_2^2
 +\frac1{N^2}\sum_{i\ne j}\E[\xi_{k,i}^\top\xi_{k,j}].
 \label{eq:mse-expand-sums}
\end{align}
For $i\ne j$, independence and zero means give
\begin{equation}
 \E[\xi_{k,i}^\top\xi_{k,j}]
 =\E[\xi_{k,i}]^\top\E[\xi_{k,j}]
 =0.
 \label{eq:cross-terms-zero}
\end{equation}
The diagonal terms are identical, so
\begin{equation}
 \E\|\widehat g_k-g_k\|_2^2
 =\frac1N\E_Q\|W_kf_k-g_k\|_2^2.
 \label{eq:mse-single-unit}
\end{equation}
Expanding the remaining squared norm yields
\begin{align}
 \E_Q\|W_kf_k-g_k\|_2^2
 &=\E_Q[W_k^2\|f_k\|_2^2]
 -2g_k^\top\E_Q[W_kf_k]
 +\|g_k\|_2^2
 \nonumber\\
 &=\E_Q[W_k^2\|f_k\|_2^2]-\|g_k\|_2^2,
 \label{eq:mse-expand-unit}
\end{align}
where the last equality uses Equation~\eqref{eq:conditional-gradient-identity}. Combining Equations~\eqref{eq:mse-single-unit} and \eqref{eq:mse-expand-unit} gives
\begin{equation}
 \E\|\widehat g_k-g_k\|_2^2
 =\frac1N\left\{\E_Q[W_k^2\|f_k\|_2^2]-\|g_k\|_2^2\right\}.
 \label{eq:rho-gradient-mse}
\end{equation}
Finally, $\|f_k(z)\|_2\le\|f_k\|_\infty$ and $\E_Q[W_k^2]=\rho_k^{-1}$ imply
\begin{align}
 \E\|\widehat g_k-g_k\|_2^2
 &\le\frac1N\E_Q[W_k^2\|f_k\|_2^2]
 \nonumber\\
 &\le\frac{\|f_k\|_\infty^2}{N}\E_Q[W_k^2]
 \nonumber\\
 &=\frac{\|f_k\|_\infty^2}{N\rho_k},
 \label{eq:mse-rho-bound-proof}
\end{align}
which proves Equation~\eqref{eq:rho-gradient-mse-bound}.

\subsection{Proof of Lemma~\ref{lem:rho-gradient-reliability}}

Apply Markov's inequality to the nonnegative random variable $\|\widehat g_k-g_k\|_2^2$:
\begin{align}
 &\Pr\left(
 \|\widehat g_k-g_k\|_2
 >\frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
 \right)
 \nonumber\\
 &\quad=
 \Pr\left(
 \|\widehat g_k-g_k\|_2^2
 >\frac{\|f_k\|_\infty^2}{\delta N\rho_k}
 \right)
 \nonumber\\
 &\quad\le
 \frac{\E\|\widehat g_k-g_k\|_2^2}{\|f_k\|_\infty^2/(\delta N\rho_k)}
 \nonumber\\
 &\quad\le\delta,
 \label{eq:markov-rho-proof}
\end{align}
where the last line uses Equation~\eqref{eq:rho-gradient-mse-bound}. Taking complements proves Equation~\eqref{eq:rho-gradient-concentration}.

\subsection{Proof of Lemma~\ref{lem:gradient-error-progress}}

Let $\|\widehat g-g\|_2\le\varepsilon$. Smoothness gives, for every $\alpha\ge0$,
\begin{equation}
 J(\theta+\alpha\widehat g)-J(\theta)
 \ge
 \alpha g^\top\widehat g
 -\frac{L\alpha^2}{2}\|\widehat g\|_2^2.
 \label{eq:smoothness-start}
\end{equation}
The inner product satisfies
\begin{align}
 g^\top\widehat g
 &=\|\widehat g\|_2^2+(g-\widehat g)^\top\widehat g
 \nonumber\\
 &\ge\|\widehat g\|_2^2-\|g-\widehat g\|_2\|\widehat g\|_2
 \nonumber\\
 &\ge\|\widehat g\|_2(\|\widehat g\|_2-\varepsilon).
 \label{eq:reliable-inner-product}
\end{align}
Substituting Equation~\eqref{eq:reliable-inner-product} into Equation~\eqref{eq:smoothness-start} gives
\begin{align}
 J(\theta+\alpha\widehat g)-J(\theta)
 &\ge
 \alpha\|\widehat g\|_2(\|\widehat g\|_2-\varepsilon)
 -\frac{L\alpha^2}{2}\|\widehat g\|_2^2.
 \label{eq:reliable-quadratic}
\end{align}
For $\widehat g\ne0$, the derivative of the right-hand side is
\begin{equation}
 \|\widehat g\|_2(\|\widehat g\|_2-\varepsilon)
 -L\alpha\|\widehat g\|_2^2.
 \label{eq:reliable-quadratic-derivative}
\end{equation}
Its nonnegative maximizer is $\alpha=L^{-1}(1-\varepsilon/\|\widehat g\|_2)_+$. When $\widehat g=0$, both the chosen step and the claimed lower bound are zero. Substituting the maximizer into Equation~\eqref{eq:reliable-quadratic} proves Equation~\eqref{eq:gradient-error-progress}.

\subsection{Proof of Theorem~\ref{thm:permissive-reliability}}

Lemma~\ref{lem:rho-gradient-reliability} shows that the event
\begin{equation}
 \|\widehat g_k-g_k\|_2
 \le
 \frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
 \label{eq:theorem-reliability-event}
\end{equation}
has probability at least $1-\delta$. On this event, apply Lemma~\ref{lem:gradient-error-progress} with $\theta=\theta_k$, $\widehat g=\widehat g_k$, $\varepsilon=\|f_k\|_\infty/\sqrt{\delta N\rho_k}$, and $L=L_k$. The resulting inequality is exactly Equation~\eqref{eq:permissive-reliability}.

\subsection{Risk comparison for a general modification}

Let $\phi_k:\mathcal Z\to\mathbb R^d$ be integrable and define $\widehat g_{\phi,k}=N^{-1}\sum_i\phi_k(Z_i)$. Add and subtract its expectation:
\begin{equation}
 \widehat g_{\phi,k}-g_k
 =\{\widehat g_{\phi,k}-\E_Q[\phi_k]\}+b_{\phi,k}.
 \label{eq:modified-bias-split}
\end{equation}
Squaring and taking expectation gives
\begin{align}
 m_{\phi,k}
 &=\E\|\widehat g_{\phi,k}-\E_Q[\phi_k]\|_2^2
 +\|b_{\phi,k}\|_2^2
 \nonumber\\
 &\quad+2b_{\phi,k}^\top\E[\widehat g_{\phi,k}-\E_Q[\phi_k]]
 \nonumber\\
 &=\frac{v_{\phi,k}}{N}+\|b_{\phi,k}\|_2^2,
 \label{eq:modified-risk-proof}
\end{align}
where the cross term is zero and independence gives the variance factor $1/N$. For the unmodified estimator, Equation~\eqref{eq:conditional-gradient-identity} gives zero bias, so $m_k=v_k/N$. Therefore
\begin{align}
 m_{\phi,k}<m_k
 &\iff
 \|b_{\phi,k}\|_2^2+\frac{v_{\phi,k}}{N}<\frac{v_k}{N}
 \nonumber\\
 &\iff
 v_k-v_{\phi,k}>N\|b_{\phi,k}\|_2^2,
 \label{eq:crossover-proof}
\end{align}
which proves Proposition~\ref{prop:clipping-crossover}.

It remains to connect total risk to the expected policy-improvement certificate used in Corollary~\ref{cor:clipping-needed}. For any random estimate $\widehat g$ of $g_k$ and any $0<\eta_k\le1/L_k$, smoothness gives
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
Substituting and using $\eta_k\le1/L_k$ yields
\begin{align}
 J(\theta_k+\eta_k\widehat g)-J(\theta_k)
 &\ge
 \frac{\eta_k}{2}\{\|g_k\|_2^2-\|\widehat g-g_k\|_2^2\}
 \nonumber\\
 &\quad+\frac{\eta_k}{2}(1-L_k\eta_k)\|\widehat g\|_2^2
 \nonumber\\
 &\ge
 \frac{\eta_k}{2}\{\|g_k\|_2^2-\|\widehat g-g_k\|_2^2\}.
 \label{eq:fixed-step-polarized}
\end{align}
Taking expectation gives a positive lower bound exactly when the estimator risk is smaller than $\|g_k\|_2^2$. Applying this statement to $\widehat g_{\phi,k}$ and $\widehat g_k$ proves Corollary~\ref{cor:clipping-needed}.

\subsection{Intact prompt groups}

Suppose $M$ prompts are independent and each prompt has $G$ conditionally independent responses. Keep each prompt group intact and define its contribution by $\Xi_{k,i}:=G^{-1}\sum_{j=1}^G W_{k,ij}f_{k,ij}$. A leave-one-out baseline is admissible because it excludes the focal response. Conditional on the prompt and the remaining responses, the focal score still has mean zero. Hence $\E[\Xi_{k,i}]=g_k$.

Jensen's inequality gives
\begin{align}
 \|\Xi_{k,i}\|_2^2
 &=\left\|\frac1G\sum_{j=1}^G W_{k,ij}f_{k,ij}\right\|_2^2
 \nonumber\\
 &\le\frac1G\sum_{j=1}^G W_{k,ij}^2\|f_{k,ij}\|_2^2.
 \label{eq:group-jensen}
\end{align}
Repeating the sample-mean calculation across the $M$ independent prompt groups gives
\begin{align}
 \E\left\|\frac1M\sum_{i=1}^M\Xi_{k,i}-g_k\right\|_2^2
 &\le\frac1M\E\|\Xi_{k,1}\|_2^2
 \nonumber\\
 &\le\frac{\|f_k\|_\infty^2}{M\rho_k}.
 \label{eq:group-rho-bound}
\end{align}
Markov's inequality then gives the same form as Lemma~\ref{lem:rho-gradient-reliability} with the number of independent prompts $M$ in place of the number of responses $N$.

\subsection{Finite-moment relaxation}

The exact equality in Equation~\eqref{eq:rho-gradient-mse} requires only $\E_Q[W_k^2\|f_k\|_2^2]<\infty$. The bounded-contribution condition is used solely to obtain the standard form $\|f_k\|_\infty^2/(N\rho_k)$. More generally, define $M_{2,k}:=\E_Q[W_k^2\|f_k\|_2^2]/\E_Q[W_k^2]$. The same calculation gives $\E\|\widehat g_k-g_k\|_2^2\le M_{2,k}/(N\rho_k)$. We use the bounded form in the main text to match conventional importance-sampling notation and make the role of normalized ESS explicit.'''


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def normalize_spacing(text: str) -> str:
    text = text.replace("\u2014", ",")
    text = text.replace("---", "--")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", line)
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def validate(text: str) -> None:
    forbidden_feedback = (
        "[What is Renyi",
        "[The special relation",
        "[The Ess has",
        "[Z_k, superscript",
        "[Add remark",
        "[...",
    )
    for token in forbidden_feedback:
        if token in text:
            raise RuntimeError(f"Author feedback marker survived: {token}")
    for token in (r"\widetilde g", r"Z_k^{\mathrm P}", r"Z_k^{\mathrm C}"):
        if token in text:
            raise RuntimeError(f"Obsolete notation survived: {token}")
    if "\u2014" in text or "---" in text:
        raise RuntimeError("Em dash survived")
    if re.search(r"(?<=\S) {2,}(?=\S)", text):
        raise RuntimeError("Double word spacing survived")
    theory = text.split(r"\section{Theoretical analysis}", 1)[1].split(
        r"\section{Contextual-bandit validation of the framework}", 1
    )[0]
    if r"\mathcal F" in theory:
        raise RuntimeError("History sigma-field notation survived in main theory")
    lemma_one = theory.split(
        r"\begin{lemma}[Normalized ESS controls the gradient-error radius]", 1
    )[1].split(r"\end{lemma}", 1)[0]
    if lemma_one.count(r"\begin{equation}") != 1:
        raise RuntimeError("First lemma must contain one displayed conclusion")
    if "with probability at least" not in lemma_one:
        raise RuntimeError("First lemma is not stated in theorem style")
    lemma_two = theory.split(
        r"\begin{lemma}[A reliable gradient estimate yields policy improvement]", 1
    )[1].split(r"\end{lemma}", 1)[0]
    if lemma_two.count(r"\begin{equation}") != 1:
        raise RuntimeError("Second lemma must contain one displayed conclusion")
    theorem = theory.split(
        r"\begin{theorem}[When a permissive update remains reliable]", 1
    )[1].split(r"\end{theorem}", 1)[0]
    if theorem.count(r"\begin{equation}") != 1:
        raise RuntimeError("Main theorem must contain one displayed conclusion")
    for required in (
        r"\widehat g_{\phi,k}",
        r"rem:token-truncation-vs-clipping",
        r"rem:high-ess-clipping",
        r"eq:sample-ess",
    ):
        if required not in theory:
            raise RuntimeError(f"Required clipping rewrite missing: {required}")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    text = replace_block(
        text,
        r"\section{Preliminaries}",
        r"\section{Theoretical analysis}",
        PRELIMINARIES,
    )
    text = replace_block(
        text,
        r"\section{Theoretical analysis}",
        r"\section{Contextual-bandit validation of the framework}",
        THEORY,
    )
    text = replace_block(
        text,
        r"\section{Proofs for the theoretical results}",
        r"\bibliographystyle{plainnat}",
        PROOFS,
    )
    text = normalize_spacing(text)
    validate(text)
    PATH.write_text(text, encoding="utf-8")
    print("Rewrote preliminaries, theory, clipping comparison, and proofs.")


if __name__ == "__main__":
    main()
