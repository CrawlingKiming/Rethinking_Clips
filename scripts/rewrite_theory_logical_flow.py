from pathlib import Path


THEORY = r'''\section{Theoretical analysis}
\label{sec:theory}

Figure~\ref{fig:llm-motivation} raises a specific question: when does an
unused portion of a fixed rollout stop supporting a reliable update for the
current policy?  Answering this question requires two links.  We first connect
normalized sequence ESS to the error of the finite-batch policy-gradient
estimate.  We then connect that error to the population improvement produced by
the next update.

At update $k$, condition on the preceding history and write
$W_k:=W_{\theta_k}$, $f_k:=f_{\theta_k}$,
$g_k:=g(\theta_k)$, $\widehat g_k:=\widehat g_N(\theta_k)$, and
$\rho_k:=\rho_{\theta_k}$.  Throughout this section, suppose that the current
gradient contribution is bounded,
$\|f_k\|_\infty:=\sup_z\|f_k(z)\|_2<\infty$.  This condition fixes the scale of
one response contribution, while $\rho_k$ captures the additional loss of
reliability caused by policy mismatch.

\subsection{From normalized ESS to a gradient-error radius}
\label{sec:raw}

Because the minibatch used at update $k$ has not appeared in any previous
update, its $N$ units are independent draws from $Q$ after conditioning on the
history.  Change of measure therefore gives
\begin{equation}
  \E[\widehat g_k\mid\mathcal F_{k-1}]=g_k.
  \label{eq:gradient-unbiasedness}
\end{equation}
Independence then reduces the squared error of the sample average to the
second moment of one centered contribution:
\begin{equation}
  \E\!\left[
    \|\widehat g_k-g_k\|_2^2
    \mid\mathcal F_{k-1}
  \right]
  =\frac1N
  \left\{
    \E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]
    -\|g_k\|_2^2
  \right\}.
  \label{eq:rho-gradient-mse}
\end{equation}
The bounded contribution condition and the definition
$\rho_k^{-1}=\E_Q[W_k^2\mid\mathcal F_{k-1}]$ now give
\begin{equation}
  \E\!\left[
    \|\widehat g_k-g_k\|_2^2
    \mid\mathcal F_{k-1}
  \right]
  \le
  \frac{\|f_k\|_\infty^2}{N\rho_k}.
  \label{eq:rho-gradient-mse-bound}
\end{equation}
Equation~\eqref{eq:rho-gradient-mse-bound} identifies the average scale of the
estimation error.  The actual update, however, is computed from one realized
minibatch, so we need an event-level statement.  Applying Markov's inequality
to the squared error yields the following conclusion.

\begin{lemma}[Normalized ESS controls the gradient-error radius]
\label{lem:rho-gradient-reliability}
For every $\delta\in(0,1)$,
\begin{equation}
  \Pr\!\left(
    \|\widehat g_k-g_k\|_2
    \le
    \frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
    \,\middle|\,
    \mathcal F_{k-1}
  \right)
  \ge 1-\delta.
  \label{eq:rho-gradient-concentration}
\end{equation}
\end{lemma}

Lemma~\ref{lem:rho-gradient-reliability} turns normalized ESS into a directly
interpretable uncertainty radius.  For fixed $N$ and $\|f_k\|_\infty$, the
radius grows as $\rho_k^{-1/2}$; equivalently, the nominal batch size is reduced
to the effective sequence count $N\rho_k$.

\subsection{From gradient accuracy to policy improvement}
\label{sec:mse-improvement}

The previous lemma tells us when the sampled direction is close to the true
gradient.  To determine whether that accuracy is sufficient, we must translate
a gradient-error radius into a change in the population objective.  Smoothness
provides this second link: an estimated direction that lies within radius
$\varepsilon$ of the true gradient must retain a controlled component in an
ascent direction.  Optimizing the resulting quadratic lower bound gives the
next lemma.

\begin{lemma}[A reliable direction yields policy improvement]
\label{lem:gradient-error-progress}
Let $g=\nabla J(\theta)$ and suppose that
$\|\widetilde g-g\|_2\le\varepsilon$.  If $J$ is $L$-smooth along the update
segment, define
$\alpha=L^{-1}(1-\varepsilon/\|\widetilde g\|_2)_+$, with $\alpha=0$ when
$\widetilde g=0$.  Then
\begin{equation}
  J(\theta+\alpha\widetilde g)-J(\theta)
  \ge
  \frac{1}{2L}
  \left(\|\widetilde g\|_2-\varepsilon\right)_+^2.
  \label{eq:gradient-error-progress}
\end{equation}
\end{lemma}

The lower bound increases as the error radius decreases.  Thus estimator
reliability is not merely a diagnostic quantity: it determines both whether a
nonzero step can be certified and how large that certificate can be.  This is
the same smoothness mechanism used in safe policy-gradient analyses
\citep{pirotta2013adaptive,papini2022smoothing}.

\subsection{When does a permissive update remain reliable?}

Lemma~\ref{lem:rho-gradient-reliability} supplies an ESS-dependent error radius
for the current minibatch.  Lemma~\ref{lem:gradient-error-progress} converts any
such radius into population improvement.  Substituting the first conclusion
into the second yields the central result.

\begin{theorem}[When a permissive update remains reliable]
\label{thm:permissive-reliability}
For $\delta\in(0,1)$, let
$r_k(\delta):=\|f_k\|_\infty/\sqrt{\delta N\rho_k}$ and define
$\alpha_k:=L_k^{-1}(1-r_k(\delta)/\|\widehat g_k\|_2)_+$, with
$\alpha_k=0$ when $\widehat g_k=0$.  If $J$ is $L_k$-smooth along the update
segment, then, conditional on $\mathcal F_{k-1}$, with probability at least
$1-\delta$,
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

The certificate is positive exactly when the observed gradient norm exceeds its
ESS-dependent uncertainty radius.  This comparison explains why failure can be
delayed: the same rollout can support many useful updates while $N\rho_k$
remains large, and only becomes statistically inconclusive after effective
support has sufficiently deteriorated.

\begin{remark}[Interpretation]
\label{rem:rho-interpretation}
Theorem~\ref{thm:permissive-reliability} identifies a loss of certification,
not a deterministic collapse point.  A zero lower bound means that the rollout
no longer certifies the sampled direction at the chosen confidence level; the
realized update may still improve.

The criterion complements policy-deviation guarantees.  Those guarantees study
population-level consequences of changing the policy, whereas the present
result studies whether a fixed finite rollout still estimates the current
policy gradient accurately enough.  Both effects may matter, but they answer
different questions.

For fixed $N$, contribution scale, confidence level, and observed gradient
signal, all mismatch-dependent degradation enters through $\rho_k$.  No
universal numerical threshold follows, because the required value also depends
on those other quantities.
\end{remark}

The theorem diagnoses when the permissive estimator loses a useful guarantee,
but it does not prescribe what should replace it.  That distinction leads to a
separate question: can a more conservative estimator recover reliability after
the permissive certificate disappears?

\subsection{When can clipping recover reliability?}
\label{sec:clipping-recovery}

Let $Z_k^{\mathrm P}:=W_k f_k$ be the per-unit permissive contribution and let
$Z_k^{\mathrm C}$ be the contribution produced by a chosen clipping rule.  For
$e\in\{\mathrm P,\mathrm C\}$, define
$\widehat g_k^e:=N^{-1}\sum_{i=1}^N Z_{k,i}^e$ and its total conditional risk
$m_k^e:=\E[\|\widehat g_k^e-g_k\|_2^2\mid\mathcal F_{k-1}]$.  Write
$b_k^{\mathrm C}:=\E[Z_k^{\mathrm C}\mid\mathcal F_{k-1}]-g_k$ and
$v_k^e:=\E[\|Z_k^e-\E Z_k^e\|_2^2\mid\mathcal F_{k-1}]$.  Independence gives
\begin{equation}
  m_k^{\mathrm P}=\frac{v_k^{\mathrm P}}{N},
  \qquad
  m_k^{\mathrm C}=\|b_k^{\mathrm C}\|_2^2+rac{v_k^{\mathrm C}}{N}.
  \label{eq:clipping-risk-decomposition}
\end{equation}

To compare these risks with policy improvement, take expectation in the
polarized smoothness bound for a common step size $0<\eta_k\le1/L_k$.  For
either estimator does this require unbiasedness, and it gives
\begin{equation}
  \E\!\left[
    J(\theta_k+\eta_k\widehat g_k^e)-J(\theta_k)
    \mid\mathcal F_{k-1}
  \right]
  \ge
  \frac{\eta_k}{2}
  \left(\|g_k\|_2^2-m_k^e\right).
  \label{eq:expected-risk-progress}
\end{equation}
The lower bound becomes stronger exactly when total estimation risk decreases.
Substituting Equation~\eqref{eq:clipping-risk-decomposition} therefore reduces
the comparison to a single bias--variance condition.

\begin{proposition}[Bias--variance condition for clipping]
\label{prop:clipping-crossover}
The clipped estimator has lower total gradient risk than the permissive
estimator if and only if
\begin{equation}
  v_k^{\mathrm P}-v_k^{\mathrm C}
  >N\|b_k^{\mathrm C}\|_2^2.
  \label{eq:crossover}
\end{equation}
\end{proposition}

Proposition~\ref{prop:clipping-crossover} says that variance reduction alone is
not enough: it must exceed the squared bias introduced by the clipping rule.  A
lower risk is also weaker than recovery, because both estimators may still have
errors larger than the underlying gradient signal.  Clipping is needed in the
narrower regime where only the clipped estimator retains a positive lower
bound.

\begin{corollary}[When clipping recovers the certificate]
\label{cor:clipping-needed}
Under Equation~\eqref{eq:expected-risk-progress}, the clipped estimator has a
positive expected-improvement certificate while the permissive estimator does
not exactly when
\begin{equation}
  m_k^{\mathrm C}<\|g_k\|_2^2\le m_k^{\mathrm P}.
  \label{eq:clipping-needed}
\end{equation}
\end{corollary}

A small $\rho_k$ therefore explains why the permissive estimator becomes
vulnerable, but it does not justify every clipping rule.  Recovery occurs only
when the chosen rule lowers total error enough to cross the gradient-signal
level.  If neither estimator crosses that level, the framework points to a new
rollout rather than stronger modification of the same data.

The comparison is estimator-agnostic.  Full-sequence truncation, token-level
PPO/GRPO masking, and detached coefficient capping define different
$Z_k^{\mathrm C}$ and hence different biases and variances.  The experiments
ask whether normalized ESS identifies the regime in which each concrete
safeguard becomes useful.

\subsection{Observable reliability diagnostic}
\label{sec:practice}

Population $\rho_k$ is unavailable during training.  We use the standard
normalized plug-in estimate
\begin{equation}
  \widehat\rho_k
  :=\frac{(\sum_{i=1}^N W_{k,i})^2}
          {N\sum_{i=1}^N W_{k,i}^2}.
  \label{eq:sample-ess}
\end{equation}
It is computed from complete-response ratios before the update.  A finite batch
can miss rare large-ratio responses and therefore overstate population support,
so $\widehat\rho_k$ is an observable diagnostic rather than an exact substitute
inside Theorem~\ref{thm:permissive-reliability}.  The numerical value $0.1$ used
in the experiments is prespecified as a conservative operational event; it is
not a constant derived from the theorem.

'''


PROOFS = r'''\section{Proofs for the theoretical results}
\label{app:proofs}

This appendix supplies the calculations that lead to the single conclusion in
each main-text result.  All statements are read conditional on the history
before update $k$, so the current policy is fixed and the unused sampling units
are independent draws from $Q$.

\subsection{Policy-gradient identity and mean-square calculation}

By definition,
$f_k=(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)$ and
$W_k=dP_{\theta_k}/dQ$.  Change of measure gives
\begin{align}
  \E_Q[W_k f_k\mid\mathcal F_{k-1}]
  &=\E_{P_{\theta_k}}
    [(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)
      \mid\mathcal F_{k-1}]
  \nonumber\\
  &=\E_{P_{\theta_k}}
    [R\nabla_\theta\log P_{\theta_k}(Y\mid X)
      \mid\mathcal F_{k-1}]
  \nonumber\\
  &\quad-
    \E_{P_{\theta_k}}
    [b(X)\nabla_\theta\log P_{\theta_k}(Y\mid X)
      \mid\mathcal F_{k-1}].
  \label{eq:proof-baseline-split}
\end{align}
For the baseline term, condition further on $X$:
\begin{align}
  &\E_{P_{\theta_k}}
    [b(X)\nabla_\theta\log P_{\theta_k}(Y\mid X)
      \mid X,\mathcal F_{k-1}]
  \nonumber\\
  &\qquad=
  b(X)\sum_y P_{\theta_k}(y\mid X)
  \nabla_\theta\log P_{\theta_k}(y\mid X)
  \nonumber\\
  &\qquad=
  b(X)\sum_y\nabla_\theta P_{\theta_k}(y\mid X)
  \nonumber\\
  &\qquad=
  b(X)\nabla_\theta 1
  =0.
  \label{eq:proof-baseline-zero}
\end{align}
The remaining term is the score-function identity, and hence
\begin{equation}
  \E_Q[W_k f_k\mid\mathcal F_{k-1}]
  =\nabla J(\theta_k)
  =g_k.
  \label{eq:conditional-gradient-identity}
\end{equation}
Averaging $N$ identical units proves
Equation~\eqref{eq:gradient-unbiasedness}.

For the mean-square calculation, define
$\xi_{k,i}:=W_{k,i}f_{k,i}-g_k$.  Equation~\eqref{eq:conditional-gradient-identity}
implies
\begin{equation}
  \E[\xi_{k,i}\mid\mathcal F_{k-1}]=0.
  \label{eq:centered-unit}
\end{equation}
Using $\widehat g_k-g_k=N^{-1}\sum_{i=1}^N\xi_{k,i}$ and expanding the squared
norm gives
\begin{align}
  &\E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\quad=
  \frac1{N^2}\sum_{i=1}^N
  \E[\|\xi_{k,i}\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\qquad+
  \frac1{N^2}\sum_{i\ne j}
  \E[\xi_{k,i}^\top\xi_{k,j}\mid\mathcal F_{k-1}].
  \label{eq:mse-expand-sums}
\end{align}
For $i\ne j$, conditional independence and
Equation~\eqref{eq:centered-unit} imply
\begin{align}
  \E[\xi_{k,i}^\top\xi_{k,j}\mid\mathcal F_{k-1}]
  &=\E[\xi_{k,i}\mid\mathcal F_{k-1}]^\top
    \E[\xi_{k,j}\mid\mathcal F_{k-1}]
  \nonumber\\
  &=0.
  \label{eq:cross-terms-zero}
\end{align}
The remaining diagonal terms are identical, so
\begin{equation}
  \E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  =\frac1N
  \E_Q[\|W_kf_k-g_k\|_2^2\mid\mathcal F_{k-1}].
  \label{eq:mse-single-unit}
\end{equation}
Expanding the last squared norm yields
\begin{align}
  &\E_Q[\|W_kf_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\quad=
  \E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]
  -2g_k^\top\E_Q[W_kf_k\mid\mathcal F_{k-1}]
  +\|g_k\|_2^2
  \nonumber\\
  &\quad=
  \E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]
  -\|g_k\|_2^2,
  \label{eq:mse-expand-unit}
\end{align}
where the last equality uses
Equation~\eqref{eq:conditional-gradient-identity}.  Combining
Equations~\eqref{eq:mse-single-unit} and \eqref{eq:mse-expand-unit} proves
Equation~\eqref{eq:rho-gradient-mse}.

Finally, $\|f_k(z)\|_2\le\|f_k\|_\infty$ for every $z$, so
\begin{align}
  \E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  &\le\frac1N
    \E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\le\frac{\|f_k\|_\infty^2}{N}
    \E_Q[W_k^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &=\frac{\|f_k\|_\infty^2}{N\rho_k}.
  \label{eq:mse-rho-bound-proof}
\end{align}
This proves Equation~\eqref{eq:rho-gradient-mse-bound}.

\subsection{Proof of Lemma~\ref{lem:rho-gradient-reliability}}

Apply Markov's inequality to the nonnegative random variable
$\|\widehat g_k-g_k\|_2^2$:
\begin{align}
  &\Pr\!\left(
    \|\widehat g_k-g_k\|_2
    >\frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
    \,\middle|\mathcal F_{k-1}
  \right)
  \nonumber\\
  &\quad=
  \Pr\!\left(
    \|\widehat g_k-g_k\|_2^2
    >\frac{\|f_k\|_\infty^2}{\delta N\rho_k}
    \,\middle|\mathcal F_{k-1}
  \right)
  \nonumber\\
  &\quad\le
  \frac{
    \E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  }{
    \|f_k\|_\infty^2/(\delta N\rho_k)
  }
  \nonumber\\
  &\quad\le\delta,
  \label{eq:markov-rho-proof}
\end{align}
where the final line uses
Equation~\eqref{eq:rho-gradient-mse-bound}.  Taking complements proves
Equation~\eqref{eq:rho-gradient-concentration}.

\subsection{Proof of Lemma~\ref{lem:gradient-error-progress}}

Let $\|\widetilde g-g\|_2\le\varepsilon$.  Smoothness gives, for every
$\alpha\ge0$,
\begin{equation}
  J(\theta+\alpha\widetilde g)-J(\theta)
  \ge
  \alpha g^\top\widetilde g
  -\frac{L\alpha^2}{2}\|\widetilde g\|_2^2.
  \label{eq:smoothness-start}
\end{equation}
The inner product is bounded from below as follows:
\begin{align}
  g^\top\widetilde g
  &=\|\widetilde g\|_2^2
    +(g-\widetilde g)^\top\widetilde g
  \nonumber\\
  &\ge\|\widetilde g\|_2^2
    -\|g-\widetilde g\|_2\|\widetilde g\|_2
  \nonumber\\
  &\ge\|\widetilde g\|_2
    (\|\widetilde g\|_2-\varepsilon).
  \label{eq:reliable-inner-product}
\end{align}
Substituting Equation~\eqref{eq:reliable-inner-product} into
Equation~\eqref{eq:smoothness-start} gives
\begin{align}
  J(\theta+\alpha\widetilde g)-J(\theta)
  &\ge
  \alpha\|\widetilde g\|_2
  (\|\widetilde g\|_2-\varepsilon)
  \nonumber\\
  &\quad-
  \frac{L\alpha^2}{2}\|\widetilde g\|_2^2.
  \label{eq:reliable-quadratic}
\end{align}
For $\widetilde g\ne0$, the derivative of the right-hand side is
\begin{equation}
  \|\widetilde g\|_2
  (\|\widetilde g\|_2-\varepsilon)
  -L\alpha\|\widetilde g\|_2^2.
  \label{eq:reliable-quadratic-derivative}
\end{equation}
Its nonnegative maximizer is
$\alpha=L^{-1}(1-\varepsilon/\|\widetilde g\|_2)_+$; when
$\widetilde g=0$, both the chosen step and the claimed lower bound are zero.
Substituting the maximizer into Equation~\eqref{eq:reliable-quadratic} gives
Equation~\eqref{eq:gradient-error-progress}.

\subsection{Proof of Theorem~\ref{thm:permissive-reliability}}

Lemma~\ref{lem:rho-gradient-reliability} shows that the event
$\|\widehat g_k-g_k\|_2\le r_k(\delta)$ has conditional probability at least
$1-\delta$.  On that event, apply
Lemma~\ref{lem:gradient-error-progress} with
$\theta=\theta_k$, $\widetilde g=\widehat g_k$,
$\varepsilon=r_k(\delta)$, and $L=L_k$.  The resulting inequality is exactly
Equation~\eqref{eq:permissive-reliability}.

\subsection{Expected-risk form used for estimator comparison}

For any direction $\widetilde g$ and any $0<\eta\le1/L$, smoothness gives
\begin{equation}
  J(\theta+\eta\widetilde g)-J(\theta)
  \ge
  \eta g^\top\widetilde g
  -\frac{L\eta^2}{2}\|\widetilde g\|_2^2.
  \label{eq:fixed-step-smoothness}
\end{equation}
The polarization identity gives
\begin{equation}
  2g^\top\widetilde g
  =\|g\|_2^2+\|\widetilde g\|_2^2
   -\|\widetilde g-g\|_2^2.
  \label{eq:polarization}
\end{equation}
Substitution yields
\begin{align}
  J(\theta+\eta\widetilde g)-J(\theta)
  &\ge
  \frac{\eta}{2}
  \{\|g\|_2^2-\|\widetilde g-g\|_2^2\}
  \nonumber\\
  &\quad+
  \frac{\eta}{2}(1-L\eta)\|\widetilde g\|_2^2.
  \label{eq:fixed-step-polarized}
\end{align}
Because $\eta\le1/L$, the last term is nonnegative.  Dropping it, taking
conditional expectation, and setting
$m_k^e=\E[\|\widehat g_k^e-g_k\|_2^2\mid\mathcal F_{k-1}]$ proves
Equation~\eqref{eq:expected-risk-progress}.

\subsection{Intact prompt groups}

Suppose $m$ prompts are independent and each prompt has $G$ conditionally
independent responses.  Keep each prompt group intact and define its contribution
by $\Xi_{k,i}:=G^{-1}\sum_{g=1}^G W_{k,ig}f_{k,ig}$.  A leave-one-out baseline is
admissible because it excludes the focal response; conditional on the prompt and
the remaining responses, the focal score still has mean zero.  Thus
$\E[\Xi_{k,i}\mid\mathcal F_{k-1}]=g_k$.

Jensen's inequality gives
\begin{align}
  \|\Xi_{k,i}\|_2^2
  &=\left\|\frac1G\sum_{g=1}^G W_{k,ig}f_{k,ig}\right\|_2^2
  \nonumber\\
  &\le\frac1G\sum_{g=1}^G
  W_{k,ig}^2\|f_{k,ig}\|_2^2.
  \label{eq:group-jensen}
\end{align}
Repeating the sample-mean calculation across the $m$ independent groups gives
\begin{align}
  \E\!\left[
    \left\|
      \frac1m\sum_{i=1}^m\Xi_{k,i}-g_k
    \right\|_2^2
    \middle|\mathcal F_{k-1}
  \right]
  &\le\frac1m\E[\|\Xi_{k,1}\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\le\frac{\|f_k\|_\infty^2}{m\rho_k}.
  \label{eq:group-rho-bound}
\end{align}
Markov's inequality then gives the same conclusion as
Lemma~\ref{lem:rho-gradient-reliability} with the independent-unit count $N$
replaced by the number of prompt groups $m$.

\subsection{Proof of Proposition~\ref{prop:clipping-crossover}}

For the permissive estimator,
$\E[Z_k^{\mathrm P}\mid\mathcal F_{k-1}]=g_k$.  Conditional independence gives
\begin{equation}
  m_k^{\mathrm P}
  =\E\!\left[
    \left\|
      \frac1N\sum_{i=1}^N
      (Z_{k,i}^{\mathrm P}-g_k)
    \right\|_2^2
    \middle|\mathcal F_{k-1}
  \right]
  =\frac{v_k^{\mathrm P}}{N}.
  \label{eq:permissive-risk-proof}
\end{equation}
For the clipped estimator, add and subtract its conditional mean:
\begin{equation}
  \widehat g_k^{\mathrm C}-g_k
  =
  (\widehat g_k^{\mathrm C}-\E\widehat g_k^{\mathrm C})
  +b_k^{\mathrm C}.
  \label{eq:clipped-bias-split}
\end{equation}
Squaring and taking conditional expectation gives
\begin{align}
  m_k^{\mathrm C}
  &=\E\|\widehat g_k^{\mathrm C}-\E\widehat g_k^{\mathrm C}\|_2^2
    +\|b_k^{\mathrm C}\|_2^2
  \nonumber\\
  &\quad+
  2(b_k^{\mathrm C})^\top
  \E[\widehat g_k^{\mathrm C}-\E\widehat g_k^{\mathrm C}]
  \nonumber\\
  &=\frac{v_k^{\mathrm C}}{N}+\|b_k^{\mathrm C}\|_2^2,
  \label{eq:clipped-risk-proof}
\end{align}
where the cross term is zero.  Therefore
\begin{align}
  m_k^{\mathrm C}<m_k^{\mathrm P}
  &\iff
  \|b_k^{\mathrm C}\|_2^2+\frac{v_k^{\mathrm C}}{N}
  <\frac{v_k^{\mathrm P}}{N}
  \nonumber\\
  &\iff
  v_k^{\mathrm P}-v_k^{\mathrm C}
  >N\|b_k^{\mathrm C}\|_2^2.
  \label{eq:crossover-proof}
\end{align}
This is Equation~\eqref{eq:crossover}.

\subsection{Proof of Corollary~\ref{cor:clipping-needed}}

Equation~\eqref{eq:expected-risk-progress} is positive for estimator $e$ exactly
when $m_k^e<\|g_k\|_2^2$.  It is therefore positive for the clipped estimator
and nonpositive for the permissive estimator exactly when
$m_k^{\mathrm C}<\|g_k\|_2^2\le m_k^{\mathrm P}$, which is
Equation~\eqref{eq:clipping-needed}.

\subsection{Finite-moment relaxation}

Equation~\eqref{eq:rho-gradient-mse} requires only
$\E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]<\infty$.  The bounded contribution
condition is used solely to obtain the conventional importance-sampling form in
Equation~\eqref{eq:rho-gradient-mse-bound}.  More generally, define
$M_{2,k}:=\E_Q[W_k^2\|f_k\|_2^2]/\E_Q[W_k^2]$.  The same calculation gives
$\E\|\widehat g_k-g_k\|_2^2\le M_{2,k}/(N\rho_k)$.  We use the bounded form in
the main text because it isolates normalized ESS as the mismatch-dependent
factor.

'''


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    if start not in text:
        raise RuntimeError(f"Missing start marker: {start}")
    if end not in text:
        raise RuntimeError(f"Missing end marker: {end}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + replacement + end + after


def one_display(block: str, name: str) -> None:
    count = block.count(r"\begin{equation}") + block.count(r"\begin{align}")
    if count != 1:
        raise RuntimeError(f"{name} must contain exactly one displayed conclusion; found {count}")


def main() -> None:
    path = Path("main.tex")
    text = path.read_text(encoding="utf-8")

    text = replace_between(
        text,
        r"\section{Theoretical analysis}",
        r"\section{Contextual-bandit validation of the framework}",
        THEORY,
    )

    text = replace_between(
        text,
        r"\section{Proofs for the theoretical results}",
        r"\bibliographystyle{plainnat}",
        PROOFS,
    )

    lemma1 = text.split(
        r"\begin{lemma}[Normalized ESS controls the gradient-error radius]", 1
    )[1].split(r"\end{lemma}", 1)[0]
    one_display(lemma1, "Lemma 1")

    lemma2 = text.split(
        r"\begin{lemma}[A reliable direction yields policy improvement]", 1
    )[1].split(r"\end{lemma}", 1)[0]
    one_display(lemma2, "Lemma 2")

    theorem = text.split(
        r"\begin{theorem}[When a permissive update remains reliable]", 1
    )[1].split(r"\end{theorem}", 1)[0]
    one_display(theorem, "Main theorem")

    proposition = text.split(
        r"\begin{proposition}[Bias--variance condition for clipping]", 1
    )[1].split(r"\end{proposition}", 1)[0]
    one_display(proposition, "Clipping proposition")

    corollary = text.split(
        r"\begin{corollary}[When clipping recovers the certificate]", 1
    )[1].split(r"\end{corollary}", 1)[0]
    one_display(corollary, "Clipping corollary")

    if r"\begin{assumption}" in text:
        raise RuntimeError("Assumption block survived")
    if "sigma_{g" in text or r"\mathrm{nESS}" in text:
        raise RuntimeError("Obsolete notation survived")
    if "figures/figures_mains/" in text:
        raise RuntimeError("Duplicated figure path survived")

    path.write_text(text, encoding="utf-8")
    print("Rewrote the theoretical analysis with single-conclusion results and line-by-line proofs.")


if __name__ == "__main__":
    main()
