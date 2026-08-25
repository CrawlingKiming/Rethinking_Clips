from __future__ import annotations

import re
from pathlib import Path

MAIN = Path("main.tex")
BIB = Path("references.bib")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"Missing start marker: {start}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise RuntimeError(f"Missing end marker: {end}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


ABSTRACT = r'''\begin{abstract}
Permissive policy optimization in reinforcement learning with verifiable rewards
can exhibit a delayed failure: training improves for many updates on a fixed
rollout and then collapses abruptly.  Existing policy-deviation guarantees
explain why large changes can be unsafe, but they do not directly characterize
when a finite rollout stops supporting an accurate update for the current
learner.  We develop an effective-support framework for this transition.  By
viewing the policy gradient as a vector-valued importance-sampling estimator, we
show that its mean-squared error is controlled by normalized sequence effective
sample size.  A smoothness argument then converts estimator reliability into a
one-step population-improvement guarantee.  The resulting theorem explains why
permissive updates can remain effective for an extended period and why their
reliability can deteriorate sharply after effective support collapses.  We
further characterize when clipping can recover a positive improvement
certificate: its variance reduction must outweigh the bias it introduces.  An
exactly evaluable contextual bandit verifies the predicted relations among
effective support, gradient error, and harmful updates.  Large-model experiments
show the same delayed transition and demonstrate that selective safeguards can
preserve early learning while preventing late failure.
\end{abstract}'''


INTRO_RELATED = r'''\section{Introduction}

Reinforcement learning with verifiable rewards repeatedly generates a batch of
responses and then improves the model using that batch.  In many large-model
runs, a permissive update rule works surprisingly well: performance rises
quickly, remains stable for many updates, and then fails abruptly.  Figure~\ref{fig:llm-motivation}
shows this pattern.  The same batch supports useful learning for a long period,
yet near the end of the run the update becomes unstable and performance
collapses.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures_mains/motivation/overall.pdf}
  \caption{A delayed failure in Qwen3-30B-A3B training.  The permissive runs
  improve for many updates and fail only after the fixed rollout is effectively
  supported by very few responses.  This transition is the phenomenon studied
  in the paper.}
  \label{fig:llm-motivation}
\end{figure}

Classical theory explains why moving too far from the model that generated the
data can be dangerous \citep{schulman2015trpo,schulman2017ppo,qi2026dppo}.
That explanation is essential, but it does not tell us why the transition in
Figure~\ref{fig:llm-motivation} is delayed.  It does not track how much
information in a finite batch remains useful for the model after each successive
update.  As a result, it offers no direct answer to the practical question
raised by the figure: why does a permissive rule remain effective for many steps
and then suddenly stop being reliable?

Our starting point is that the relevant object is not distance alone, but
support.  As the model changes, more of the batch may become irrelevant to the
direction in which the model now needs to move.  The nominal batch size can
therefore remain unchanged while the amount of useful information shrinks
dramatically.  We develop a theory that measures this loss and connects it to
the quality of the next update.

The framework has two parts.  First, it quantifies how accurately the fixed
batch estimates the direction that would improve the current model.  Second, it
shows how that accuracy determines whether the next step is guaranteed to
improve the true objective.  Together, these results predict a clear
transition: permissive updates are reliable while the batch retains broad
support, and become vulnerable only after that support has substantially
deteriorated.

A conservative safeguard is not automatically beneficial once this transition
begins.  It helps only when the noise it removes is larger than the useful signal
it suppresses.  Our theory therefore separates two questions: when a permissive
update loses reliability, and when a safeguard can recover it.

We test the framework in a controlled task where the exact update direction and
true performance are known, and then in large-model training.  Across both
settings, the proposed support measure tracks the rise in update error, the
onset of harmful steps, and the point at which selective protection becomes
useful.

\subsection{Contributions}

\begin{enumerate}
  \item We provide a theory showing how the useful portion of a fixed rollout
  controls the accuracy of the next update.
  \item We show how this accuracy determines whether the next step is
  guaranteed to improve the model.
  \item We identify when a selective safeguard can recover that guarantee and
  test the resulting predictions in controlled and large-scale experiments.
\end{enumerate}

\section{Related work}

\paragraph{Policy-deviation guarantees.}
Trust-region and proximal methods control policy movement to obtain stable
improvement guarantees or practical update rules
\citep{schulman2015trpo,schulman2017ppo,qi2026dppo}.  These analyses provide the
standard explanation for why large policy changes can be unsafe.  Our framework
is complementary: it studies when a fixed finite rollout ceases to provide a
reliable estimate of the current policy gradient.  The distinction is between a
population-level approximation guarantee and a finite-sample reliability
question.

\paragraph{Importance sampling and effective support.}
Importance-sampling policy optimization uses likelihood-ratio moments and
R\'enyi divergence to quantify the reliability of off-policy estimators
\citep{metelli2018pois,metelli2020is}.  In these results, effective rather than
nominal sample size controls estimation error.  We apply the same principle
directly to the vector-valued policy-gradient estimator and use normalized
sequence ESS as the regime coordinate.

\paragraph{Reliable policy-gradient updates.}
Smoothness-based analyses translate the error of a finite gradient estimate into
a lower bound on policy improvement and use this relation to choose step sizes
or sample sizes \citep{pirotta2013adaptive,papini2022smoothing}.  We combine
this optimization argument with an ESS-dependent off-policy reliability bound.

\paragraph{Stability in RLVR.}
Recent methods alter ratio boundaries, sequence weights, divergence controls,
or update rules to improve the stability of large-model reinforcement learning
\citep{yu2025dapo,park2025clipping,minimax2025m1,zheng2025m2po,shen2026vespo,
huang2026vcpo,fakoor2026p3o}.  Rather than proposing another static boundary,
we study the statistical transition that makes a permissive update unreliable
and then ask when an intervention can recover reliability.'''


PRELIMINARIES = r'''\section{Preliminaries}
\label{sec:setting}

\subsection{Policy-gradient estimation in fixed-rollout RLVR}

Let $X\sim\nu$ be a prompt, let $Q(\cdot\mid X)$ be the rollout policy, and let
$P_\theta(\cdot\mid X)$ be the current policy.  A complete response is denoted
by $Y=(Y_1,\ldots,Y_T)$, and $R(X,Y)$ is its verifier reward.  The population
objective is $J(\theta):=\E_{P_\theta}[R]$.  Assume
$P_\theta(\cdot\mid X)\ll Q(\cdot\mid X)$ and define the complete-response
likelihood ratio and policy-gradient contribution by
\begin{equation}
  W_\theta(X,Y)
  :=\frac{P_\theta(Y\mid X)}{Q(Y\mid X)},
  \qquad
  f_\theta(X,Y)
  :=\{R(X,Y)-b(X)\}\nabla_\theta\log P_\theta(Y\mid X),
  \label{eq:weight-and-contribution}
\end{equation}
where $b(X)$ is a detached prompt-level baseline.  Because the response is
autoregressive, the sequence score is the sum of the token scores.  The score
identity and change of measure give
\begin{equation}
  g(\theta):=\nabla J(\theta)
  =\E_Q[W_\theta f_\theta],
  \qquad
  \widehat g_N(\theta)
  :=\frac1N\sum_{i=1}^N W_{\theta,i}f_{\theta,i}.
  \label{eq:true-gradient}
\end{equation}
Thus the policy gradient itself is a vector-valued importance-sampling
estimand, and $\widehat g_N(\theta)$ is its estimate from the rollout data.

We analyze one optimization phase.  Before optimization, the rollout batch is
randomly partitioned into disjoint minibatches, and each minibatch is used at
most once in the theoretical traversal.  At update $k$, condition on all
previous updates.  The current policy is then fixed, while the unused minibatch
is a fresh sample from $Q$.  This is the only role of the one-pass construction
in the theory.  When several responses share a prompt or a group-relative
baseline, the intact prompt group is treated as one independent sampling unit;
Appendix~\ref{app:proofs} gives the corresponding extension.

\subsection{R\'enyi divergence and normalized effective sample size}

For $\alpha>1$, define the R\'enyi divergence
$D_\alpha(P_\theta\|Q):=(\alpha-1)^{-1}\log\E_Q[W_\theta^\alpha]$ and its
exponentiated form $d_\alpha(P_\theta\|Q):=\exp\{D_\alpha(P_\theta\|Q)\}$.
At order two, $d_2(P_\theta\|Q)=\E_Q[W_\theta^2]$.  Following
importance-sampling policy optimization \citep{metelli2018pois,metelli2020is},
we define normalized sequence ESS by
\begin{equation}
  \rho_\theta
  :=\frac{1}{d_2(P_\theta\|Q)}
  =\frac{1}{\E_Q[W_\theta^2]}
  =\exp\{-D_2(P_\theta\|Q)\}.
  \label{eq:population-ess}
\end{equation}
The effective sequence count in a batch of size $N$ is $N\rho_\theta$.  When
$P_\theta=Q$, $\rho_\theta=1$ and the full batch is effective.  As the current
policy separates from the rollout policy, $\rho_\theta$ decreases.  Since
$\E_Q[W_\theta]=1$, the identity
$\E_Q[(W_\theta-1)^2]=\rho_\theta^{-1}-1$ shows that normalized ESS is exactly
the inverse second-moment measure of sequence-ratio concentration.'''


THEORY = r'''\section{Theoretical analysis}
\label{sec:theory}

The analysis follows a single chain.  Normalized sequence ESS controls the
reliability of the importance-weighted gradient estimate, and estimator
reliability controls the improvement produced by the next policy update.

At update $k$, condition on the preceding history and write
$W_k:=W_{\theta_k}$, $f_k:=f_{\theta_k}$,
$g_k:=g(\theta_k)$, $\widehat g_k:=\widehat g_N(\theta_k)$, and
$\rho_k:=\rho_{\theta_k}$.  Throughout this section, suppose that the current
gradient contribution is bounded,
$\|f_k\|_\infty:=\sup_z\|f_k(z)\|_2<\infty$.  This is the standard bounded
integrand condition used in importance-sampling bounds.  It controls the scale
of one response contribution; the amplification caused by policy mismatch is
captured separately by $\rho_k$.

\subsection{Normalized ESS controls gradient reliability}
\label{sec:raw}

\begin{lemma}[Normalized ESS controls gradient reliability]
\label{lem:rho-gradient-reliability}
If the unused minibatch contains $N$ independent rollout units, then
\begin{align}
  \E[\widehat g_k\mid\mathcal F_{k-1}]
  &=g_k,
  \label{eq:gradient-unbiasedness}\\
  \E\!\left[
    \|\widehat g_k-g_k\|_2^2
    \mid\mathcal F_{k-1}
  \right]
  &=\frac1N
  \left\{
    \E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]
    -\|g_k\|_2^2
  \right\}
  \nonumber\\
  &\le
  \frac{\|f_k\|_\infty^2}{N\rho_k}.
  \label{eq:rho-gradient-mse}
\end{align}
Consequently, for every $\delta\in(0,1)$,
\begin{equation}
  \Pr\!\left(
    \|\widehat g_k-g_k\|_2
    \le
    \frac{\|f_k\|_\infty}
         {\sqrt{\delta N\rho_k}}
    \,\middle|\,
    \mathcal F_{k-1}
  \right)
  \ge 1-\delta.
  \label{eq:rho-gradient-concentration}
\end{equation}
\end{lemma}

Lemma~\ref{lem:rho-gradient-reliability} is the vector-valued counterpart of the
standard importance-sampling reliability bound.  For a fixed contribution
scale, all mismatch-dependent degradation enters through normalized ESS.  The
nominal batch size $N$ is replaced by the effective sequence count $N\rho_k$.

\subsection{Gradient reliability controls policy improvement}
\label{sec:mse-improvement}

\begin{lemma}[Gradient reliability controls policy improvement]
\label{lem:gradient-error-progress}
Let $g=\nabla J(\theta)$ and let $\widetilde g$ be any estimated update
direction.  If $J$ is $L$-smooth along the update segment, then for every
$0<\eta\le1/L$,
\begin{equation}
  J(\theta+\eta\widetilde g)-J(\theta)
  \ge
  \frac{\eta}{2}
  \left\{
    \|g\|_2^2-\|\widetilde g-g\|_2^2
  \right\}.
  \label{eq:gradient-error-progress}
\end{equation}
If $\widetilde g$ is random with finite second moment, then
\begin{equation}
  \E[J(\theta+\eta\widetilde g)-J(\theta)]
  \ge
  \frac{\eta}{2}
  \left\{
    \|g\|_2^2-\E\|\widetilde g-g\|_2^2
  \right\}.
  \label{eq:expected-gradient-error-progress}
\end{equation}
\end{lemma}

Lemma~\ref{lem:gradient-error-progress} isolates the quantity relevant to one
update: the total squared error of the estimated direction relative to the true
gradient.  The result does not require the estimator to be unbiased; any bias
and variance are both included in the same error term.  The argument is the
standard smoothness step used in safe policy-gradient analyses
\citep{pirotta2013adaptive,papini2022smoothing}.

\subsection{When does a permissive update remain reliable?}

\begin{theorem}[When a permissive update remains reliable]
\label{thm:permissive-reliability}
Suppose $J$ is $L_k$-smooth along the update segment and
$0<\eta_k\le1/L_k$.  Then the unmodified importance-weighted update satisfies
\begin{equation}
  \E\!\left[
    J(\theta_k+\eta_k\widehat g_k)-J(\theta_k)
    \mid\mathcal F_{k-1}
  \right]
  \ge
  \frac{\eta_k}{2}
  \left(
    \|g_k\|_2^2
    -
    \frac{\|f_k\|_\infty^2}{N\rho_k}
  \right).
  \label{eq:permissive-reliability}
\end{equation}
In particular, the lower bound is positive whenever
$N\rho_k\|g_k\|_2^2>\|f_k\|_\infty^2$.
\end{theorem}

\begin{remark}[Interpretation]
\label{rem:rho-interpretation}
Theorem~\ref{thm:permissive-reliability} characterizes a loss of certification,
not a deterministic collapse point.  When the sufficient condition holds, the
fixed rollout supports a positive expected-improvement guarantee for the
permissive update.  When it fails, the update may still improve, but the theorem
can no longer certify that its finite-batch direction is accurate enough.

The result explains the delayed transition in Figure~\ref{fig:llm-motivation}.
A fixed rollout can support many successful updates while $N\rho_k$ remains
large.  Reliability deteriorates only after normalized ESS has fallen enough
for the estimation-error term to compete with the gradient signal.

This criterion complements policy-deviation guarantees.  Those guarantees
control population-level consequences of changing the policy; the present
bound controls the finite-sample reliability of estimating the current gradient
from a fixed rollout.  Both effects can matter, but they answer different
questions.

For fixed $N$, contribution scale, and gradient signal, all mismatch-dependent
degradation in the bound enters through $\rho_k$.  Nevertheless, no universal
numerical threshold follows from the theorem because the required level also
depends on $N$, $\|f_k\|_\infty$, and $\|g_k\|_2$.
\end{remark}

A high-probability adaptive-step version of the same result is given in
Appendix~\ref{app:proofs}.  It replaces the expected-error comparison by the
radius in Equation~\eqref{eq:rho-gradient-concentration}.

\subsection{When can clipping recover reliability?}
\label{sec:clipping-recovery}

The preceding theorem diagnoses when the unmodified estimator loses a positive
reliability certificate; it does not prescribe a remedy.  We now consider a
clipped estimator as a competing gradient estimator.  Let
$Z_k^{\mathrm P}:=W_k f_k$ be the per-unit permissive contribution and let
$Z_k^{\mathrm C}$ be the contribution produced by a chosen clipping rule.  For
$e\in\{\mathrm P,\mathrm C\}$, define
$\widehat g_k^e:=N^{-1}\sum_{i=1}^N Z_{k,i}^e$ and
$m_k^e:=\E[\|\widehat g_k^e-g_k\|_2^2\mid\mathcal F_{k-1}]$.  Write
$b_k^{\mathrm C}:=\E[Z_k^{\mathrm C}\mid\mathcal F_{k-1}]-g_k$ for the
clipping bias and
$v_k^e:=\E[\|Z_k^e-\E Z_k^e\|_2^2\mid\mathcal F_{k-1}]$ for the per-unit
variance.

\begin{proposition}[Bias--variance condition for clipping]
\label{prop:clipping-crossover}
For independent units with finite second moments,
\begin{equation}
  m_k^{\mathrm P}=\frac{v_k^{\mathrm P}}{N},
  \qquad
  m_k^{\mathrm C}=\|b_k^{\mathrm C}\|_2^2+rac{v_k^{\mathrm C}}{N},
  \qquad
  m_k^{\mathrm C}<m_k^{\mathrm P}
  \iff
  v_k^{\mathrm P}-v_k^{\mathrm C}
  >N\|b_k^{\mathrm C}\|_2^2.
  \label{eq:crossover}
\end{equation}
\end{proposition}

Thus clipping gives a stronger expected-improvement lower bound than the
permissive estimator only when the variance it removes exceeds the squared bias
it introduces.

\begin{corollary}[When clipping is needed to recover the certificate]
\label{cor:clipping-needed}
Under the conditions of Lemma~\ref{lem:gradient-error-progress}, the clipped
estimator has a positive expected-improvement certificate while the permissive
estimator does not exactly in the regime
\begin{equation}
  m_k^{\mathrm C}<\|g_k\|_2^2\le m_k^{\mathrm P}.
  \label{eq:clipping-needed}
\end{equation}
\end{corollary}

A small $\rho_k$ therefore diagnoses vulnerability of the permissive estimator,
but does not by itself justify a particular clipping rule.  Recovery requires
that the rule reduce total estimation error after accounting for its bias.  If
both $m_k^{\mathrm P}$ and $m_k^{\mathrm C}$ exceed the gradient signal, neither
estimator has a positive certificate; collecting a fresh rollout is then the
natural response suggested by the theory.

The result is estimator-agnostic.  Full-sequence truncation, token-level
PPO/GRPO masking, and detached coefficient capping define different
$Z_k^{\mathrm C}$ and therefore different biases and variances.  The proposition
does not equate these operations.  The experiments test whether normalized ESS
identifies the regime in which each concrete safeguard becomes useful.

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
not a constant derived from the theorem.'''


CONTEXTUAL_OPENING = r'''\section{Contextual-bandit validation of the framework}
\label{sec:simulation}

The contextual bandit tests three predictions of the theory.  First, normalized
sequence ESS should track the error of the permissive policy-gradient estimator.
Second, harmful updates should become more frequent as this reliability
deteriorates.  Third, a clipped estimator should recover reliability only when
its reduction in variance compensates for its bias, as characterized by
Proposition~\ref{prop:clipping-crossover}.  Because the finite population exposes
both the exact gradient and the exact reward, all three quantities can be
measured directly.  The experiment does not estimate the worst-case contribution
scale in Theorem~\ref{thm:permissive-reliability}; instead, it tests the
predicted ordering of observable errors and update outcomes.'''


LLM_OPENING = r'''\section{Language-model evidence for delayed failure and recovery}
\label{sec:llm-experiments}

The contextual bandit isolates the mechanism with an exact oracle.  We now ask
whether the same delayed transition appears in language-model training, where
the true gradient and population reward change are unavailable.  The framework
predicts that permissive learning should remain effective while normalized
sequence ESS is high, become vulnerable after effective support deteriorates,
and benefit from a selective safeguard only in the later regime.  The following
experiments test this trajectory-level prediction.'''


CONCLUSION = r'''\section{Conclusion}

Figure~\ref{fig:llm-motivation} presents a delayed failure: a permissive update
rule learns successfully from a fixed rollout for many steps and then becomes
unstable.  Existing policy-deviation guarantees provide an essential account of
why large changes can be unsafe, but they do not directly track when a finite
batch stops supporting an accurate update for the current learner.  This paper
develops a complementary effective-support framework for that transition.

The central quantity is normalized sequence ESS, $\rho_k$.  Treating the policy
gradient as a vector-valued importance-sampling estimator shows that its
finite-batch mean-squared error is bounded by the contribution scale divided by
the effective sequence count $N\rho_k$.  Smoothness then converts this
reliability statement into a one-step population-improvement bound.  The result
explains why permissive optimization can remain reliable for an extended period
and why the guarantee disappears only after effective support has sufficiently
deteriorated.

The same framework clarifies the role of clipping.  A low $\rho_k$ diagnoses
vulnerability of the permissive estimator, but it does not automatically make a
clipped estimator preferable.  Clipping recovers the positive certificate only
when its variance reduction exceeds its induced bias and lowers total estimator
risk below the gradient signal.  If neither estimator meets this condition, the
theory points to fresh data rather than stronger clipping.

The contextual bandit confirms the predicted relation among normalized ESS,
gradient error, harmful updates, and estimator crossover.  The language-model
runs show the same temporal ordering at scale: permissive learning succeeds
while effective support is broad, and selective protection becomes useful only
after support deteriorates.  Future work should develop tighter online estimates
of the contribution scale and test whether the resulting reliability boundary
transfers across models, rollout sizes, and verifier tasks.'''


PROOFS = r'''\section{Proofs for the theoretical results}
\label{app:proofs}

This appendix gives the algebraic steps omitted from the main text.  All
statements are read conditional on the history before update $k$, so the current
policy is fixed and the unused sampling units are independent draws from $Q$.

\subsection{Conditional policy-gradient identity}

By definition, $f_k=(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)$ and
$W_k=dP_{\theta_k}/dQ$.  Therefore
\begin{align}
  \E_Q[W_k f_k\mid\mathcal F_{k-1}]
  &=\E_{P_{\theta_k}}
    [(R-b(X))\nabla_\theta\log P_{\theta_k}(Y\mid X)
      \mid\mathcal F_{k-1}]
      \label{eq:proof-change-measure}\\
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

\subsection{Proof of Lemma~\ref{lem:rho-gradient-reliability}}

Let $Z_{k,i}:=W_{k,i}f_{k,i}$ and
$\xi_{k,i}:=Z_{k,i}-g_k$.  Equation~\eqref{eq:conditional-gradient-identity}
implies
\begin{equation}
  \E[\xi_{k,i}\mid\mathcal F_{k-1}]=0.
  \label{eq:centered-unit}
\end{equation}
Since $\widehat g_k-g_k=N^{-1}\sum_{i=1}^N\xi_{k,i}$,
\begin{align}
  &\E\!\left[
    \|\widehat g_k-g_k\|_2^2
    \mid\mathcal F_{k-1}
  \right]
  \nonumber\\
  &\quad=
  \E\!\left[
    \left\|
      \frac1N\sum_{i=1}^N\xi_{k,i}
    \right\|_2^2
    \middle|\mathcal F_{k-1}
  \right]
  \nonumber\\
  &\quad=
  \frac1{N^2}
  \sum_{i=1}^N
  \E[\|\xi_{k,i}\|_2^2\mid\mathcal F_{k-1}]
  \nonumber\\
  &\qquad+
  \frac1{N^2}
  \sum_{i\ne j}
  \E[\xi_{k,i}^\top\xi_{k,j}\mid\mathcal F_{k-1}].
  \label{eq:mse-expand-sums}
\end{align}
For $i\ne j$, conditional independence and
Equation~\eqref{eq:centered-unit} give
\begin{equation}
  \E[\xi_{k,i}^\top\xi_{k,j}\mid\mathcal F_{k-1}]
  =\E[\xi_{k,i}\mid\mathcal F_{k-1}]^\top
   \E[\xi_{k,j}\mid\mathcal F_{k-1}]
  =0.
  \label{eq:cross-terms-zero}
\end{equation}
The units are identically distributed, so
\begin{align}
  \E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  &=\frac1N
    \E_Q[\|W_kf_k-g_k\|_2^2\mid\mathcal F_{k-1}].
  \label{eq:mse-single-unit}
\end{align}
Expanding the remaining squared norm gives
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
where the final equality uses
Equation~\eqref{eq:conditional-gradient-identity}.  Combining
Equations~\eqref{eq:mse-single-unit} and \eqref{eq:mse-expand-unit} proves the
exact equality in Equation~\eqref{eq:rho-gradient-mse}.

Next, $\|f_k(z)\|_2\le\|f_k\|_\infty$ for every $z$, and therefore
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
This proves the MSE bound.

For the probability statement, apply Markov's inequality to the nonnegative
random variable $\|\widehat g_k-g_k\|_2^2$:
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
  &\quad\le\delta.
  \label{eq:markov-rho-proof}
\end{align}
Rearranging proves Equation~\eqref{eq:rho-gradient-concentration}.

\subsection{Proof of Lemma~\ref{lem:gradient-error-progress}}

The quadratic lower bound for an $L$-smooth function gives
\begin{equation}
  J(\theta+\eta\widetilde g)-J(\theta)
  \ge
  \eta g^\top\widetilde g
  -\frac{L\eta^2}{2}\|\widetilde g\|_2^2.
  \label{eq:smoothness-start}
\end{equation}
The polarization identity gives
\begin{equation}
  2g^\top\widetilde g
  =\|g\|_2^2+\|\widetilde g\|_2^2
   -\|\widetilde g-g\|_2^2.
  \label{eq:polarization}
\end{equation}
Substituting Equation~\eqref{eq:polarization} into
Equation~\eqref{eq:smoothness-start} yields
\begin{align}
  J(\theta+\eta\widetilde g)-J(\theta)
  &\ge
  \frac{\eta}{2}
  \{\|g\|_2^2-\|\widetilde g-g\|_2^2\}
  \nonumber\\
  &\quad+
  \frac{\eta}{2}(1-L\eta)\|\widetilde g\|_2^2.
  \label{eq:smoothness-polarized}
\end{align}
Because $0<\eta\le1/L$, the final term is nonnegative.  Dropping it proves
Equation~\eqref{eq:gradient-error-progress}.  Taking expectation on both sides
proves Equation~\eqref{eq:expected-gradient-error-progress}.

\subsection{Proof of Theorem~\ref{thm:permissive-reliability}}

Apply Equation~\eqref{eq:expected-gradient-error-progress} with
$\theta=\theta_k$, $\widetilde g=\widehat g_k$, and $\eta=\eta_k$.  This gives
\begin{align}
  &\E[J(\theta_k+\eta_k\widehat g_k)-J(\theta_k)
      \mid\mathcal F_{k-1}]
  \nonumber\\
  &\quad\ge
  \frac{\eta_k}{2}
  \left\{
    \|g_k\|_2^2
    -
    \E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
  \right\}.
  \label{eq:theorem-before-rho}
\end{align}
Lemma~\ref{lem:rho-gradient-reliability} gives
$\E[\|\widehat g_k-g_k\|_2^2\mid\mathcal F_{k-1}]
\le\|f_k\|_\infty^2/(N\rho_k)$.  Substituting this bound into
Equation~\eqref{eq:theorem-before-rho} proves
Equation~\eqref{eq:permissive-reliability}.

\subsection{High-probability adaptive-step form}

Let $r_k(\delta):=\|f_k\|_\infty/\sqrt{\delta N\rho_k}$.  By
Equation~\eqref{eq:rho-gradient-concentration}, the event
$\|\widehat g_k-g_k\|_2\le r_k(\delta)$ has conditional probability at least
$1-\delta$.  On this event,
\begin{align}
  g_k^\top\widehat g_k
  &=\|\widehat g_k\|_2^2
    +(g_k-\widehat g_k)^\top\widehat g_k
  \nonumber\\
  &\ge\|\widehat g_k\|_2^2
    -\|g_k-\widehat g_k\|_2\|\widehat g_k\|_2
  \nonumber\\
  &\ge\|\widehat g_k\|_2
    \{\|\widehat g_k\|_2-r_k(\delta)\}.
  \label{eq:adaptive-inner-product}
\end{align}
Smoothness therefore gives, for every $\alpha\ge0$,
\begin{align}
  J(\theta_k+\alpha\widehat g_k)-J(\theta_k)
  &\ge
  \alpha\|\widehat g_k\|_2
  \{\|\widehat g_k\|_2-r_k(\delta)\}
  \nonumber\\
  &\quad-
  \frac{L_k\alpha^2}{2}\|\widehat g_k\|_2^2.
  \label{eq:adaptive-quadratic}
\end{align}
The right-hand side is a concave quadratic.  If $\widehat g_k\ne0$, its
nonnegative maximizer is
$\alpha_k^*=L_k^{-1}
(1-r_k(\delta)/\|\widehat g_k\|_2)_+$; set $\alpha_k^*=0$ otherwise.
Substitution gives, with conditional probability at least $1-\delta$,
\begin{equation}
  J(\theta_k+\alpha_k^*\widehat g_k)-J(\theta_k)
  \ge
  \frac1{2L_k}
  \left(
    \|\widehat g_k\|_2
    -\frac{\|f_k\|_\infty}{\sqrt{\delta N\rho_k}}
  \right)_+^2.
  \label{eq:adaptive-rho-improvement}
\end{equation}

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
The bound counts independent prompts rather than all responses and is therefore
conservative.

\subsection{Proof of Proposition~\ref{prop:clipping-crossover} and
Corollary~\ref{cor:clipping-needed}}

For the permissive contribution,
$\E[Z_k^{\mathrm P}\mid\mathcal F_{k-1}]=g_k$.  Independence gives
\begin{equation}
  m_k^{\mathrm P}
  =\E\!\left[
    \left\|
      \frac1N\sum_{i=1}^N
      \{Z_{k,i}^{\mathrm P}-g_k\}
    \right\|_2^2
    \middle|\mathcal F_{k-1}
  \right]
  =\frac{v_k^{\mathrm P}}{N}.
  \label{eq:permissive-risk-proof}
\end{equation}
For the clipped estimator, add and subtract its conditional mean:
\begin{align}
  \widehat g_k^{\mathrm C}-g_k
  &=
  \{\widehat g_k^{\mathrm C}-\E\widehat g_k^{\mathrm C}\}
  +b_k^{\mathrm C}.
  \label{eq:clipped-bias-split}
\end{align}
Squaring and taking expectation gives
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
  \|b_k^{\mathrm C}\|_2^2+rac{v_k^{\mathrm C}}{N}
  <\frac{v_k^{\mathrm P}}{N}
  \nonumber\\
  &\iff
  v_k^{\mathrm P}-v_k^{\mathrm C}
  >N\|b_k^{\mathrm C}\|_2^2,
  \label{eq:crossover-proof}
\end{align}
which proves Proposition~\ref{prop:clipping-crossover}.

Finally, Lemma~\ref{lem:gradient-error-progress} gives the expected-improvement
lower bound $\eta_k\{\|g_k\|_2^2-m_k^e\}/2$ for estimator $e$.  This lower bound
is positive for the clipped estimator and nonpositive for the permissive
estimator exactly when
$m_k^{\mathrm C}<\|g_k\|_2^2\le m_k^{\mathrm P}$, proving
Corollary~\ref{cor:clipping-needed}.

\subsection{Finite-moment relaxation}

The exact equality in Equation~\eqref{eq:rho-gradient-mse} requires only
$\E_Q[W_k^2\|f_k\|_2^2\mid\mathcal F_{k-1}]<\infty$.  The bounded contribution
condition is used solely to obtain the simple standard form
$\|f_k\|_\infty^2/(N\rho_k)$.  More generally, define
$M_{2,k}:=\E_Q[W_k^2\|f_k\|_2^2]/\E_Q[W_k^2]$.  Then the same calculation gives
$\E\|\widehat g_k-g_k\|_2^2\le M_{2,k}/(N\rho_k)$.  We use the bounded form in
the main text to match the conventional importance-sampling notation and to
make the role of normalized ESS explicit.'''


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")

    text = text.replace(
        r"\newtheorem{assumption}[theorem]{Assumption}\n", ""
    )
    text = text.replace(
        r"\title{\bf When Does Clipping Help?\\\nEffective Sample Size and Policy-Gradient Reliability}",
        r"\title{\bf When Does Permissive Policy Optimization Fail?\\\nEffective Sample Size and Gradient Reliability in RLVR}",
    )

    text = replace_block(text, r"\begin{abstract}", r"\end{abstract}", ABSTRACT)
    text = replace_block(text, r"\section{Introduction}", r"\section{Preliminaries}", INTRO_RELATED)
    text = replace_block(text, r"\section{Preliminaries}", r"\section{Theoretical analysis}", PRELIMINARIES)
    text = replace_block(
        text,
        r"\section{Theoretical analysis}",
        r"\section{Contextual-bandit validation of the theory}",
        THEORY,
    )
    text = replace_block(
        text,
        r"\section{Contextual-bandit validation of the theory}",
        r"\subsection{Testbed and oracle}",
        CONTEXTUAL_OPENING,
    )
    text = replace_block(
        text,
        r"\section{Language-model evidence for the regime change}",
        r"\subsection{A minimal intervention}",
        LLM_OPENING,
    )
    text = replace_block(text, r"\section{Conclusion}", r"\appendix", CONCLUSION)
    text = replace_block(
        text,
        r"\section{Proofs for the theoretical results}",
        r"\bibliographystyle{plainnat}",
        PROOFS,
    )

    text = text.replace(
        r"Theorem~\ref{thm:current-ess}",
        r"Theorem~\ref{thm:permissive-reliability}",
    )
    text = text.replace(
        r"Equation~\eqref{eq:ess-policy-improvement}",
        r"Equation~\eqref{eq:permissive-reliability}",
    )
    text = text.replace("figures/figures_mains/", "figures_mains/")

    # Align the closing paragraph of the controlled study with the new claims.
    old = (
        "The controlled study therefore supports the empirical claim that ESS tracks a\n"
        "reliability transition relevant to the sequence-level masking analogue.  It\n"
        "does not by itself establish the objective certificate or transfer the result\n"
        "to token-local PPO, CISPO, or full-sequence truncation.  The next section asks\n"
        "whether a related operational transition is visible in language-model RLVR,\n"
        "where the population gradient cannot be enumerated."
    )
    new = (
        "The controlled study supports the framework in three ways.  Normalized ESS\n"
        "tracks the error of the permissive gradient, harmful updates become more\n"
        "frequent as effective support deteriorates, and the relative risk of the\n"
        "masked and permissive estimators changes across the same regime.  The study\n"
        "does not identify a universal threshold or equate the sequence-level mask\n"
        "with token-local clipping.  The next section tests whether the same delayed\n"
        "failure and selective-recovery pattern appears in language-model RLVR."
    )
    if old in text:
        text = text.replace(old, new)

    # Main-text style checks requested by the author.
    intro = text.split(r"\section{Introduction}", 1)[1].split(r"\section{Related work}", 1)[0]
    for token in ("$", r"\begin{equation}", r"\begin{align}", r"\[", r"\("):
        if token in intro:
            raise RuntimeError(f"Introduction contains forbidden mathematics token: {token}")
    for jargon in (
        "R\\'enyi",
        "importance sampling",
        "variance",
        "bias",
        "gradient",
        "ESS",
        "effective sample size",
        "trust region",
        "surrogate",
        "divergence",
    ):
        if jargon.lower() in intro.lower():
            raise RuntimeError(f"Introduction contains forbidden jargon: {jargon}")

    prelim_theory = text.split(r"\section{Preliminaries}", 1)[1].split(
        r"\section{Contextual-bandit validation of the framework}", 1
    )[0]
    if r"\begin{assumption}" in prelim_theory or "Assumption~" in prelim_theory:
        raise RuntimeError("Assumption block survived in the main theory")
    if "sigma_{g" in prelim_theory or r"\mathrm{nESS}" in prelim_theory:
        raise RuntimeError("Obsolete theory notation survived")
    if "measurable" in prelim_theory.lower():
        raise RuntimeError("Unnecessary measurable wording survived")
    if r"\rho_k" not in prelim_theory:
        raise RuntimeError("Normalized ESS notation is missing")

    theorem = text.split(r"\begin{theorem}[When a permissive update remains reliable]", 1)[1].split(
        r"\end{theorem}", 1
    )[0]
    if theorem.count(r"\begin{equation}") != 1 or r"\begin{align}" in theorem:
        raise RuntimeError("Main theorem must contain exactly one displayed equation")

    if "figures/figures_mains/" in text:
        raise RuntimeError("Duplicated figure path survived")
    if "sigma_{g" in text:
        raise RuntimeError("Old sigma notation survived somewhere in the paper")

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate labels: {duplicates}")
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    missing = sorted(refs - set(labels))
    if missing:
        raise RuntimeError(f"Undefined references: {missing}")

    MAIN.write_text(text, encoding="utf-8")

    bib = BIB.read_text(encoding="utf-8")
    if "@inproceedings{schulman2015trpo" not in bib:
        entry = r'''@inproceedings{schulman2015trpo,
  title     = {Trust Region Policy Optimization},
  author    = {Schulman, John and Levine, Sergey and Moritz, Philipp and Jordan, Michael I. and Abbeel, Pieter},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {37},
  pages     = {1889--1897},
  publisher = {PMLR},
  year      = {2015},
  url       = {https://proceedings.mlr.press/v37/schulman15.html}
}

'''
        bib = entry + bib
        BIB.write_text(bib, encoding="utf-8")

    print("Rewrote the paper around delayed failure, normalized ESS, and reliability recovery.")


if __name__ == "__main__":
    main()
