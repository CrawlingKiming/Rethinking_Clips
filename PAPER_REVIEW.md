# Theory-first revision record

## Core argument

1. Multiple updates on fixed rollout data do not by themselves make an
   unclipped estimator unreliable.
2. Policy drift matters through loss of sequence coverage, measured by
   normalized sequence ESS.
3. Exact gradient MSE depends jointly on ESS, the weighted gradient scale, and
   the true-gradient norm. ESS alone is not a reliability certificate.
4. Bounded verifier advantages control one component of the weighted gradient
   scale, while score control remains necessary.
5. Gradient MSE enters a first-order policy-optimization guarantee.
6. Clipping is justified only when covariance reduction exceeds squared bias.
7. The cap and any ESS gate are calibrated estimator choices, not universal
   constants.

## Main-text logic

| Order | Role | Reader takeaway |
|---|---|---|
| Introduction | Problem and thesis | Update count is not a reliability criterion; coverage and gradient influence are. |
| Section 2 | Problem formulation | Separate the population target from finite-sample reliability. |
| Section 3 | Main theorem | ESS governs ratio-driven MSE after isolating the weighted gradient scale. |
| Section 3.1 | RLVR specialization | Bounded advantages help, but do not replace score control. |
| Section 4 | Optimization bridge | Lower gradient MSE tightens a standard stationarity guarantee. |
| Section 5 | Estimator comparison | Clipping helps exactly at a covariance-bias crossover. |
| Section 6 | Decision rule | Calibrate the estimator using ESS, tail mass, and gradient scale. |
| Section 7 | Empirical validation | Test the theorem, the failure of ESS alone, and the clipping crossover on an established benchmark. |
| Appendix | Reproducibility | Define the Optdigits bandit, policies, exact population quantities, sampling protocol, and results. |

The causal chain is:

`coverage and gradient scale -> gradient MSE -> optimization consequence -> clipping decision`.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Fixed-data optimization does not make the fixed-learner estimator intrinsically biased. | Change-of-measure identity and unbiasedness proof. | Supported pointwise |
| ESS controls the ratio contribution to gradient MSE. | Exact factorization in `eq:ess-factorization`. | Supported conditional on weighted gradient scale |
| Bounded RLVR advantages restrict one component of that scale. | Binary-reward observation and group-advantage proposition. | Supported; score control remains necessary |
| Gradient MSE matters for policy optimization. | MSE-controlled stationarity proposition. | Supported for plain stochastic gradient ascent |
| Clipping is preferable at a covariance-bias crossover. | Exact clipping-crossover theorem. | Supported for detached coefficients |
| Upper truncation can cap variance with controlled bias. | Tail-excess bias and MSE corollary. | Supported under bounded sequence contributions |
| The raw MSE identity predicts sampled error. | Optdigits contextual bandit, exact population calculation, and 100 sampled batches per condition. | Supported computationally |
| ESS alone cannot order reliability. | Similar-ESS aligned and shifted loggers with different weighted gradient scales and MSE. | Supported computationally |
| Calibrated clipping can reduce MSE in variance-dominated regimes. | Cap chosen on 40 calibration batches and tested on 60 separate batches. | Supported in this testbed |
| The rule improves language-model training. | Matched-state language-model experiments. | To be tested |

## Self-review

- The abstract and introduction keep notation and cap values out of the story.
- Each section advances one link in the causal chain.
- Proofs expose the algebra line by line and state the identity used.
- The experiment adopts a published contextual-bandit protocol rather than an
  invented two-state environment.
- Main text states why the benchmark is useful, what is measured, and which
  claim each measurement tests. The appendix contains full details.
- Empirical claims are restricted to pointwise gradient estimation. No policy
  return or language-model claim is inferred from the simulation.
- Limitations, conclusion, and language-model transfer remain explicit
  placeholders.
