# Theory-first revision record

## Core argument

1. Multiple policy updates on fixed rollout data do not by themselves make an
   unclipped estimator unreliable.
2. Policy drift matters through loss of sequence coverage, measured by
   normalized sequence ESS.
3. The exact finite-sample MSE factors into sequence coverage, the weighted
   gradient scale, and the true-gradient norm.
4. RLVR reward structure can control the advantage component of the weighted
   gradient scale, while score control remains necessary.
5. Gradient MSE enters a first-order policy-optimization guarantee, which makes
   estimator reliability operationally relevant.
6. Clipping is justified only when its variance reduction exceeds its squared
   gradient bias.
7. An ESS-guided estimator rule is therefore a calibrated decision rule, not a
   universal threshold theorem.

## Main-text logic

| Order | Role | Reader takeaway |
|---|---|---|
| Introduction | Problem and thesis | Optimization step count is not a reliability criterion; sequence coverage is. |
| Section 2 | Problem formulation | Define the target gradient, the unclipped estimator, and sequence ESS before making claims. |
| Section 3 | Main theorem | ESS governs ratio-driven MSE only after isolating the weighted gradient scale. |
| Section 3.1 | RLVR specialization | Bounded verifier advantages help control that scale, but do not replace score control. |
| Section 4 | Optimization bridge | Lower gradient MSE tightens a standard stationarity guarantee. |
| Section 5 | Estimator comparison | Clipping helps exactly when covariance reduction pays for squared bias. |
| Section 6 | Decision rule | Estimate ESS and the missing scale and tail quantities before selecting an estimator. |
| Section 7 | Controlled validation | Hold the target, reward, true gradient, and weighted gradient scale fixed; vary only coverage. |
| Appendix | Reproducibility | Derive the bandit identities and enumerate the exact finite-sample experiment. |

This order follows one causal chain:

`coverage -> ESS -> gradient MSE -> optimization consequence -> clipping decision`.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Fixed-data optimization does not make the fixed-learner estimator intrinsically biased. | Change-of-measure identity and unbiasedness proof. | Supported pointwise |
| Sequence ESS controls the ratio contribution to gradient MSE. | Exact factorization in `eq:ess-factorization`. | Supported conditional on the weighted gradient scale |
| Bounded RLVR advantages restrict one component of that scale. | Binary-reward observation and standardized group-advantage proposition. | Supported; score control remains necessary |
| Gradient MSE matters for policy optimization. | MSE-controlled stationarity proposition. | Supported for plain stochastic gradient ascent |
| Clipping is preferable at a covariance-bias crossover. | Exact clipping-crossover theorem. | Supported for detached coefficients |
| Upper truncation can cap variance with limited bias. | Tail-excess bias and MSE corollary. | Supported under bounded sequence contributions |
| The ESS-to-MSE mechanism appears when other factors are held fixed. | Exactly enumerated contextual bandit with constant `G_2=1/16`. | Supported computationally |
| At sufficiently low ESS, caps 3 and 5 improve both MSE and the next update in the testbed. | Exact one-step policy-value enumeration. | Supported in the stated bandit |
| A sample-ESS gate improves token-level RLVR. | Projected-gradient MSE and matched-state language-model experiments. | To be tested |

## Self-review

- **Single thesis:** every main section advances the coverage-to-decision chain.
- **Notation exposure:** the abstract has no displayed notation, and the
  introduction defers symbols to the theory.
- **Theorem order:** the main ESS result precedes its optimization consequence;
  clipping appears only after the diagnostic has been justified.
- **Proofs:** proofs expose each algebraic step and name the identity or
  assumption used.
- **Scope:** pointwise fixed-learner claims are separated from adaptive-path and
  language-model claims.
- **Experiment:** the main text reports the controlled result briefly; the full
  construction, derivation, and values are in the appendix.
- **References:** citations resolve through `references.bib`.
- **Placeholders:** limitations, conclusion, and language-model transfer remain
  explicitly unfinished.
