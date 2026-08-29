# Theory-first revision record

## Core argument

1. A rollout batch remains useful within an optimization phase only while it
   retains effective support for the current policy.
2. Normalized ESS isolates the explicit likelihood-ratio contribution to the
   finite-sample error of the Raw policy-gradient estimator.
3. Gradient accuracy controls whether the next population update can be
   certified through a smoothness argument.
4. Clipping has lower estimator MSE only when variance reduction exceeds its
   additional squared bias.
5. Lower MSE is not the complete update criterion. Fixed-step improvement also
   depends on population-gradient alignment and the update second moment.
6. The correct statewise reference is therefore a full-certificate oracle,
   with a null update when every candidate certificate is nonpositive.
7. ESS diagnoses the reliability regime, but its numerical threshold depends
   on the signal, estimator, batch size, and step size.

## Main-text logic

| Order | Role | Reader takeaway |
|---|---|---|
| Introduction | Delayed failure and thesis | Effective support, rather than update count alone, determines when rollout data continue to support a reliable update. |
| Section 2 | Related work | Finite-sample reliability complements policy-deviation guarantees. |
| Section 3 | Preliminaries | Factor ratios, scores, advantages, Raw, PPO, and normalized population ESS are defined before the theory uses them. |
| Section 4.1 | Raw reliability | ESS enters the Raw error radius through `(N rho)^(-1/2)`, and smoothness turns that radius into a realized one-step guarantee. |
| Section 4.2 | Estimator and update selection | MSE ranks estimators; the full certificate ranks their fixed-step updates by accounting for alignment and update scale. |
| Section 5 | Exact one-step mechanism | With `N=32`, PPO lowers MSE before it becomes the better update; the safe oracle has explicit ESS regimes on the evaluated branch. |
| Section 6 | Large-model evidence | Permissive learning succeeds at high ESS and selective protection activates after ESS deteriorates. |
| Appendix A--B | Exact details | Gaussian moment calculations and estimator-specific coefficient rules are removed from the main argument. |
| Appendix C | Auxiliary Optdigits validation | Exact finite-population enumeration corroborates the mechanism without carrying the main oracle claim. |
| Remaining appendices | Diagnostics and proofs | Reproduce the empirical details and prove every theoretical result. |

The causal chain is:

`effective support -> estimator risk -> full one-step certificate -> update choice`.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| ESS controls the mismatch contribution to Raw gradient error. | Exact importance-sampling MSE identity and the `1/(N rho)` envelope. | Supported under the stated contribution condition |
| A reliable gradient estimate yields policy improvement. | Smoothness Lemma and the high-probability permissive-update theorem. | Supported for one-step updates |
| Lower PPO MSE need not imply a better update. | Full-certificate theorem and Gaussian interval `0.039230 < rho < 0.122756`. | Supported analytically |
| The Gaussian certificate is exact rather than only a bound. | Quadratic objective with global smoothness `L=1`. | Supported analytically |
| An ESS gate can reproduce the safe oracle in the Gaussian construction. | On the stated branch, `rho=exp(-delta^2)` is one-to-one and the certificate ordering has Raw--PPO--no-update regimes. | Supported on the stated branch |
| Optdigits exhibits related finite-population behavior. | Exact context--action enumeration and auxiliary appendix figures. | Supported computationally |
| Selective safeguards improve late-stage large-model behavior. | Reported Qwen3-30B-A3B runs with prespecified sample-ESS threshold `0.1`. | Supported for the reported configurations |
| One universal ESS threshold applies across settings. | Thresholds differ across the Gaussian and language-model settings. | Explicitly rejected |

## Writing standard used in the revision

- Each section answers one reader question before introducing algebra.
- Every theorem is preceded by its purpose and followed by its operational
  meaning.
- Definitions appear in the preliminaries when more than one later result uses
  them.
- The main text carries the argument; derivations and implementation-specific
  coefficient rules sit in the appendix.
- Scope is stated once at the relevant boundary instead of repeated as
  defensive caveats.
- Paragraphs follow claim, evidence, and implication, with one primary claim
  per paragraph.

## Self-review

- The main theoretical message is stated before estimator-specific details.
- MSE oracle, full-certificate oracle, and cumulative performance are kept
  distinct.
- The Gaussian calculation is reproducible from closed-form truncated normal
  moments and checked by committed code and data.
- The LLM threshold is explicitly not transferred from the Gaussian example.
- Optdigits is auxiliary and no longer determines the main story.
- The conclusion states the remaining gap: practical online estimation of
  alignment and second-moment terms.
