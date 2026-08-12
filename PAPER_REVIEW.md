# Theory-first revision record

## Core story

1. Reusing a rollout batch does not make an unclipped update intrinsically unstable.
2. Policy drift changes full-sequence likelihood ratios; sequence ESS measures their concentration.
3. For the exact full-sequence gradient, the ratio-driven MSE scales through `1/(N rho)`.
4. Clipping is useful only when covariance reduction pays for squared gradient bias.
5. A sample-ESS gate for token-level RLVR is an empirical hypothesis, not a theorem.

## Reverse outline

| Section | Paragraph role | Message |
|---|---|---|
| Abstract | challenge | Staleness alone does not imply instability. |
| Abstract | theory | Sequence ESS links policy coverage to raw gradient MSE. |
| Abstract | decision | Clipping requires an MSE crossover, not merely low ESS. |
| Introduction | opening | Reuse creates mismatch, which raises the reliability question. |
| Introduction | gap | Existing clipping and adaptive methods do not by themselves quantify raw gradient MSE. |
| Introduction | bridge | ESS turns policy drift into effective sequence count. |
| Introduction | evidence | The exact raw MSE factors through sequence ESS and weighted gradient scale. |
| Introduction | intervention | Clipping is a bias--variance decision after reliability is diagnosed. |
| Sections 2--4 | derivation | Build change of measure, raw estimator MSE, ESS factorization, and clipping crossover in order. |
| Section 5 | scope | Separate the exact full-sequence result from a practical sample-ESS gate. |
| Section 6 | evidence needed | State only the three experiments required to validate the practical bridge. |

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| Batch reuse does not itself bias the fixed-learner full-sequence estimator. | Exact change-of-measure identity and raw-estimator unbiasedness. | Supported theoretically |
| Sequence ESS governs the likelihood-ratio contribution to raw gradient error. | Exact MSE factorization in `eq:ess-bridge`. | Supported theoretically, conditional on weighted gradient scale |
| High ESS can preserve an unclipped estimator after multiple updates. | `MSE <= G_2/(N rho)` at fixed learner. | Supported pointwise; not a uniform adaptive-path theorem |
| Clipping is preferred exactly at the covariance--bias crossover. | Theorem 2. | Supported theoretically for detached coefficients |
| Gradient MSE is relevant to optimization. | Smooth-ascent stationarity proposition. | Supported for plain stochastic gradient ascent |
| Sample ESS gates practical token-local RLVR updates. | Direct MSE, matched-state, and transfer tests. | Needs experiments |

## Self-review

- **Clarity:** the abstract and introduction follow problem, failed proxy, governing quantity, exact bridge, decision rule, and scope.
- **Flow:** each sentence advances a stated relation, each paragraph hands off an unresolved quantity, and every theory section supplies the next section's input.
- **Terminology:** `sequence ESS`, `raw estimator`, `modified estimator`, and `MSE crossover` are used consistently.
- **Proofs:** every proof now exposes each algebraic step and states the assumption or identity used for that step.
- **References:** all citations resolve through `references.bib`; no bibliography is embedded in `main.tex`.
- **Unsupported claims:** no empirical crossover, performance gain, or universal threshold is asserted.
- **Missing evidence:** the practical losses, threshold rule, and three experiment result slots remain intentionally blank.
