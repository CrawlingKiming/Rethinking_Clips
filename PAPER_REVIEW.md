# Theory-first revision record

## Core story

1. Reusing a rollout batch does not make an unclipped update intrinsically unstable.
2. Policy drift changes full-sequence likelihood ratios; sequence ESS measures their concentration.
3. For the exact full-sequence gradient, the ratio-driven MSE scales through `1/(N rho)`.
4. Bounded RLVR advantages control the weighted gradient scale only with score control.
5. Upper ratio truncation exchanges a tail-excess bias for a capped variance.
6. Clipping is useful only when covariance reduction pays for squared gradient bias.
7. A sample-ESS gate for token-level RLVR is an empirical hypothesis, not a theorem.

## Reverse outline

| Section | Paragraph role | Message |
|---|---|---|
| Abstract | challenge | Staleness alone does not imply instability. |
| Abstract | insight | Sequence ESS links policy coverage to update reliability. |
| Abstract | decision | Clipping trades stability against systematic distortion. |
| Abstract | evidence | A controlled simulation supports the mechanism but not an LLM deployment claim. |
| Introduction | opening | Reuse creates mismatch, which raises the reliability question. |
| Introduction | gap | Existing clipping and adaptive methods do not first establish whether the raw update is unreliable. |
| Introduction | bridge | ESS distinguishes coverage failure from batch age. |
| Introduction | qualification | Coverage and the influence of individual responses jointly govern reliability. |
| Introduction | intervention | Clipping is a conditional estimator choice after reliability is diagnosed. |
| Introduction | evidence | The controlled test validates estimation error but not multi-step reward improvement. |
| Section 3 | scale | Separate bounded advantages from the additional score-control requirement. |
| Section 4 | intervention | Derive the clipping crossover and upper-truncation risk bound. |
| Section 5 | scope | Separate the exact full-sequence result from a practical sample-ESS gate. |
| Section 6 | controlled evidence | Validate exact estimator claims and test a matched-batch intervention in a linear contextual bandit. |
| Sections 7--8 | placeholders | Limitations and conclusion remain intentionally unwritten. |

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| Batch reuse does not itself bias the fixed-learner full-sequence estimator. | Exact change-of-measure identity and raw-estimator unbiasedness. | Supported theoretically |
| Sequence ESS governs the likelihood-ratio contribution to raw gradient error. | Exact MSE factorization in `eq:ess-bridge`. | Supported theoretically, conditional on weighted gradient scale |
| High ESS can preserve an unclipped estimator after multiple updates. | `MSE <= G_2/(N rho)` at fixed learner. | Supported pointwise; not a uniform adaptive-path theorem |
| Verifier advantages can be bounded in RLVR and GRPO. | Binary-reward bound and finite-group standardized-advantage proposition. | Supported; GRPO coupling still requires a group-level analysis |
| Upper truncation has controlled MSE when tail excess is small. | Corollary bounding bias by `H_max tau_c` and MSE by `H_max^2(tau_c^2+c/N)`. | Supported theoretically under bounded sequence contribution |
| Clipping is preferred exactly at the covariance--bias crossover. | Theorem 2. | Supported theoretically for detached coefficients |
| Gradient MSE is relevant to optimization. | Smooth-ascent stationarity proposition. | Supported for plain stochastic gradient ascent |
| The ESS identity and cap crossover occur in a controlled bandit. | Exact finite enumeration over 19 target policies. | Supported computationally and reproducibly |
| Lower gradient MSE improves eight reused updates. | Paired contextual-bandit intervention. | Not established; intervals overlap |
| Sample ESS gates practical token-local RLVR updates. | Projected-gradient MSE, matched-state LLM intervention, and held-out transfer. | Needs LLM experiments |

## Self-review

- **Clarity:** the abstract and introduction follow problem, failed proxy, governing quantity, exact bridge, decision rule, and scope.
- **Notation exposure:** the abstract contains no displayed notation, and the introduction defers every equation and symbol to the theory sections.
- **Flow:** each sentence advances a stated relation, each paragraph hands off an unresolved quantity, and every theory section supplies the next section's input.
- **Terminology:** `sequence ESS`, `raw estimator`, `modified estimator`, and `MSE crossover` are used consistently.
- **Proofs:** every proof now exposes each algebraic step and states the assumption or identity used for that step.
- **References:** all citations resolve through `references.bib`; no bibliography is embedded in `main.tex`.
- **Unsupported claims:** no universal cap, universal ESS threshold, or LLM performance gain is asserted.
- **Controlled evidence:** the simulation code, raw results, and publication-ready figure are tracked in the repository.
- **Missing evidence:** projected LLM-gradient MSE, matched-state LLM results, held-out threshold transfer, limitations, and conclusion remain intentionally blank.
