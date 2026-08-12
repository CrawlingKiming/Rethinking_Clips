# Reviewer-facing revision record

## Compact story outline

- **Challenge:** rollout-batch reuse makes RLVR data stale and concentrates full-sequence importance weights.
- **Diagnostic:** normalized sequence ESS isolates the likelihood-ratio second moment, but not the full gradient error.
- **Estimator insight:** clipping is beneficial only when covariance reduction exceeds the squared bias penalty.
- **Optimization bridge:** gradient MSE enters a standard smooth nonconvex stationarity bound.
- **Operational hypothesis:** use a low-bias update in the healthy-ESS regime and a bounded update after a validated crossover.
- **Validation:** test the MSE crossover and policy-improvement reversal with direct resampling and matched-state forks.

## Claim--evidence map

| Claim | Evidence in the paper | Status |
|---|---|---|
| The raw full-sequence estimator is unbiased at fixed parameter and has the stated exact MSE. | Derivation and finite-sample reliability results in Sections 2--3. | Supported theoretically |
| Sequence ESS isolates likelihood-ratio concentration but not the entire gradient variance. | Factorization through the order-two likelihood-ratio moment and weighted gradient scale. | Supported theoretically |
| A bounded estimator has lower MSE exactly at the stated covariance--bias crossover. | Exact MSE decomposition in Section 4. | Supported theoretically |
| Gradient MSE is optimization-relevant under the stated smooth plain-gradient model. | Nonconvex stationarity theorem in Section 4. | Supported under stated assumptions |
| An ESS gate improves practical PPO/CISPO training. | Proposed matched-state and training comparisons in Section 7. | Needs experiments |
| An ESS threshold near 0.1 marks the practical crossover. | Candidate threshold and planned diagnostic sweep in Section 7. | Needs experiments; not universal |

## Five-dimension self-review

1. **Contribution:** The exact estimator crossover and its optimization bridge are clear; empirical novelty remains contingent on completing the protocol.
2. **Writing clarity:** The Abstract, Introduction, and Conclusion now distinguish proved results, proposed mechanisms, and empirical hypotheses.
3. **Experimental strength:** No completed results are currently included; claims of superior learning curves or final performance must wait for measured evidence.
4. **Evaluation completeness:** The protocol includes main comparisons, matched-state forks, direct MSE diagnostics, and threshold/batch-size ablations; implementation placeholders must be resolved.
5. **Method soundness:** ESS is correctly treated as a partial diagnostic rather than a sufficient decision statistic; transfer of any fixed threshold remains a stated limitation.

## Remaining rejection risks

- Replace every `[EXACT ...]` placeholder before submission.
- Add completed experiments with uncertainty estimates and fair compute-matched baselines.
- Report how the threshold was selected and whether it transfers across models, tasks, and batch sizes.
- Verify all theorem assumptions against the implemented optimizer and clearly separate plain-SGD theory from Adam-based training.
