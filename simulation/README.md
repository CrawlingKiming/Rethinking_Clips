# Contextual-bandit simulation

This study uses a bounded linear contextual bandit with Bernoulli rewards and a
softmax policy. It fixes 96 contexts, 8 actions, 6 features, and batches of 32
samples. Context norms are at most one and the linear reward means lie in
`[0.1, 0.9]`. The finite context and action sets make the population gradient,
ESS, estimator bias, covariance, and MSE exactly enumerable. Paired sampled
batches are reused for eight updates in the intervention study.

The exact MSE calculation is the primary validation. The reused-batch
intervention is deliberately reported with uncertainty intervals because a
lower one-step gradient MSE need not order nonlinear multi-step policy values.

Run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/contextual_bandit_ess.py
```

The script fixes all random seeds and writes:

- `simulation/results/contextual_bandit_results.csv`
- `figures/contextual_bandit_ess.pdf`
- `figures/contextual_bandit_ess.png`
