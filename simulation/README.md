# Exact ESS theory validation

This experiment uses a two-context, two-action contextual bandit. The correct
action depends on the context and returns a Bernoulli reward with mean 0.75; the
other action has mean 0.25. The target policy selects either action with equal
probability. The behavior policy selects the correct action with probability
`q`, which is varied to change sequence ESS while the target policy, reward
model, true gradient, and weighted gradient scale remain fixed.

For a batch of 32 samples, the script computes estimator MSE, clipping bias,
variance, and expected reward after one gradient step by exact finite
enumeration. No Monte Carlo estimate is used.

Run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/contextual_bandit_ess.py
```

To verify the exact identities without regenerating the figure or installing
Matplotlib, run:

```powershell
python simulation/contextual_bandit_ess.py --skip-figure
```

The script writes:

- `simulation/results/contextual_bandit_results.csv`
- `figures/ess_theory_validation.pdf`
- `figures/ess_theory_validation.png`
