# Contextual-bandit validation

This experiment follows the classification-to-contextual-bandit protocol used
by Wang, Agarwal, and Dudik at ICML 2017. A labeled Optdigits image is a
context, the ten digit labels are actions, and selecting the correct label gives
reward one. The finite labeled population makes the true policy gradient,
importance-weight moments, and policy reward exactly computable.

The experiment tests three claims from the paper:

1. Higher ESS should predict lower raw-gradient MSE when gradient scale is
   controlled.
2. Lower gradient MSE should yield better one-step population improvement.
3. An ESS gate should retain raw updates at high coverage and apply the actual
   PPO advantage-sign gradient mask after coverage falls.

The controlled logger sweep fixes one evaluation policy and perturbs logging
logits independently of the gradient contributions. The fixed-rollout study
then compares unclipped, always-clipped PPO, and ESS-gated updates. Each study
uses 100 independent batches. Forty runs select the ESS threshold and 60
held-out runs report the optimization result.

Install the two dependencies and run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/ess_policy_optimization.py --repetitions 100 --optimization-steps 35
```

The script downloads the official UCI Optdigits archive on first use. It rejects
more than 100 repetitions.

To recompute the CSV without Matplotlib, use:

```powershell
python simulation/ess_policy_optimization.py --repetitions 100 --optimization-steps 35 --skip-figure
```

Outputs:

- `simulation/results/ess_coverage_results.csv`
- `simulation/results/policy_optimization_summary.csv`
- `simulation/results/policy_optimization_paths.csv`
- `figures/ess_policy_validation.pdf`
- `figures/ess_policy_validation.png`

References:

- Wang, Agarwal, and Dudik, *Optimal and Adaptive Off-policy Evaluation in
  Contextual Bandits*, ICML 2017.
- Alpaydin and Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine
  Learning Repository, 1998.
