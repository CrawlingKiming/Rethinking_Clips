# Contextual-bandit validation

This experiment follows the classification-to-contextual-bandit protocol used
by Wang, Agarwal, and Dudik at ICML 2017. A labeled Optdigits image is a
context, the ten digit labels are actions, and selecting the correct label gives
reward one. The finite labeled population makes the true policy gradient,
importance-weight moments, and policy reward exactly computable.

The experiment tests three links in the paper's proposed mechanism:

1. Higher ESS should predict lower raw-gradient MSE when gradient scale is
   controlled.
2. Lower gradient MSE should yield better one-step population improvement.
3. A prespecified ESS gate should retain raw updates at high coverage and apply
   the actual PPO advantage-sign gradient mask after coverage falls.

The controlled overlap sweep fixes one current policy and perturbs the rollout
policy independently of rewards and gradient contributions. For each condition,
the script enumerates the exact MSE of the raw and PPO-masked gradients, then
compares always-raw, always-masked, and ESS-gated updates on the same batches.
The gate uses the raw update when sample ESS is at least 0.1 and the PPO mask
otherwise. The threshold is prespecified to match the motivating LLM observation
and is not tuned on these results. Each condition uses 100 independent batches
of 2,048 context-action-reward observations.

The optimization study includes a fresh rollout, whose policy equals the
initial current policy, and a deliberately stale-rollout stress condition with
population ESS 0.002531. Each batch is fixed for 16 updates, while current
probabilities, ratios, scores, ESS, and the gate decision are recomputed before
every update. The main figure reports the one-step mechanism. The result files
add the 16-update trajectories at step sizes 2 and 5. Each setting uses 100
independent batches and paired comparisons.

Install the two dependencies and run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/ess_policy_optimization.py --repetitions 100
```

The script downloads the official UCI Optdigits archive on first use. It rejects
more than 100 repetitions.

To recompute the CSV without Matplotlib, use:

```powershell
python simulation/ess_policy_optimization.py --repetitions 100 --skip-figure
```

Outputs:

- `simulation/results/ess_coverage_results.csv`
- `simulation/results/ess_optimization_summary.csv`
- `simulation/results/ess_optimization_paths.csv`
- `figures/ess_policy_validation.pdf`
- `figures/ess_policy_validation.png`

References:

- Wang, Agarwal, and Dudik, *Optimal and Adaptive Off-policy Evaluation in
  Contextual Bandits*, ICML 2017.
- Alpaydin and Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine
  Learning Repository, 1998.
