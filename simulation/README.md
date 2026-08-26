# Optdigits policy-optimization experiments

## Main theory experiment

`optdigits_categorical_theory.py` is the experiment used in the main paper. It is a one-step contextual bandit with the ten digit labels as the action space.

- The complete Optdigits dataset is the finite context population.
- At each policy iteration, 600 images are sampled without replacement.
- One action in `{0, ..., 9}` is drawn for each image from a frozen rollout policy.
- The reward is one when the sampled action equals the image label and zero otherwise.
- The rollout is shuffled into 12 minibatches and used for one optimization epoch.
- A new rollout is collected after the epoch.
- No ESS threshold or ESS-gated update is used.

The script runs static unmodified and static PPO trajectories only to produce policy states with different levels of mismatch. It then freezes 30 states across the observed ESS range and uses independent redraws to evaluate the theoretical links:

1. population normalized ESS versus gradient mean-squared error;
2. realized gradient error relative to the population gradient signal versus one-step population improvement.

The action distribution is a linear softmax policy. Because the context population is finite and the action space contains only ten classes, the population value, population gradient, and population ESS are all computed exactly.

Run from the repository root:

```bash
python -m pip install numpy matplotlib
python simulation/optdigits_categorical_theory.py --replications 12
```

Main outputs:

- `simulation/results/optdigits_categorical_frozen_states.csv`
- `simulation/results/optdigits_categorical_redraws.csv`
- `simulation/results/optdigits_categorical_ess_bins.csv`
- `simulation/results/optdigits_categorical_error_bins.csv`
- `simulation/results/optdigits_categorical_summary.txt`
- `figures/optdigits_categorical_theory.pdf`
- `figures/optdigits_categorical_theory.png`

## Legacy structured-action experiment

`ess_policy_optimization.py` contains the earlier 16-bit structured-action and ESS-gating study. It is retained for reference, but it is not the main Optdigits theory validation and is not used to support the fixed-threshold claim in the main text.
