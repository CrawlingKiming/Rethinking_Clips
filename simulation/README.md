# Optdigits policy-optimization experiments

## Main categorical experiment

`optdigits_categorical_theory.py` contains both Optdigits studies used in the paper. The task is a one-step contextual bandit with the ten digit labels as actions.

- The complete Optdigits dataset is the finite context population.
- Each policy iteration draws 600 contexts independently from that population.
- One action in `{0, ..., 9}` is sampled per context from a frozen rollout policy.
- The rollout is divided into 12 minibatches and used for one optimization epoch.
- A new rollout is collected after the epoch.

The first study freezes 30 states and uses independent redraws to test the two theoretical links: population normalized ESS versus gradient MSE, and realized gradient error versus one-step population improvement.

The second study compares static unmodified and PPO updates with an exact MSE oracle. At every update, the oracle computes the full conditional MSE of both estimators, including PPO bias, and selects the smaller-risk estimator. This is the oracle version of the crossover proposition. It does not use an ESS threshold. The script also evaluates sample-ESS gates at thresholds `0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8` as practical heuristics.

The 100-replication control study uses learning rate `2.0`. A 12-replication diagnostic learning-rate sweep is stored in `simulation/results/optdigits_oracle_lr_sweep.csv`. All optimization methods share rollout randomness within each paired replication. Run from the repository root:

```bash
python -m pip install numpy matplotlib
python simulation/optdigits_categorical_theory.py --replications 100
```

Main outputs:

- `simulation/results/optdigits_categorical_frozen_states.csv`
- `simulation/results/optdigits_categorical_redraws.csv`
- `simulation/results/optdigits_categorical_runs.csv`
- `simulation/results/optdigits_categorical_final_values.csv`
- `simulation/results/optdigits_categorical_thresholds.csv`
- `simulation/results/optdigits_categorical_pairwise.csv`
- `figures/optdigits_categorical_theory.pdf`
- `figures/optdigits_categorical_control.pdf`

## Legacy structured-action experiment

`ess_policy_optimization.py` contains the earlier 16-bit structured-action study. It is retained for reference and is not the main Optdigits experiment.
