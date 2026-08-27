# Optdigits theoretical validation

`optdigits_theory_validation.py` is the Optdigits experiment used in the paper. The task is a one-step contextual bandit with the ten digit labels as actions.

- The complete Optdigits dataset is treated as a finite population of 5,620 contexts.
- Each policy iteration draws 320 contexts.
- One digit action is sampled per context from a frozen rollout policy.
- The exact baseline is the rollout probability of the correct class, so the advantage is `reward - Q(correct | image)`.
- Each rollout is divided into 8 minibatches of 40 and used for one optimization epoch.
- A fresh rollout is collected at the next policy iteration.
- All evaluated policy updates use learning rate `0.17`, below the global certified limit computed from the feature covariance.

The script performs two theory-validation studies.

1. It freezes policy pairs across a wide population-ESS range and compares the exact MSE, harmful-update rate, and one-step population change of the unmodified and PPO-masked estimators.
2. It reports the full 40-update learning curves for the two estimators under the certified step size.

No threshold-based update rule or oracle update is evaluated in the reported Optdigits study.

Run from the repository root:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/optdigits_theory_validation.py --replications 100 --diagnostic-replications 40
```

Main outputs:

- `simulation/results/optdigits_estimator_states.csv`
- `simulation/results/optdigits_estimator_redraws.csv`
- `simulation/results/optdigits_estimator_ess_bins.csv`
- `simulation/results/optdigits_estimator_error_bins.csv`
- `simulation/results/optdigits_certified_learning_paths.csv`
- `simulation/results/optdigits_certified_learning_final.csv`
- `figures/optdigits_estimator_comparison.pdf`
- `figures/optdigits_certified_learning.pdf`

`optdigits_categorical_theory.py` and `ess_policy_optimization.py` are retained as exploratory and legacy scripts. They are not used for the reported Optdigits figures.
