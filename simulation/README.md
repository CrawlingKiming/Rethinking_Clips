# Optdigits theoretical validation

`optdigits_theory_validation.py` is the Optdigits experiment used in the paper. The task is a one-step contextual bandit with the ten digit labels as actions.

- The complete Optdigits dataset is treated as a finite context population of 5,620 images.
- Each policy iteration draws 320 contexts.
- One digit action is sampled per context from a frozen rollout policy.
- The exact baseline is the rollout probability of the correct class, so the advantage is `reward - Q(correct | image)`.
- Each rollout is divided into 8 minibatches of 40 and used for one optimization epoch.
- A fresh rollout is collected at the next policy iteration.

The script performs two theory-validation studies.

1. It freezes policy states across a wide population-ESS range and tests the link from normalized ESS to gradient MSE and one-step population improvement.
2. It compares static unmodified and PPO estimators with an infeasible oracle that computes their exact conditional MSEs and selects the lower-risk estimator. This directly tests the estimator-crossover proposition.

No ESS-gated update rule or ESS threshold is evaluated in the Optdigits study.

Run from the repository root:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/optdigits_theory_validation.py --replications 100 --diagnostic-replications 40
```

Main outputs:

- `simulation/results/optdigits_theory_frozen_states.csv`
- `simulation/results/optdigits_theory_redraws.csv`
- `simulation/results/optdigits_theory_ess_bins.csv`
- `simulation/results/optdigits_theory_error_bins.csv`
- `simulation/results/optdigits_crossover_bins.csv`
- `simulation/results/optdigits_crossover_final.csv`
- `simulation/results/optdigits_crossover_pairwise.csv`
- `figures/optdigits_theory_validation.pdf`
- `figures/optdigits_mse_crossover.pdf`

`optdigits_categorical_theory.py` and `ess_policy_optimization.py` are retained as exploratory and legacy scripts. They are not used for the reported Optdigits results.
