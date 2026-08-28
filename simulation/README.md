# Theory-validation simulations

## Exactly solvable Gaussian one-step example

`gaussian_quadratic_one_step.py` generates the main-text theoretical example.
The current policy is `Normal(theta, 1)`, the rollout policy is
`Normal(theta - delta, 1)`, and the reward is quadratic. The script evaluates
the Raw and PPO-style masked estimators from closed-form truncated Gaussian
polynomial moments. It uses no Monte Carlo samples.

The reported setting is `N=32`, PPO radius `0.20`, gradient signal `g=2`, and
step size `eta=0.40`. It distinguishes the persistent estimator-risk crossover
from the exact one-step improvement crossover and records the point below which
neither update has a positive certificate. The displayed branch uses
`delta >= 0.20`, corresponding to normalized population ESS at most `0.9608`.

Run from the repository root:

```bash
python simulation/gaussian_quadratic_one_step.py
```

Outputs:

- `simulation/results/gaussian_quadratic_one_step.csv`
- `simulation/results/gaussian_quadratic_one_step_summary.json`
- `figures/gaussian_quadratic_one_step.pdf`
- `figures/gaussian_quadratic_one_step.png`

## Optdigits appendix validation

`optdigits_theory_validation.py` generates the auxiliary Optdigits experiment
reported in the appendix. The task is a one-step contextual bandit with the ten
digit labels as actions.

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

## Exact full-certificate diagnostic

`optdigits_full_certificate.py` compares estimator risk with the full fixed-step certificate on a deterministic library of common frozen states. It uses only the official Optdigits training split. A supervised softmax classifier is fit for 400 full-population steps, the rollout policy is set to `0.20 * W_fit`, and the current policy follows the path from that rollout policy toward and beyond `W_fit`. At every state, the script enumerates every training context and all ten actions to compute exact Raw and PPO means, second moments, MSEs, and

```text
B = eta * <g, mu> - 0.5 * L * eta^2 * E[||g_hat||^2].
```

The defaults use iid estimator batch size `N=320`, PPO radius `0.20`, and `eta=0.17`, which is checked against the global categorical smoothness limit. The output distinguishes the MSE oracle, the full-certificate oracle, and a safe oracle that chooses no update when both certificates are nonpositive. Exact Raw/PPO ties are flagged separately even though the stored deterministic tie-break is Raw. The script also aborts if the inherited shifted-logit clip would invalidate the analytic gradient calculation and records a finite-difference gradient check. This controlled comparison is not a cumulative-return oracle.

Run from the repository root:

```bash
python simulation/optdigits_full_certificate.py
```

The ESS proxy JSON reports two audits so that the threshold's branch-selection sensitivity is explicit. The `full_path` audit includes the small high-ESS PPO mask-boundary blip. The `focused_main_branch` audit uses path scale `lambda >= 0.05` and isolates the persistent Raw-to-PPO-to-no-op branch shown in the figure. Each scope fits its threshold on alternating calibration states and evaluates it on the disjoint alternating half. Both are held-out interpolation on one controlled path, not evidence of transfer to an independent training trajectory.

Outputs:

- `simulation/results/optdigits_full_certificate_path.csv`
- `simulation/results/optdigits_full_certificate_summary.json`
- `simulation/results/optdigits_full_certificate_ess_proxy.json`
- `figures/optdigits_full_certificate.pdf`
- `figures/optdigits_full_certificate.png`
