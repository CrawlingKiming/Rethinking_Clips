# Contextual-bandit validation

This experiment follows the classification-to-contextual-bandit protocol used
by Wang, Agarwal, and Dudik at ICML 2017. A labeled Optdigits image is a
context, the ten digit labels are actions, and selecting the correct label gives
reward one. The finite labeled population makes the true policy gradient and
all importance-weight moments exactly computable.

The experiment tests three claims from the paper:

1. Observed raw-gradient MSE should match the exact theoretical MSE.
2. ESS alone should not order reliability when the weighted gradient scale
   changes.
3. A cap chosen on separate calibration batches should reduce held-out MSE only
   when variance reduction exceeds clipping bias.

Aligned logging policies are temperature variants of the target classifier.
Shifted logging policies are trained under covariate shift before their
temperatures are varied. This creates regimes with similar ESS but different
associations between importance weights and gradient contributions.

Install the two dependencies and run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/contextual_bandit_ess.py --repetitions 100
```

The script downloads the official UCI Optdigits archive on first use. It rejects
more than 100 repetitions. Each condition uses 40 batches to select a cap from a
fixed grid and 60 held-out batches to report MSE. No cap value is interpreted as
a scientific result.

To recompute the CSV without Matplotlib, use:

```powershell
python simulation/contextual_bandit_ess.py --repetitions 100 --skip-figure
```

Outputs:

- `simulation/results/contextual_bandit_results.csv`
- `figures/ess_theory_validation.pdf`
- `figures/ess_theory_validation.png`

References:

- Wang, Agarwal, and Dudik, *Optimal and Adaptive Off-policy Evaluation in
  Contextual Bandits*, ICML 2017.
- Alpaydin and Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine
  Learning Repository, 1998.
