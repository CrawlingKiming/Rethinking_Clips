# RLVR-style ESS simulation

This experiment tests ESS inside the optimization procedure used in RLVR.  It
does not create an artificial rollout policy or sweep an external mismatch
variable.

An Optdigits image is a prompt.  The policy emits a 16-token binary response
that encodes the digit, and an exact-match verifier returns reward one only for
the correct response.  The finite population makes expected reward and its
gradient exactly computable.

Each of the 100 default replications follows this loop:

1. Draw 128 prompts from the current rollout policy.
2. Generate 16 complete responses per prompt and compute leave-one-out
   group-relative advantages.
3. Shuffle the prompt groups into 16 PPO minibatches.
4. Before every minibatch update, recompute current sequence ratios and raw
   sequence ESS.
5. Apply an always-raw update, an always-PPO-masked update, or an ESS-gated
   update that uses the raw gradient for ESS at least 0.1 and the mask below
   0.1.
6. After 16 updates, generate the next rollout batch from the updated policy.

The run contains eight rollout batches.  All methods share initial weights,
prompt draws, sampling uniforms, and minibatch orders within each replication.
The policy uses 16 factorized Bernoulli token heads, so the sequence ratio is a
product of token ratios while the exact population gradient remains available.

Run from the repository root:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/ess_policy_optimization.py --replications 100
```

The script downloads the official UCI Optdigits archive only when the local
copy is absent.  It rejects more than 100 replications.

Outputs:

- `simulation/results/rlvr_training_paths.csv`
- `simulation/results/rlvr_summary.csv`
- `simulation/results/rlvr_minibatch_diagnostics.csv`
- `simulation/results/rlvr_diagnostic_bins.csv`
- `figures/ess_policy_validation.pdf`
- `figures/ess_policy_validation.png`

`rlvr_minibatch_diagnostics.csv` contains every diagnostic minibatch from the
first 20 replications, including ESS, exact squared gradient error, and exact
one-update reward changes.  `rlvr_training_paths.csv` contains the aggregated
population reward after every new rollout batch.

References:

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models*, 2024.
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*,
  2025.
- Wang, Agarwal, and Dudik, *Optimal and Adaptive Off-policy Evaluation in
  Contextual Bandits*, ICML 2017.
- Alpaydin and Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine
  Learning Repository, 1998.
