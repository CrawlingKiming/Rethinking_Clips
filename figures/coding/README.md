# Evaluation code & know-how (Qwen3-30B-A3B RL checkpoints)

How to evaluate the trained B200 checkpoints on math benchmarks (AIME-2024/2025, MATH-500,
HMMT-Feb-2025, BRUMO-2025, BeyondAIME), what every flag means, and the traps we hit getting it
to actually run. Snapshots of the scripts live in this folder; the **canonical runnable copies**
are in `runs/` and `data/` (the sweep and `bolt_submit --env LAUNCHER=runs/...` reference those
paths — edit those, not the snapshots here).

| file (canonical) | snapshot here | role |
|---|---|---|
| `data/prep_boxed_benchmarks.py` | `prep_boxed_benchmarks.py` | build HMMT/BRUMO/BeyondAIME parquet in the \boxed schema |
| `runs/eval_qwen3_30b_a3b_megatron_boxed.sh` | `eval_qwen3_30b_a3b_megatron_boxed.sh` | eval ONE checkpoint (val_only, avg@N over a benchmark list) |
| `runs/eval_sweep_checkpoints.sh` | `eval_sweep_checkpoints.sh` | one Bolt job iterating a manifest of checkpoints |
| `for_paper/eval_manifest.tsv` | `eval_manifest.tsv` | the 11 usable checkpoints (Section A + B1) |

Related: `for_paper/checkpoints.md` (where each checkpoint lives on S3), `for_paper/bolt_tasks.md`
(run index / metrics).

---

## 1. The big picture

1. **Checkpoints already contain a full HF export.** Each `global_step_N/actor/model/huggingface/`
   is a complete Qwen3-30B-A3B in safetensors (written by mbridge during training). So eval needs
   **no Megatron→HF merge** — just load the HF dir. (`for_paper/checkpoints.md` has the exact S3
   paths per run.)
2. **Eval reuses verl `val_only`** with the same vLLM rollout + grader as training, so the numbers
   are directly comparable to the per-step training-validation curve.
3. Metric per benchmark: **`val-core/<data_source>/acc/mean@N`** (e.g. `val-core/aime_2025/acc/mean@16`).

---

## 2. Data prep — `prep_boxed_benchmarks.py`

Writes `data/dapo_boxed/<name>.parquet` in the EXACT schema of the training/val sets (the model was
RL-trained on this format, so eval must match it):

```
data_source = "<benchmark_name>"                         # distinct -> per-set metrics
prompt      = [{"role":"user","content":"<problem> Please reason step by step, and put your final answer within \\boxed{}.\n"}]
ability     = "math"
reward_model= {"style":"rule","ground_truth":"<answer>"}
extra_info  = {"index": i, "split": "test"}
```

Run: `python3 data/prep_boxed_benchmarks.py` (all) or `--datasets hmmt_feb_2025 beyond_aime`.
Sources: HMMT `MathArena/hmmt_feb_2025` (30), BRUMO `MathArena/brumo_2025` (30), BeyondAIME
`ByteDance-Seed/BeyondAIME` (100). It auto-detects the problem/answer columns and prints row 0 —
**eyeball that print**: content must be the bare problem + the \boxed instruction, answer a plain
value `is_equiv` can match. **Commit the parquet after prep** — the Bolt pod clones your HEAD sha.

Grader: `verl/verl/trainer/ppo/boxed_reward.py` → `math_reward.compute_score` (extracts the last
`\boxed{...}`, grades with `is_equiv`; handles ints and LaTeX exprs like `2^{99}`).

---

## 3. Eval one checkpoint — `eval_qwen3_30b_a3b_megatron_boxed.sh`

Loads a checkpoint's HF export and runs `trainer.val_only=True` over `VAL_FILES`, avg@N via vLLM.
Model/checkpoint source (pick one; priority order):

1. `HF_MODEL=/local/hf/dir` — eval a local HF dir directly (no resume).
2. `SRC_TASK_ID=<task>` + `SRC_HF_PATH=ckpt/<project>/<exp>/global_step_N/actor/model/huggingface`
   — **recommended**; downloads the HF export from that task's S3, evals it (no resume).
3. `CKPT=/local/global_step_N` — Megatron `resume_from_path` (loads the dist-checkpoint).
4. `SRC_TASK_ID` + `SRC_CKPT_PATH=ckpt/<project>/<exp>/global_step_N` — download verl ckpt, resume.
5. nothing set — evaluate the BASE model (a floor).

### Flags (env)

| env | default | meaning |
|---|---|---|
| `SRC_TASK_ID` | — | Bolt task id that produced the checkpoint (source of the S3 download) |
| `SRC_HF_PATH` | — | artifact path (under `<task>/artifacts/`) of the HF export dir |
| `SRC_CKPT_PATH` | — | artifact path of the verl checkpoint dir (resume route instead of HF) |
| `HF_MODEL` / `CKPT` | — | use a local HF dir / local `global_step_N` already on the pod |
| `VAL_FILES` | `aime_2024 aime_2025 math_500` | space-separated benchmark names (or full parquet paths) |
| `N` | `16` | avg@N samples per problem. **Use 16 to match the training-val metric** (`mean@16`); 32/64 = lower variance but a different metric name |
| `TEMPERATURE` / `TOP_P` | `0.6` / `0.95` | sampling — same as training validation |
| `EVAL_MAX_PROMPT_LENGTH` | `2048` | prompt cap — **matches training** (see trap #4) |
| `EVAL_MAX_RESPONSE_LENGTH` | `8192` | response cap — **matches training**; bump for harder sets if truncating |
| `EVAL_PPO_MAX_TOKEN_LEN_PER_GPU` | `30720` | dynamic-bsz token budget/GPU (≥ prompt+response) |
| `TRAIN_BATCH_SIZE` | `16` | UNUSED by eval, but verl asserts a non-empty train loader (see trap #3) |
| `ACTOR_TP/PP/EP/ETP` | `2/1/8/1` | Megatron topology — **must match the run that wrote the checkpoint** |
| `ROLLOUT_TP` / `ROLLOUT_GPU_MEM_UTIL` | `4` / `0.8` | vLLM rollout tensor-parallel / GPU mem |
| `DL_ROOT` | `/mnt/system_runtime/eval_dl` | where the HF export is downloaded on the pod |
| `LOG_DIR` | artifacts mount when on Bolt | TB event file + `val_generations` + `eval.log` (→ S3) |

Fixed in-script (do not change without reason): `trainer.val_only=True`, `val_before_train=True`,
`rollout.val_kwargs.do_sample=True` (mandatory — greedy collapses mean@N), grader = `boxed_reward`.

### Where results are stored (3 places)

- **Console log** (stdout → Bolt S3 `tasks/<id>/logs/console`) — the durable text record; verl prints
  the full metric dict each validation.
- **TensorBoard event file** — `LOG_DIR/tb/` (scalars `val-core/<set>/acc/mean@N`).
- **`val_generations/`** — `log_val_generations=20` prompt+response+score samples per set.
On Bolt, `LOG_DIR` is on the artifacts mount so all three stream to S3.

---

## 4. Sweep many checkpoints — `eval_sweep_checkpoints.sh`

One 8×B200 Bolt job iterates `eval_manifest.tsv` sequentially: download each HF export → eval →
parse `val-core/*/acc/mean@N` into `SWEEP_DIR/results.tsv` → free the ~60 GB → next. Uses `set -uo
pipefail` (NOT `-e`) so one bad checkpoint doesn't kill the sweep. Knobs: `MANIFEST`, `VAL_FILES`,
`N`, `STEP` (default 200), `TEMPERATURE`/`TOP_P`. Manifest is TAB-separated
`name <TAB> task_id <TAB> ckpt/<project>/<exp>`; the driver appends
`/global_step_${STEP}/actor/model/huggingface`.

Submit (P2 is gated for `fm_proj_speech` → use `--priority 3`; see trap #1):

```bash
CONFIG=bolt_config_train_b200.yaml bash bolt_submit.sh \
  --env SKIP_CONDA=1 --env LAUNCHER=runs/eval_sweep_checkpoints.sh \
  --env N=16 --env STEP=200 \
  --env VAL_FILES="aime_2024 aime_2025 math_500 hmmt_feb_2025 brumo_2025 beyond_aime" \
  --env PROJECT_NAME=verl_qwen3_30b_eval \
  --env EXPERIMENT_NAME=eval_sweep_A_B1_$(date +%Y%m%d_%H%M%S) \
  --priority 3 --max-retries 3
```

Single-checkpoint smoke (recommended before a full sweep — bugs only surface at trainer init):

```bash
CONFIG=bolt_config_train_b200.yaml bash bolt_submit.sh \
  --env SKIP_CONDA=1 --env LAUNCHER=runs/eval_qwen3_30b_a3b_megatron_boxed.sh \
  --env SRC_TASK_ID=udg7vbgfsn \
  --env SRC_HF_PATH=ckpt/verl_grpo_qwen3_30b_run/qwen3_30b_a3b_grpo_20260810_201650/global_step_200/actor/model/huggingface \
  --env VAL_FILES="aime_2024" --env N=4 --priority 3 --max-retries 1
```

---

## 5. Traps we hit (and the fixes) — read before editing

1. **P2 submission gated.** `Bolt precondition was not met ... P2 ... restricted to p2_schedulers`.
   The b200 config is `priority: 2`; `fm_proj_speech` now restricts P2. Fix: add `--priority 3`
   (lower number = higher priority; P3 is fine for eval), or ask the admin for the group.

2. **`download_dir` signature.** The in-pod turibolt is `download_dir(s3_url, local_dir, endpoint=)`
   with a FULL `s3://` URL — NOT `download_dir(task_id=, artifact_path=, local_path=)` (that raises
   `TypeError: unexpected keyword 'task_id'`; the hint printed in training logs is wrong for this
   version). Build `s3://<bucket>/tasks/<SRC_TASK_ID>/artifacts/<path>` from this task's
   `BOLT_TASK_OUTPUT_PATH`, with a conductor-endpoint fallback. (Mirrors `runs/bolt_ckpt_persist.py`.)
   After download, resolve the real dir by locating `config.json` (HF) / `ckpt_contents.json` (resume),
   since `download_dir` may nest the s3 path under the dest.

3. **`Train dataloader is empty!`** verl builds a train dataloader even for `val_only` and asserts it
   is non-empty. `train_files` reuses a small val set (aime = 30 rows), so a large
   `train_batch_size` (256) gives 0 batches → crash. Fix: `TRAIN_BATCH_SIZE=16`
   (`ppo_mini_batch_size=16` for divisibility). It is never used, only satisfies the assert.

4. **Response length silently clobbered to 512.** `bolt_run_train.sh` EXPORTS
   `MAX_PROMPT_LENGTH=1024` / `MAX_RESPONSE_LENGTH=512`, which overrode a bare `${MAX_...:-8192}`
   default → responses capped at 512 tokens → destroys AIME (needs long chains). Fix: read
   `EVAL_MAX_PROMPT_LENGTH` / `EVAL_MAX_RESPONSE_LENGTH` (the wrapper doesn't set those) →
   **2048 / 8192, identical to training**. Always verify the live command shows
   `data.max_response_length=8192`.

5. **Checkpoints on ephemeral disk are gone.** Older runs (`q5ec38ysrs` GRPO-orig, `vdt83kasxx`
   clip-higher-orig, `q3cfydj8eu` GSPO) saved to `/mnt/system_runtime/checkpoint` (ephemeral pod
   NVMe, never streamed to S3) → unrecoverable. Only runs whose `default_local_dir` was the
   **artifacts mount** (`/mnt/task_wrapper/user_output/artifacts/ckpt/...`) survive. GSPO has no
   recovered checkpoint. (The end-of-run "no complete checkpoint found" from the persist wrapper is
   a FALSE NEGATIVE for the surviving runs — it checks the wrong path / for `.pt` files.)

6. **"incorrect regex pattern / fix_mistral_regex" warning is harmless.** Generic `transformers`
   tokenizer notice; not Mistral-specific and doesn't apply to the Qwen tokenizer. The HF export
   bundles the same tokenizer used at training, so eval tokenization matches training.

7. **`bolt_submit` deploys the current HEAD sha** — commit AND push (launcher, manifest, and the
   benchmark parquet) before submitting, or the pod clones a sha missing your files.

8. **Retention `keep=3`.** Only `global_step_{150,175,200}` survive per run; earlier steps are pruned
   from the mount (and S3). Use `STEP=200` (final) unless you specifically want 150/175.
