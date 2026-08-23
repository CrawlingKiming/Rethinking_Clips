#!/usr/bin/env bash
# eval_qwen3_30b_a3b_megatron_boxed.sh
# =====================================================================
# val_only EVALUATION of a Qwen3-30B-A3B B200 checkpoint on one or more \boxed
# math benchmarks (avg@N via vLLM). Uses the SAME Megatron topology that WROTE the
# checkpoint (TP2/PP1/EP8, mbridge), so the distributed checkpoint loads natively --
# no HF merge needed. Grades with boxed_reward.py (last \boxed{} + is_equiv).
#
# Headline metric per set: val-core/<data_source>/acc/mean@N  (e.g. val-core/aime_2025/acc/mean@16).
#
# ---- pointing at a checkpoint (pick ONE) ----
#   RECOMMENDED (HF export, model-only, no merge/optimizer):
#     SRC_TASK_ID=<bolt task id> \
#     SRC_HF_PATH=ckpt/<project>/<exp>/global_step_200/actor/model/huggingface
#   -- or a local HF dir already on the pod --
#     HF_MODEL=/path/to/global_step_200/actor/model/huggingface
#   -- Megatron resume from the verl checkpoint (loads the dist-checkpoint) --
#     CKPT=/path/to/global_step_200         (local)   OR
#     SRC_TASK_ID=<id> SRC_CKPT_PATH=ckpt/<project>/<exp>/global_step_200   (download)
#   -- or evaluate the BASE model (nothing set) --
#   (SRC_* artifact paths are relative to the task's artifacts/; see for_paper/checkpoints.md.)
#
# ---- benchmarks ----
#   VAL_FILES defaults to the three in-repo sets. Add prepped ones by name (they must
#   exist under data/dapo_boxed/<name>.parquet -- run data/prep_boxed_benchmarks.py):
#     VAL_FILES="aime_2024 aime_2025 math_500 hmmt_feb_2025 brumo_2025 beyond_aime"
#
# Example (Bolt):
#   SRC_TASK_ID=8hsr6fh27p SRC_CKPT_PATH=checkpoint/verl_.../global_step_200 \
#   VAL_FILES="aime_2024 aime_2025 math_500 hmmt_feb_2025 beyond_aime" N=32 \
#   CONFIG=bolt_config_train_b200.yaml bash bolt_submit.sh \
#     --env SKIP_CONDA=1 --env LAUNCHER=runs/eval_qwen3_30b_a3b_megatron_boxed.sh \
#     --env SRC_TASK_ID=... --env SRC_CKPT_PATH=... --env VAL_FILES="..." --env N=32 --max-retries 3
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

python3 -m pip install -q turibolt 2>/dev/null || echo "[eval-30b] WARN: turibolt install failed"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
NNODES=${NNODES:-1}; NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ------------------- model / checkpoint selection (pick ONE) ------------------------
# The B200 runs save a FULL HF export per checkpoint at
#   <artifacts>/ckpt/<project>/<exp>/global_step_<N>/actor/model/huggingface
# so the RECOMMENDED eval route is HF-export (model-only, no dist-checkpoint/optimizer, no
# merge): point MODEL_PATH at that dir and run with no resume. Modes, in priority order:
#   1. HF_MODEL=/local/hf/dir                          -> eval it directly (no resume)
#   2. SRC_TASK_ID + SRC_HF_PATH=ckpt/.../global_step_N/actor/model/huggingface  (download HF, no resume)
#   3. CKPT=/local/global_step_N                       -> Megatron resume_from_path
#   4. SRC_TASK_ID + SRC_CKPT_PATH=ckpt/.../global_step_N  (download verl ckpt, resume)
#   5. nothing set                                     -> evaluate BASE MODEL_PATH
DL_ROOT=${DL_ROOT:-/mnt/system_runtime/eval_dl}
CKPT=${CKPT:-}; HF_MODEL=${HF_MODEL:-}

# Download an artifact dir from ANOTHER Bolt task's S3. The in-pod turibolt.download_dir takes a
# FULL s3:// URL positionally: download_dir(s3_addr, local_dir, endpoint=...) -- NOT task_id=/
# artifact_path= kwargs. Build <bucket>/tasks/<SRC_TASK_ID>/artifacts/<artifact_path> from this
# task's BOLT_TASK_OUTPUT_PATH (== s3://<bucket>/tasks/<this_task>). (See runs/bolt_ckpt_persist.py.)
_dl () {  # $1=artifact_path (relative to <task>/artifacts/)  $2=dest_dir
    mkdir -p "$2"
    python3 - "$SRC_TASK_ID" "$1" "$2" <<'PY'
import os, sys, turibolt as tb
task_id, artifact_path, local = sys.argv[1:4]
root = os.environ.get("BOLT_TASK_OUTPUT_PATH", "").rstrip("/")   # s3://<bucket>/tasks/<this_task>
if not root:
    sys.exit("[eval-30b] BOLT_TASK_OUTPUT_PATH unset; cannot build s3 address")
base = root.rsplit("/", 1)[0]                                    # s3://<bucket>/tasks
s3_addr = f"{base}/{task_id}/artifacts/{artifact_path}"
endpoint = os.environ.get("BOLT_BLOBBY_ENDPOINT") or "https://conductor.data.apple.com"
err = None
for kwargs in ({}, {"endpoint": endpoint}):
    try:
        tb.download_dir(s3_addr, local, **kwargs)
        print(f"[eval-30b] downloaded {s3_addr} -> {local}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        err = e; print(f"[eval-30b] download attempt failed: {e}", file=sys.stderr)
sys.exit(f"[eval-30b] download failed: {err}")
PY
}

# download_dir may recreate the s3 path under dest; resolve the dir holding a marker file.
_find_dir_with () { find "$1" -type f -name "$2" -printf '%h\n' 2>/dev/null | head -1; }

if [ -z "${HF_MODEL}" ] && [ -n "${SRC_TASK_ID:-}" ] && [ -n "${SRC_HF_PATH:-}" ]; then
    _dl "${SRC_HF_PATH}" "${DL_ROOT}/hf" || { echo "[eval-30b] HF download failed" >&2; exit 1; }
    HF_MODEL="$(_find_dir_with "${DL_ROOT}/hf" config.json)"
    [ -n "${HF_MODEL}" ] || { echo "[eval-30b] no config.json under ${DL_ROOT}/hf" >&2; exit 1; }
fi
if [ -z "${CKPT}" ] && [ -z "${HF_MODEL}" ] && [ -n "${SRC_TASK_ID:-}" ] && [ -n "${SRC_CKPT_PATH:-}" ]; then
    _dl "${SRC_CKPT_PATH}" "${DL_ROOT}/ckpt" || { echo "[eval-30b] ckpt download failed" >&2; exit 1; }
    _actor_dir="$(_find_dir_with "${DL_ROOT}/ckpt" ckpt_contents.json)"   # .../global_step_N/actor
    [ -n "${_actor_dir}" ] || { echo "[eval-30b] no ckpt_contents.json under ${DL_ROOT}/ckpt" >&2; exit 1; }
    CKPT="$(dirname "${_actor_dir}")"                                    # .../global_step_N
fi

RESUME_ARGS=()
if [ -n "${HF_MODEL}" ]; then
    [ -d "${HF_MODEL}" ] || { echo "[eval-30b] HF_MODEL not a directory: ${HF_MODEL}" >&2; exit 1; }
    MODEL_PATH="${HF_MODEL}"; SKIP_MODEL_DOWNLOAD=1     # trained weights ARE the model; no resume
    echo "[eval-30b] evaluating HF export ${MODEL_PATH} (no resume)"
elif [ -n "${CKPT}" ]; then
    [ -d "${CKPT}" ] || { echo "[eval-30b] CKPT not a directory: ${CKPT}" >&2; exit 1; }
    case "$(basename "${CKPT}")" in
        global_step_*) ;;
        *) echo "[eval-30b] CKPT must end with global_step_<N>: ${CKPT}" >&2; exit 1 ;;
    esac
    RESUME_ARGS+=( trainer.resume_mode=resume_path "trainer.resume_from_path=${CKPT}" )
    echo "[eval-30b] evaluating verl checkpoint ${CKPT} (resume_from_path)"
else
    echo "[eval-30b] no checkpoint set -> evaluating BASE model ${MODEL_PATH}"
fi

# mbridge only READS local safetensors (never downloads) -> ensure MODEL_PATH shards are local.
# For a hub id this completes the download; for a local dir (HF export) SKIP_MODEL_DOWNLOAD=1.
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "[eval-30b] completing ${MODEL_PATH} download (mbridge needs full local shards) ..."
    huggingface-cli download "${MODEL_PATH}" --exclude "original/*" \
      || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_PATH}', ignore_patterns=['original/*'])" \
      || { echo "[eval-30b] FATAL: model download failed"; exit 1; }
fi

# ------------------- benchmark val files ------------------------
BOXED_DIR=${BOXED_DIR:-${REPO_DIR}/data/dapo_boxed}
VAL_FILES=${VAL_FILES:-"aime_2024 aime_2025 math_500"}
VAL_LIST=()
for name in ${VAL_FILES}; do
    # allow either a bare benchmark name or a full path
    if [ -f "${name}" ]; then p="${name}"; else p="${BOXED_DIR}/${name}.parquet"; fi
    if [ -f "${p}" ]; then VAL_LIST+=("${p}"); else echo "[eval-30b] WARN: missing ${p} (skipping ${name})"; fi
done
[ ${#VAL_LIST[@]} -gt 0 ] || { echo "[eval-30b] no val files found" >&2; exit 1; }
# Hydra list literal: "['a','b',...]"
VAL_LITERAL="[$(printf "'%s'," "${VAL_LIST[@]}")"; VAL_LITERAL="${VAL_LITERAL%,}]"
# train_files is required by verl even for val_only; reuse the first val file.
TRAIN_LITERAL="['${VAL_LIST[0]}']"
echo "[eval-30b] val_files=${VAL_LITERAL}"

REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_DIR}/verl/verl/trainer/ppo/boxed_reward.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

# ---- eval knobs ----
# NOTE: use EVAL_-prefixed names. bolt_run_train.sh EXPORTS MAX_PROMPT_LENGTH=1024 /
# MAX_RESPONSE_LENGTH=512, which would clobber a bare `${MAX_...:-}` default and cap AIME
# reasoning at 512 tokens -> broken scores. EVAL_* names are not exported by the wrapper.
N=${N:-16}                                    # avg@N rollouts per problem
TEMPERATURE=${TEMPERATURE:-0.6}; TOP_P=${TOP_P:-0.95}
MAX_PROMPT_LENGTH=${EVAL_MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${EVAL_MAX_RESPONSE_LENGTH:-8192}   # harder sets may need more; bump if you have memory
PPO_MAX_TOKEN_LEN_PER_GPU=${EVAL_PPO_MAX_TOKEN_LEN_PER_GPU:-30720}
# train_files/train_batch are UNUSED by val_only, but verl still builds a train dataloader and
# asserts it is non-empty. Our val sets are small (aime=30), so keep the train batch tiny so the
# single reused val file yields >=1 batch. mini_batch == train_batch to satisfy divisibility.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}

# ---- topology (MUST match the training run that wrote the checkpoint) ----
ACTOR_TP=${ACTOR_TP:-2}; ACTOR_PP=${ACTOR_PP:-1}; ACTOR_EP=${ACTOR_EP:-8}; ACTOR_ETP=${ACTOR_ETP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}; ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}

PROJECT_NAME=${PROJECT_NAME:-verl_qwen3_30b_eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_qwen3_30b_$( [ -n "${CKPT}" ] && basename "${CKPT}" || echo base )_$(date +%Y%m%d_%H%M%S)}
LOGGER=${LOGGER:-console,tensorboard}
# Persist eval outputs. The Bolt console log (stdout) ALWAYS streams to S3 and contains the
# printed metric dict, but the TensorBoard event file + sampled generations only survive if
# LOG_DIR is on the artifacts mount (-> S3). Prefer it when on Bolt; else local ./logs.
if [ -z "${LOG_DIR:-}" ]; then
    if [ -n "${TURIBOLT_ARTIFACT_DIR:-}" ]; then LOG_DIR="${TURIBOLT_ARTIFACT_DIR}/eval_logs/${EXPERIMENT_NAME}"
    else LOG_DIR="./logs/${EXPERIMENT_NAME}"; fi
fi
mkdir -p "${LOG_DIR}"; export TENSORBOARD_DIR="${LOG_DIR}/tb"
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/val_generations}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-20}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_LITERAL}" \
    data.val_files="${VAL_LITERAL}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=${REWARD_FN_NAME} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.val_kwargs.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    "trainer.logger=[${LOGGER}]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    model_engine=megatron \
    "${RESUME_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/eval.log"
