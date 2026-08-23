#!/usr/bin/env bash
# eval_sweep_checkpoints.sh
# =====================================================================
# Iterate a manifest of trained checkpoints and run the SAME val_only eval on each
# (avg@N over the benchmark set), one after another on a single 8xB200 pod. Designed to
# be the LAUNCHER of a Bolt job (submit via bolt_submit.sh). Per checkpoint it downloads
# only the HF export (actor/model/huggingface), evaluates, frees it, then moves on.
#
# Each checkpoint's numbers land in three places (see for_paper/checkpoints.md):
#   - this job's console log (streamed to S3)         -> the durable text record
#   - <SWEEP_DIR>/<name>/tb/                            -> TensorBoard event file
#   - <SWEEP_DIR>/results.tsv                           -> aggregated name/set/mean@N (parsed here)
# SWEEP_DIR is on the artifacts mount when on Bolt, so all of it streams to S3.
#
# Knobs:
#   MANIFEST   default for_paper/eval_manifest.tsv  (name <TAB> task_id <TAB> ckpt/<project>/<exp>)
#   VAL_FILES  benchmarks (default: the 6 boxed sets)     STEP  checkpoint step (default 200)
#   N          avg@N (default 16 = same as training val)  TEMPERATURE/TOP_P (default 0.6/0.95)
#
# Submit (Bolt):
#   CONFIG=bolt_config_train_b200.yaml bash bolt_submit.sh \
#     --env SKIP_CONDA=1 --env LAUNCHER=runs/eval_sweep_checkpoints.sh \
#     --env N=16 --env STEP=200 \
#     --env PROJECT_NAME=verl_qwen3_30b_eval \
#     --env EXPERIMENT_NAME=eval_sweep_A_B1_$(date +%Y%m%d_%H%M%S) --max-retries 3
set -uo pipefail   # NOTE: not -e; one bad checkpoint must not kill the whole sweep

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/.." && pwd)"

MANIFEST=${MANIFEST:-${REPO_DIR}/for_paper/eval_manifest.tsv}
[ -f "${MANIFEST}" ] || { echo "[sweep] manifest not found: ${MANIFEST}" >&2; exit 1; }
VAL_FILES=${VAL_FILES:-"aime_2024 aime_2025 math_500 hmmt_feb_2025 brumo_2025 beyond_aime"}
N=${N:-16}; STEP=${STEP:-200}
TEMPERATURE=${TEMPERATURE:-0.6}; TOP_P=${TOP_P:-0.95}
DL_ROOT=${DL_ROOT:-/mnt/system_runtime/eval_dl}
TS=$(date +%Y%m%d_%H%M%S)

if [ -n "${SWEEP_DIR:-}" ]; then :; elif [ -n "${TURIBOLT_ARTIFACT_DIR:-}" ]; then
    SWEEP_DIR="${TURIBOLT_ARTIFACT_DIR}/eval_sweep_${TS}"
else SWEEP_DIR="./logs/eval_sweep_${TS}"; fi
mkdir -p "${SWEEP_DIR}"
RESULTS="${SWEEP_DIR}/results.tsv"
printf "name\tdata_source\tmetric\tvalue\ttask_id\n" > "${RESULTS}"

echo "[sweep] manifest=${MANIFEST}"
echo "[sweep] benchmarks=${VAL_FILES} | step=${STEP} | avg@${N} | temp=${TEMPERATURE} top_p=${TOP_P}"
echo "[sweep] output -> ${SWEEP_DIR}"

n_done=0; n_fail=0
# strip comments/blank lines; fields are TAB-separated
while IFS=$'\t' read -r name task exp_dir; do
    case "${name}" in ''|\#*) continue ;; esac
    [ -n "${task:-}" ] && [ -n "${exp_dir:-}" ] || { echo "[sweep] SKIP malformed line: '${name}'"; continue; }
    hf_path="${exp_dir}/global_step_${STEP}/actor/model/huggingface"
    run_log_dir="${SWEEP_DIR}/${name}"
    echo ""
    echo "================================================================"
    echo "[sweep] EVAL ${name}  (task ${task}, step ${STEP})"
    echo "[sweep]   ${hf_path}"
    echo "================================================================"

    LOG_DIR="${run_log_dir}" \
    SRC_TASK_ID="${task}" SRC_HF_PATH="${hf_path}" \
    VAL_FILES="${VAL_FILES}" N="${N}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" \
    PROJECT_NAME="${PROJECT_NAME:-verl_qwen3_30b_eval}" \
    EXPERIMENT_NAME="eval_${name}_gs${STEP}_${TS}" \
    DL_ROOT="${DL_ROOT}" \
    bash "${SELF_DIR}/eval_qwen3_30b_a3b_megatron_boxed.sh"
    rc=$?

    if [ ${rc} -eq 0 ]; then n_done=$((n_done+1)); else n_fail=$((n_fail+1)); echo "[sweep] ${name} FAILED rc=${rc} (continuing)"; fi

    # parse the printed metric dict from this run's eval.log -> append to results.tsv
    if [ -f "${run_log_dir}/eval.log" ]; then
        python3 - "${name}" "${task}" "${run_log_dir}/eval.log" "${RESULTS}" <<'PY' || echo "[sweep] parse skipped for ${name}"
import re, sys
name, task, logf, out = sys.argv[1:5]
txt = open(logf, errors="ignore").read()
# keep the LAST value seen per (data_source, metric) e.g. val-core/aime_2025/acc/mean@16: 0.31
pat = re.compile(r"val-core/([^/'\"]+)/acc/(mean@\d+)'?\s*[:=]\s*([0-9.]+)")
last = {}
for src, metric, val in pat.findall(txt):
    last[(src, metric)] = val
with open(out, "a") as f:
    for (src, metric), val in sorted(last.items()):
        f.write(f"{name}\t{src}\t{metric}\t{val}\t{task}\n")
print(f"[sweep] parsed {len(last)} metric(s) for {name}")
PY
    fi

    rm -rf "${DL_ROOT}/hf" 2>/dev/null || true   # free ~60GB before the next checkpoint
    command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
done < "${MANIFEST}"

echo ""
echo "================================================================"
echo "[sweep] DONE. ok=${n_done} failed=${n_fail}. Results table:"
echo "================================================================"
cat "${RESULTS}" || true
