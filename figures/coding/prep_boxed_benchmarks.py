#!/usr/bin/env python3
# prep_boxed_benchmarks.py
# =====================================================================
# Prepare "AIME and above" math benchmarks into ./data/dapo_boxed/<name>.parquet
# in the SAME \boxed schema as prep_dapo_boxed.py, so they drop straight into the
# eval launcher (runs/eval_qwen3_30b_a3b_megatron_boxed.sh) and grade with
# verl.utils.reward_score.math_reward via boxed_reward.py.
#
# Row schema (must match data/dapo_boxed/aime_2024.parquet exactly):
#   data_source = "<benchmark_name>"                       # distinct -> per-set metrics
#   prompt      = [{"role":"user","content":"<problem> <BOXED_INSTR>\n"}]
#   ability     = "math"
#   reward_model= {"style":"rule","ground_truth":"<answer>"}
#   extra_info  = {"index": i, "split": "test"}
#
# Run (writes only the sets you ask for; default = all):
#   python3 data/prep_boxed_benchmarks.py
#   python3 data/prep_boxed_benchmarks.py --datasets hmmt_feb_2025 beyond_aime
#
# Then eval with e.g.:
#   VAL_FILES="aime_2024 aime_2025 math_500 hmmt_feb_2025 brumo_2025 beyond_aime" \
#     CONFIG=bolt_config_train_b200.yaml bash bolt_submit.sh --env LAUNCHER=runs/eval_qwen3_30b_a3b_megatron_boxed.sh ...
import argparse
import json
import os

import datasets

BOXED_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."

# name -> (hf_id, hf_config_or_None, split). All are public, integer/short-answer
# competition sets that the \boxed grader (is_equiv) can score.
SOURCES = {
    "hmmt_feb_2025": ("MathArena/hmmt_feb_2025", None, "train"),
    "brumo_2025": ("MathArena/brumo_2025", None, "train"),
    "beyond_aime": ("ByteDance-Seed/BeyondAIME", None, "test"),
}

# Field auto-detection: dataset column names differ across sources.
_PROBLEM_KEYS = ["problem", "question", "Problem", "Question", "prompt", "query"]
_ANSWER_KEYS = ["answer", "Answer", "final_answer", "solution", "gold", "ground_truth", "label"]


def _first_present(ex, keys, what, name):
    for k in keys:
        if k in ex and ex[k] is not None:
            return k
    raise KeyError(
        f"[{name}] could not find a {what} column among {keys}; actual columns = {list(ex.keys())}. "
        f"Add the right key to the _{what.upper()}_KEYS list."
    )


def _problem_text(prompt_field):
    # some sets store the problem as a chat list, most as a plain string
    if isinstance(prompt_field, list):
        us = [m.get("content", "") for m in prompt_field if isinstance(m, dict) and m.get("role") == "user"]
        return (us[-1] if us else prompt_field[-1].get("content", "")).strip()
    return str(prompt_field).strip()


def convert(ds, name):
    rows = []
    pk = ak = None
    for ex in ds:
        if pk is None:
            pk = _first_present(ex, _PROBLEM_KEYS, "problem", name)
            ak = _first_present(ex, _ANSWER_KEYS, "answer", name)
        problem = _problem_text(ex[pk])
        gt = ex[ak]
        # normalize answer to a bare string (is_equiv handles LaTeX/number equivalence);
        # ints/floats -> str, strip any wrapping the source added.
        gt = str(gt).strip()
        if gt.startswith("\\boxed{") and gt.endswith("}"):
            gt = gt[len("\\boxed{"):-1].strip()
        rows.append(
            {
                "data_source": name,
                "prompt": [{"role": "user", "content": f"{problem} {BOXED_INSTR}\n"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"index": len(rows), "split": "test"},
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(SOURCES), choices=list(SOURCES),
                    help="which benchmarks to prep (default: all)")
    ap.add_argument("--out_dir", default="./data/dapo_boxed")
    ap.add_argument("--temp_dir", default="./temp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    for name in args.datasets:
        hf_id, hf_cfg, split = SOURCES[name]
        print(f"\n[prep-bench] loading {hf_id} (config={hf_cfg}, split={split}) ...")
        ds = datasets.load_dataset(hf_id, hf_cfg)[split] if hf_cfg else datasets.load_dataset(hf_id)[split]
        print(f"[prep-bench] raw rows={len(ds)} cols={ds.column_names}")
        rows = convert(ds, name)
        print("[prep-bench] CONVERTED prompt[0]:", json.dumps(rows[0]["prompt"][0]["content"], ensure_ascii=False)[:400])
        print("[prep-bench] CONVERTED gt[0]:", rows[0]["reward_model"]["ground_truth"])
        out_path = os.path.join(args.out_dir, f"{name}.parquet")
        datasets.Dataset.from_list(rows).to_parquet(out_path)
        print(f"[prep-bench] prep -> {out_path}  | rows={len(rows)}")

    print("\n[prep-bench] DONE. Sanity-check the CONVERTED prompt/gt above (bare problem + \\boxed instruction, "
          "answer is a plain value is_equiv can match). Then add the name to VAL_FILES for the eval launcher.")


if __name__ == "__main__":
    main()
