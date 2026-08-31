"""Per-step training curves scraped from the saved tool-result JSON dumps.

    from runlog import RUNS, series, ess_cut
    x, y = series("dppo_ess", "eval")          # AIME-2024 mean@16, as a fraction
    x, y = series("dppo_ess", "ess")           # normalized sequence ESS (raw)
    x, y = series("dppo_ess", "clipped")       # fraction of updates the gate clipped

Keys are resolved through METRIC; `ex()` is the raw scraper (regex for `step:(\\d+)`, then the
metric name, first value per step). See for_paper/AGENTS.md sec. 7.
"""
import os
import re
import bisect

# Local-only: figures regenerate from the clean CSVs in the repository-level
# only_for_figures/data directory, with raw log captures used only as a fallback.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
_ONLY = os.path.join(_REPO_ROOT, "only_for_figures")
TR = {"30b": _ONLY, "8b": _ONLY}

# short name -> (tool-result root, dump file). Qwen3-30B-A3B unless noted.
RUNS = {
    "cispo5_nogate":   ("30b", "toolu_bdrk_01Ek39pmRAP665yaNH8NTJer.json"),
    "cispo5_ess":      ("30b", "toolu_bdrk_01VRWeHuzp4kdFNNLUxJZQeZ.json"),
    "grpo_base":       ("30b", "toolu_bdrk_01FyLfb9h96v3adhSvZ3Kd69.json"),
    # GRPO band 0.2/0.2, ESS -> clip (`ircyhpdmku`). CLIP mode, logged clipped max 0.875.
    # Identified by its final val AIME-2024 = 0.3854166666666667, matching that run's held-out
    # number in results_sweep_A_B1.md exactly.
    "grpo_ess_clip":   ("30b", "toolu_bdrk_01BZzMJEy7CYes2yqQkd9T31.json"),
    # --- noclip-PG (cap inf) family, all three verified against their skip counts and grad_norm ---
    # `bvrscfn6u8`: frac.skip.0.015 that never fired (skipped sums to 0.62 over 200 steps), so this
    # is the UNGATED noclip-PG datapoint. pk 44.4 / fin 34.6, grad_norm max 5.9e5.
    "noclip_ungated":  ("30b", "toolu_bdrk_01M52m4oXcMd9SqUfZC7muez.json"),
    # `q2m6j822id`: ESS.clip.0.1 at band 0.2/0.2. pk 43.3 / fin 38.1, grad_norm max 1.7e5.
    "noclip_ess_clip": ("30b", "toolu_bdrk_017ciJ3LVntPGREDggZcSpSF.json"),
    # `vm7vcynvy7`: ESS.skip.0.1, skipped sums to 92.5. pk 35.6 / fin 35.6, grad_norm max 26.8.
    # This is the dump previously mislabelled here as "grpo_noclip_ess"; it is a SKIP-mode run, so
    # its actor/gate/clipped is flat zero by construction.
    "noclip_ess_skip": ("30b", "toolu_bdrk_01Gxvu6GoG5vAg4i4hmp1BYJ.json"),
    "dapo_base":       ("30b", "toolu_bdrk_01BPfMQ6tSwhoVaN7mB4wFWB.json"),
    "dapo_ess":        ("30b", "toolu_bdrk_011KkdE9G6bxr1px9n71uyzV.json"),
    "dppo_base":       ("30b", "toolu_bdrk_01793GojpN3dMUkRkAz6rfd6.json"),
    "dppo_ess":        ("30b", "toolu_bdrk_016EShQTS3SL8t5zQDQL67hu.json"),
    "cispo3_nogate":   ("8b",  "toolu_bdrk_01XM9ftkJHYeXqHFKtwZ9TF1.json"),
    # --- Qwen3-8B (fm_proj_speech B200). baseline clips/latches every step; ESS = same rule,
    # ESS-conditional. Band-matched pairs, verified from launch env in run.log. ---
    "q8b_grpo_alwaysclip":  ("8b", "toolu_bdrk_01PbQfV9kK2uRrGvkfFWpdvy.json"),  # d95iir8hri  0.2/0.2
    "q8b_grpo_ess":         ("8b", "toolu_bdrk_0147V9PB5pgKDqaqcpcanQop.json"),  # n2bu8xky6c  ESS.clip 0.2/0.2
    "q8b_dapo_alwaysclip":  ("8b", "toolu_bdrk_01SgbSVoFVXeszH4RkHRB6AZ.json"),  # 8rx5xvf7dt  0.2/0.28
    "q8b_dapo_ess":         ("8b", "toolu_bdrk_01RE4CDjjc2m81WC9LZMn1nj.json"),  # ahx4ge5hjp  ESS.clip 0.2/0.28
    "q8b_dppo_alwayslatch": ("8b", "toolu_bdrk_01H5jTTSvsMEpSstHzrpfF7M.json"),  # zrmqamex4j  dppo-tv latch/step
    "q8b_dppo_ess":         ("8b", "toolu_bdrk_01DK7BCQnmR9KZubFyWQCPdy.json"),  # vdfa57r99z  dppo-tv + ESS
    "q8b_grpo_base":        ("8b", "toolu_vrtx_01KqiuvAizsji8UKnzV9FJg2.json"),  # yfs6ms6w6a  fresh GRPO 0815
    "q8b_dapo_base":        ("8b", "toolu_vrtx_016pSkhjFC8FRCGAErbV9Gcw.json"),  # 5sra49tycr  fresh DAPO 0815
    "q8b_dapo_ess_nonorm":  ("8b", "toolu_bdrk_015TRT8JsugTjwJRUTLFBTtb.json"),  # bxjnvy6f3s  ESS 0.2/0.28, no sum-norm
    # --- Qwen2.5-7B (fm_proj_speech B200, git sha 02c84dfc). MATH-500 val (mean@1), no AIME set. ---
    "q257b_grpo":       ("8b", "q257b_grpo_w5rwzuttpv_runlog.txt"),        # w5rwzuttpv  GRPO
    "q257b_cispo3_ess": ("8b", "q257b_cispo3ess_qzrn8vpezj_runlog.txt"),   # qzrn8vpezj  CISPO 3 + ESS (clip-latch)
    # --- R1-Distill-1.5B (fm_proj_speech B200, git sha 02c84dfc). AIME-2024 val (mean@16). ---
    "r1_15b_grpo":       ("8b", "r1_15b_grpo_9uhj22zww9_runlog.txt"),       # 9uhj22zww9  GRPO
    "r1_15b_cispo3_ess": ("8b", "r1_15b_cispo3ess_bbvp7vj9j5_runlog.txt"),  # bbvp7vj9j5  CISPO 3 + ESS (clip-latch)
    # --- Qwen3-4B HH-RLHF (fm_proj_speech B200). reward = critic/score/mean (RM scale), no AIME. ---
    "rlhf_grpo":         ("8b", "hhrlhf_grpo_funegmmbmz_runlog.txt"),       # funegmmbmz  GRPO baseline
    "rlhf_cispo3_ess":   ("8b", "hhrlhf_cispo3ess_qshqvngbnt_runlog.txt"),  # qshqvngbnt  cispo3 + ESS·clip·0.1
    # --- GSPO ESS-threshold ablation (30B, token-mean; gate inert vs GSPO's 3e-4 clip). ---
    "gspo_base":         ("8b", "gspo_q3cfydj8eu_runlog.txt"),       # q3cfydj8eu  GSPO, no gate
    "gspo_ess005":       ("8b", "ess005_c332ayragg_runlog.txt"),     # c332ayragg  GSPO + ESS 0.05
    "gspo_ess02":        ("8b", "ess02_782xyquesk_runlog.txt"),      # 782xyquesk  GSPO + ESS 0.2
    # --- PPO-minibatch (updates/rollout) sweep, cispo3+ESS (gpd_shape, sum-norm). updates=256/mini.
    # NB: mb32 used GATE_ESS=0.2, mb16/mb8 used 0.1 (threshold not matched across the sweep). ---
    "q30b_mb32":         ("8b", "q30b_mb32_uz5xrdzr9k_runlog.txt"),   # uz5xrdzr9k  mini=32 -> 8 updates
    "q30b_mb16":         ("8b", "q30b_mb16_3tw5bvbqiu_runlog.txt"),   # 3tw5bvbqiu  mini=16 -> 16 updates
    "q30b_mb8":          ("8b", "q30b_mb8_g5q2wcdp9q_runlog.txt"),    # g5q2wcdp9q  mini=8  -> 32 updates
    # --- cispo3 (gpd_shape cap3, sum-norm, raw, clip) ESS-threshold ablation; only GATE_ESS varies.
    # 0.1 midpoint = ircyhpdmku (grpo_ess_clip, same config). Gate is REAL here (unlike gspo). ---
    "cispo3_ess005":     ("8b", "cispo3_ess005_k9ec6cfvkg_runlog.txt"),  # k9ec6cfvkg  ESS 0.05
    "cispo3_ess02":      ("8b", "cispo3_ess02_mfw7j84534_runlog.txt"),   # mfw7j84534  ESS 0.2
    "gspo_ess01":        ("8b", "gspo_ess01_8m66pubxgu_runlog.txt"),     # 8m66pubxgu  GSPO ESS 0.1 (GSPO midpoint)
    # --- §C FRAC-gate family (30B, coeff). Gate trigger = upper out-of-band frac (not ESS). ---
    "frac_noclip_9p9f":  ("8b", "frac_noclip_9p9fp7hf5r_runlog.txt"),   # 9p9fp7hf5r  noclip cap3 FRAC·clip·0.015
    "frac_noclip_mb5d":  ("8b", "frac_noclip_mb5dsdddxq_runlog.txt"),   # mb5dsdddxq  noclip cap3 FRAC·clip·0.015
    "frac_noclip_yduur": ("8b", "frac_noclip_yduurpxhds_runlog.txt"),   # yduurpxhds  noclip cap3 FRAC·clip·0.015
    "frac_gpd_33c6":     ("8b", "frac_gpd_33c6pmxban_runlog.txt"),      # 33c6pmxban  GPD-shape FRAC·skip·0.015
    "frac_gpd_c7nr":     ("8b", "frac_gpd_c7nrafjupe_runlog.txt"),      # c7nrafjupe  GPD-shape FRAC·skip·0.01
    "frac_cispo3_gezu":  ("8b", "frac_cispo3_gezukrk4gy_runlog.txt"),   # gezukrk4gy  CISPO cap3 FRAC·skip·0.015
    # --- Qwen3-1.7B-base (p4d, interactive verl-gopo-train). GSM8K/MATH -> AIME-2024 mean@16.
    # lr is the key axis: 1e-5 collapses, 1e-6 stable. metrics.txt source (same format). ---
    "q17b_grpo_lr1e6":          ("8b", "q17b_grpo_lr1e6_zbx5g5nkpt_metrics.txt"),           # zbx  GRPO lr1e-6
    "q17b_cispo3_essclip_lr1e6":("8b", "q17b_cispo3_essclip_lr1e6_zbx5g5nkpt_metrics.txt"), # zbx  cispo3 ESS-clip lr1e-6 (rerun)
    "q17b_cispo3_essdppo_lr1e6":("8b", "q17b_cispo3_essdppo_lr1e6_x25bmb3v2d_metrics.txt"), # x25  cispo3 ESS-dppo lr1e-6
    "q17b_grpo_lr1e5":          ("8b", "q17b_grpo_lr1e5_x25bmb3v2d_metrics.txt"),           # x25  GRPO lr1e-5 (collapses)
}

METRIC = {
    "eval":      "val-core/aime_2024/acc/mean@16",
    "eval_math500": "val-core/math_500/acc/mean@1",
    "ess":       "actor/gate/ess_norm_raw",
    "ess_shaped":"actor/gate/ess_norm_shaped",
    "clipped":   "actor/gate/clipped",
    "frac_upper":"actor/gate/frac_upper",
    "skipped":   "actor/gate/skipped",
    "trip":      "actor/gate/trip",
    "grad_norm": "actor/grad_norm",
    "length":    "response_length/mean",
    "length_clip": "response_length/clip_ratio",
    "entropy":   "actor/entropy",
    "reward":    "critic/score/mean",
    "ppo_kl":    "actor/ppo_kl",
}


def ex(root, fname, key):
    """{step: value} for `key`, taking the first value logged at each step."""
    t = open(os.path.join(TR[root], fname), encoding="utf-8", errors="replace").read()
    sp = [(m.start(), int(m.group(1))) for m in re.finditer(r"step:(\d+)", t)]
    st = [p for p, _ in sp]
    d = {}
    for m in re.finditer(re.escape(key) + r":(?:np\.float64\()?([0-9.eE+-]+)", t):
        i = bisect.bisect_right(st, m.start()) - 1
        if i >= 0 and sp[i][1] not in d:
            try:
                d[sp[i][1]] = float(m.group(1))
            except ValueError:
                pass
    return d


_DATADIR = os.path.join(_ONLY, "data")


def _series_from_csv(run, metric, cut):
    """(steps, values) from the clean CSV in only_for_figures/data/, or None if no CSV."""
    import csv as _csv
    p = os.path.join(_DATADIR, run + ".csv")
    if not os.path.exists(p):
        return None
    xs, ys = [], []
    with open(p) as f:
        r = _csv.DictReader(f)
        if metric not in (r.fieldnames or []):
            return ([], [])                      # CSV exists but this metric wasn't logged
        for row in r:
            v = row.get(metric, "")
            if v not in ("", None):
                s = int(row["step"])
                if cut is None or s <= cut:
                    xs.append(s)
                    ys.append(float(v))
    return (xs, ys)


def series(run, metric, cut=None):
    """(steps, values) for RUNS[run], optionally truncated at step <= cut.

    Reads the clean per-metric CSV in only_for_figures/data/ (built by build_figure_data.py) first;
    only falls back to scraping the raw dump if no CSV / metric column is present."""
    csvres = _series_from_csv(run, metric, cut)
    if csvres is not None and csvres[0]:
        return csvres
    root, fname = RUNS[run]                       # fallback: scrape the raw dump
    d = ex(root, fname, METRIC[metric])
    if not d and metric == "ess":
        d = ex(root, fname, "actor/gate/ess_norm")   # some runs log only the unsuffixed key
    ks = [k for k in sorted(d) if cut is None or k <= cut]
    if not ks:
        raise KeyError(f"{run}: no data for {metric}")
    return ks, [d[k] for k in ks]



def gate_fraction(run, metric="clipped"):
    """(steps, corrected fraction) for a gate action.

    The logged fraction is taken over the micro-batches the gate can modify; the first micro-batch
    of each rollout is on-policy and is not counted, so the logged value tops out at 7/8 = 0.875
    (both clip-mode runs hit exactly that). When the gate is latched for the batch, i.e. the logged
    value is above 0.5, that on-policy micro-batch is covered too and the true fraction is

        (8y + 1) / 8 = y + 1/8,   clamped at 1.0

    Below 0.5 the gate is firing intermittently rather than latched, so the logged value stands.
    """
    xs, ys = series(run, metric)
    return xs, [min(y + 1.0 / 8.0, 1.0) if y > 0.5 else y for y in ys]


def ess_cut(run, floor=0.01):
    """Step at which the ESS trace stops being interpretable, else None.

    Rule, applied identically to every run: the first step where ESS < `floor` **and** the eval
    never again exceeds its value at that step. Past such a point the policy is degenerate, so
    ESS compares two collapsed distributions and its recovery is an artefact rather than a signal.
    A run that dips below `floor` but then improves (a working gate) is not truncated.
    """
    es, ev = series(run, "ess"), series(run, "eval")
    evd = dict(zip(*ev))
    for s, v in zip(*es):
        if v >= floor:
            continue
        prior = [evd[k] for k in evd if k <= s]
        later = [evd[k] for k in evd if k > s]
        if prior and later and max(later) <= max(prior):
            return s
    return None
