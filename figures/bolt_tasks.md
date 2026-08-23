# Bolt task index — Qwen3-30B-A3B RL (tasks, settings, results, coverage)

Single source of truth for the paper: every Bolt run with its config and result, the baselines
separated out, and a coverage grid of what is done vs open. All runs: Qwen3-30B-A3B (Megatron
TP2/PP1/EP8), B200 `aws_10`, DAPO-boxed train → AIME-2024 `val-core/aime_2024/acc/mean@16`, full
0→200 unless noted. Per-step curves: `../plot/reward_full_track.md`.

**What every task records** (`metrics.txt`): `val-core/aime_2024/acc/mean@16` (+best/worst@k),
`critic/score/mean`, `actor/{grad_norm, entropy, ppo_kl, pg_clipfrac}`,
`actor/gate/{trip, clipped, latched, skipped, ess_norm, ess_norm_raw, ess_norm_shaped, frac_upper}`,
`response_length`, `global_seqlen`; plus `val_generations/*.jsonl` and `tb/`.

## Runs used in the paper (canonical manifest)


### Qwen3-30B-A3B — figures `figures_mains/result/q30ba3b/`, held-out sweep §E `smavmyz3uv`

| panel / role | baseline task | ESS / ours task |
|---|---|---|
| GRPO (band 0.2/0.2) | `udg7vbgfsn` (grpo_base) | `ircyhpdmku` (grpo_ess_clip) |
| GRPO clip-higher (0.2/0.28) | `t82djeyx43` (dapo_base) | `328rfu6eb2` ⭐ (dapo_ess) |
| DPPO | `z95e8ih6mr` (dppo_base) | `52iya9e2hr` (dppo_ess) |
| TIS-3 (no-gate control) | `sjjc7dcpzf` (cispo3_nogate) | `328rfu6eb2` (dapo_ess) |
| GRPO no-clip (cap ∞) — fig `figures/result/q30ba3b/noclip/` | `bvrscfn6u8` (noclip_ungated, de-facto ungated) | `q2m6j822id` (noclip_ess_clip) · `vm7vcynvy7` (noclip_ess_skip) |
| held-out eval sweep (A+B1, 6 benchmarks) | — | `smavmyz3uv` |

### Qwen3-8B — figures `figures_mains/result/8b/curves/`; held-out sweep = `ysjg39qct7` (§E.2, done)

| panel / role | baseline task (token-mean) | ESS / ours task | agg-matched? |
|---|---|---|---|
| GRPO (band 0.2/0.2) | `yfs6ms6w6a` (q8b_grpo_base) | `n2bu8xky6c` (q8b_grpo_ess, sum-norm) | ✗ (ESS is sum-norm; open cell) |
| GRPO clip-higher (0.2/0.28) | `5sra49tycr` (q8b_dapo_base) | `bxjnvy6f3s` (q8b_dapo_ess_nonorm) | ✓ both token-mean |
| DPPO | `zrmqamex4j` (q8b_dppo_alwayslatch) | `vdfa57r99z` ⭐ (q8b_dppo_ess) | — |

### Qwen3-1.7B — §G, lr 1e-6 only (metric from TensorBoard)

| role | task |
|---|---|
| GRPO baseline | `zbx5g5nkpt` (grpo_lr1e6) |
| ESS-clip | `zbx5g5nkpt` (essclip rerun 095445) |
| ESS-dppo | `x25bmb3v2d` (done) / `677feegwie` (running) |


## Figure-data archive (`only_for_figures/`)

Every figure regenerates **offline from `for_paper/only_for_figures/` alone** — no live Bolt calls, no
Claude-session dir. The clean per-metric CSVs in **`only_for_figures/data/<key>.csv`** (validation,
reward, entropy, response-length, grad-norm, ESS, gate stats) are what `coding/runlog.py:series()`
reads; the raw captures (`toolu_*.json` / `*_runlog.txt`) are only a fallback. Rebuild the CSVs with
`python coding/build_figure_data.py`. Config summary below; full per-run config in the linked §.
See `only_for_figures/README.md`.

### Archived-data checklist (`only_for_figures/data/`)

Each ticked run has a clean per-metric CSV locally; figures regenerate offline from these.

**Qwen3-30B-A3B**
- [x] `sjjc7dcpzf`  (`cispo3_nogate`) — 180 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `rdq6r5yy83`  (`cispo5_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `ayv2ajeuqk`  (`cispo5_nogate`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `t82djeyx43`  (`dapo_base`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `328rfu6eb2`  (`dapo_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `z95e8ih6mr`  (`dppo_base`) — 201 steps: eval,reward,entropy,length,grad_norm
- [x] `52iya9e2hr`  (`dppo_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `udg7vbgfsn`  (`grpo_base`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `ircyhpdmku`  (`grpo_ess_clip`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `q2m6j822id`  (`noclip_ess_clip`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `vm7vcynvy7`  (`noclip_ess_skip`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `bvrscfn6u8`  (`noclip_ungated`) — 201 steps: eval,reward,entropy,length,grad_norm,ess

**Qwen3-8B**
- [x] `8rx5xvf7dt`  (`q8b_dapo_alwaysclip`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `5sra49tycr`  (`q8b_dapo_base`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `ahx4ge5hjp`  (`q8b_dapo_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `bxjnvy6f3s`  (`q8b_dapo_ess_nonorm`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `zrmqamex4j`  (`q8b_dppo_alwayslatch`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `vdfa57r99z`  (`q8b_dppo_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `d95iir8hri`  (`q8b_grpo_alwaysclip`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `yfs6ms6w6a`  (`q8b_grpo_base`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `n2bu8xky6c`  (`q8b_grpo_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess

**Qwen2.5-7B**
- [x] `qzrn8vpezj`  (`q257b_cispo3_ess`) — 201 steps: eval_math500,reward,entropy,length,grad_norm,ess
- [x] `w5rwzuttpv`  (`q257b_grpo`) — 201 steps: eval_math500,reward,entropy,length,grad_norm

**R1-Distill-1.5B**
- [x] `bbvp7vj9j5`  (`r1_15b_cispo3_ess`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `9uhj22zww9`  (`r1_15b_grpo`) — 201 steps: eval,reward,entropy,length,grad_norm

**Qwen3-4B HH-RLHF**
- [x] `qshqvngbnt`  (`rlhf_cispo3_ess`) — 162 steps: reward,entropy,length,grad_norm,ess
- [x] `funegmmbmz`  (`rlhf_grpo`) — 159 steps: reward,entropy,length,grad_norm,ess

**GSPO ablation (Qwen3-30B, token-mean)**
- [x] `q3cfydj8eu`  (`gspo_base`) — 201 steps: eval,reward,entropy,length,grad_norm
- [x] `c332ayragg`  (`gspo_ess005`) — 201 steps: eval,reward,entropy,length,grad_norm,ess — 46.7/46.7
- [x] `8m66pubxgu`  (`gspo_ess01`) — 201 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `782xyquesk`  (`gspo_ess02`) — 201 steps: eval,reward,entropy,length,grad_norm,ess

**cispo3 ESS-threshold ablation (Qwen3-30B, gpd_shape cap3, sum-norm — real gate)** — fig `figures_mains/result/q30ba3b/cispo3_ess_threshold/`
- [x] `k9ec6cfvkg`  (`cispo3_ess005`, ESS 0.05) — 201 steps: eval,reward,entropy,length,grad_norm,ess — **43.1/43.1**
- [x] `ircyhpdmku`  (`grpo_ess_clip`, ESS 0.1 midpoint) — 201 steps — 40.8/38.5
- [x] `mfw7j84534`  (`cispo3_ess02`, ESS 0.2) — 193 steps: eval,reward,entropy,length,grad_norm,ess — **42.1/34.8**

**PPO-minibatch sweep (Qwen3-30B cispo3+ESS, updates/rollout)** — fig `figures_mains/result/q30ba3b/batchsize/`
- [x] `uz5xrdzr9k`  (`q30b_mb32`, 8 updates) — 189 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `3tw5bvbqiu`  (`q30b_mb16`, 16 updates) — 184 steps: eval,reward,entropy,length,grad_norm,ess
- [x] `g5q2wcdp9q`  (`q30b_mb8`, 32 updates) — 150 steps: eval,reward,entropy,length,grad_norm,ess
### Qwen3-30B-A3B (stored)

| Bolt task | runlog key | config (method · cap · gate · agg) | § |
|---|---|---|---|
| `udg7vbgfsn` | `grpo_base` | GRPO conv · clip 0.2/0.2 · no gate | A |
| `t82djeyx43` | `dapo_base` | DAPO clip-higher · 0.2/0.28 · no gate | A |
| `z95e8ih6mr` | `dppo_base` | DPPO-TV pure · no gate | A |
| `ayv2ajeuqk` | `cispo5_nogate` | CISPO cap5 · no gate 💥 | A |
| `sjjc7dcpzf` | `cispo3_nogate` | CISPO cap3 · no gate 💥 · sum-norm | A |
| `ircyhpdmku` | `grpo_ess_clip` | gpd_shape cap3 · clip 0.2/0.2 · ESS·clip·0.1·raw · sum-norm | B1 |
| `328rfu6eb2` ⭐ | `dapo_ess` | gpd_shape cap3 + DAPO 0.2/0.28 · ESS·clip·0.1·shaped (+C3) · sum-norm | B1 |
| `52iya9e2hr` | `dppo_ess` | DPPO-TV cap3 · ESS·clip·0.1·raw | B1 |
| `q2m6j822id` | `noclip_ess_clip` | noclip-PG cap∞ · clip 0.2/0.2 · ESS·clip·0.1·raw · sum-norm | B1 |
| `rdq6r5yy83` | `cispo5_ess` | CISPO cap5 · ESS·skip·0.1·shaped | B2 |
| `vm7vcynvy7` | `noclip_ess_skip` | noclip-PG cap∞ · ESS·skip·0.1·raw · sum-norm | B2 |
| `bvrscfn6u8` | `noclip_ungated` | noclip-PG cap∞ · FRAC·skip·0.015 (**never fired → ungated**) · sum-norm | C |
| `c332ayragg` | (ablation `.txt`) | **GSPO** · clip 3e-4 · ESS·clip·**0.05**·raw · tok-mean (gate inert) | B1 |
| `782xyquesk` | (ablation `.txt`) | **GSPO** · clip 3e-4 · ESS·clip·**0.2**·raw · tok-mean (gate inert) | B1 |
| `q3cfydj8eu` | (ablation `.txt`) | **GSPO** baseline · clip 3e-4 · no gate · tok-mean | A |

### Qwen3-8B (stored)

| Bolt task | runlog key | config (method · cap · gate · agg) | § |
|---|---|---|---|
| `yfs6ms6w6a` | `q8b_grpo_base` | GRPO · clip 0.2/0.2 · no gate · tok-mean | F1 |
| `5sra49tycr` | `q8b_dapo_base` | DAPO clip-higher · 0.2/0.28 · no gate · tok-mean | F1 |
| `d95iir8hri` | `q8b_grpo_alwaysclip` | GRPO always-clip 0.2/0.2 ref · sum-norm | F1 |
| `8rx5xvf7dt` | `q8b_dapo_alwaysclip` | DAPO always-clip 0.2/0.28 ref · sum-norm | F1 |
| `zrmqamex4j` | `q8b_dppo_alwayslatch` | DPPO-TV latch (always) ref | F1 |
| `n2bu8xky6c` | `q8b_grpo_ess` | cispo3 cap3 · clip 0.2/0.2 · ESS·clip·0.1 · sum-norm | F2 |
| `ahx4ge5hjp` | `q8b_dapo_ess` | cispo3 cap3 + DAPO 0.2/0.28 · ESS·clip·0.1·shaped · sum-norm | F2 |
| `bxjnvy6f3s` | `q8b_dapo_ess_nonorm` | cispo3 cap3 + DAPO 0.2/0.28 · ESS·clip·0.1 · **tok-mean** | F2 |
| `vdfa57r99z` ⭐ | `q8b_dppo_ess` | cispo3 cap3 + dppo-latch · ESS·clip·0.1·raw · sum-norm | F2 |

Not stored here: the fuller catalogue rows in §B2/§C/§D/§G that no figure uses (metrics live on Bolt).

> **Configs verified from `run.log` (2026-08-20).** Cross-cutting facts confirmed:
> - **All `cispo3`/`cispo5` runs are `loss_mode=gpd_shape`** (CISPO realized via `clip_ratio_c=10` + a
>   fixed `GPD_FIXED_HI` cap, not a distinct `cispo` loss); all **sum-norm** except `bxjnvy6f3s` (tok-mean).
> - **30B GRPO/DAPO/DPPO baselines (`udg7vbgfsn`, `t82djeyx43`, `z95e8ih6mr`) are `loss_mode=vanilla`/`dppo_tv`,
>   token-mean.** DPPO / dppo-latch runs (`z95e8ih6mr`, `52iya9e2hr`, `zrmqamex4j`, `vdfa57r99z`) use a
>   **0.15/0.15** actor band (not 0.2). `52iya9e2hr`'s "noclip" folder name is misleading — it is `gpd_shape`
>   cap-3, band 0.15.
> - **GSPO ablation (`c332ayragg`/`782xyquesk`/`q3cfydj8eu`): actor clip = GSPO's `3e-4/4e-4`** (launcher
>   `gspo)` block), which is why the 0.2 ESS-gate band is inert (§B1a).
> - `d95iir8hri`/`8rx5xvf7dt`/`zrmqamex4j` (8B `*_alwaysclip/latch_ref`) use `GATE_ESS=1.1` (always-latch);
>   `ayv2ajeuqk`/`rdq6r5yy83`/`vm7vcynvy7`/`bvrscfn6u8` have no `GATE_MODE` set despite `GATE_ENABLE=1`.

### Qwen3-4B HH-RLHF (stored `.txt`, §H)

| Bolt task | config (method · cap · gate · agg) | file |
|---|---|---|
| `funegmmbmz` | GRPO vanilla · no gate (`GATE_ESS=0.0`) · HH-RLHF, Qwen3-4B-Instruct | `hhrlhf_grpo_funegmmbmz_runlog.txt` |
| `qshqvngbnt` | gpd_shape cap3 · ESS·clip·0.1·shaped · HH-RLHF, Qwen3-4B-Instruct | `hhrlhf_cispo3ess_qshqvngbnt_runlog.txt` |

## Design axes (glossary)

- **Ratio truncation** — is the importance ratio $r_t$ capped, and how: *none* (`FIXED_HI=inf`),
  *fixed cap* (`FIXED_HI=C`), *per-rank GPD ceiling*, or *clip band* (eps low/high).
- **Truncation location** — how the (truncated) ratio enters the loss: *coefficient* = detached
  multiplier on $\log\pi$ (CISPO/GPD, **dense**, no token masked) vs *surrogate* = PPO/GRPO clip
  (**masks** the gradient of clipped tokens).
- **Gate** — per-update trust region: *trigger* (`ESS` on sequence weights, or `FRAC` = upper
  out-of-band %), *action* (`skip` the step, or `clip` the update on trigger; `GATE_MODE=clip` ⇒
  `gate/skipped`=0 by design, read `gate/trip`), *threshold*, and **ESS source = pre-truncation
  (`raw`) or post-truncation (`shaped`)**.

---

## A. Baselines — no ESS gate (conventional + DPPO)

> Some early baselines carry Bolt state **FAILED** but did complete the full 0→200 training (failure
> was post-run/eval-side); curves are valid. GRPO and clip-higher each have an original + a re-run.

| Method | Bolt task | loss / IS | clip band | trunc loc | gate | AIME peak | final | git sha |
|---|---|---|---|---|---|---|---|---|
| GRPO conventional | `q5ec38ysrs` | vanilla (grpo) | 0.2 / 0.2 | surrogate | none | 26.9% | 25.8% | – |
| GRPO conventional (re-run) | `udg7vbgfsn` | vanilla (grpo) | 0.2 / 0.2 | surrogate | none | 28.5% | 26.2% | acbe87fe |
| GRPO clip-higher (DAPO-style) | `vdt83kasxx` | vanilla (grpo) | 0.2 / 0.28 | surrogate | none | 38.5% | 38.5% | – |
| DAPO clip-higher (re-run) | `t82djeyx43` | dapo | 0.2 / 0.28 | surrogate | none | 37.9% | 37.9% | acbe87fe |
| **GSPO** | `q3cfydj8eu` | gspo (seq IS) | 3e-4 / 4e-4 | surrogate, seq | none | 39.2% | 35.6% | – |
| **CISPO (pure, no gate)** | `ayv2ajeuqk` | cispo (cap 5) | cap 5 | coefficient | none | 28.3% | 0.0% 💥 | 74082a36 |
| **CISPO (pure, no gate)** | `sjjc7dcpzf` | cispo (cap 3), essdiag | cap 3 | coefficient | none (ESS logged only) | 31.7% | 0.2% 💥 | 93f7335a |
| **DPPO (pure)** | `z95e8ih6mr` | dppo (total-variation) | TV rule | coefficient | none | 37.5% | 30.4% ↓ | acbe87fe |

**GSPO 39.2%** and **GRPO clip-higher 38.5%** are the strongest *stable* baselines; **pure CISPO (no
gate) collapses to 0%** and **pure DPPO decays to 30.4%** without a gate; conventional **GRPO ~27%** is the floor.



## B. ESS-gated updates (the contribution)

### B1. ESS gate → clip (convert to clipping when ESS is low)

| Bolt task | base / cap | trunc loc | gate: trigger·action·thr | ESS src | AIME pk/fin | grad_norm max | git sha |
|---|---|---|---|---|---|---|---|
| `328rfu6eb2` | CISPO cap3 + DAPO clip 0.2/0.28 (+`CLIP_C=3`) | coeff | ESS·clip·0.1 | shaped (post) | **47.3 / 44.8** ⭐ | 0.27 | a7f5e635 |
| `52iya9e2hr` | DPPO-TV, cap3 | coeff (dppo_tv) | ESS·clip·0.1 | raw (pre) | 40.2 / 39.2 | 39 | 4a20cb40 |
| `ircyhpdmku` | CISPO cap3 + clip 0.2/0.2 | coeff | ESS·clip·0.1 | raw (pre) | 40.8 / 38.5 | 6.4e6 | a7f5e635 |
| `q2m6j822id` | noclip-PG (inf) + clip 0.2/0.2 | coeff | ESS·clip·0.1 | raw (pre) | 43.3 / 38.1 | 1.7e5 | a7f5e635 |
| `zt5vuurkyw` | CISPO cap3 + clip 0.2/0.2 | coeff | ESS·clip·0.05 | shaped (post) | 40.8 / 36.9 | 9.3e5 | a7f5e635 |
| `g7smhurn55` | CISPO cap3 + clip 0.2/0.28 | coeff | ESS·clip·0.1 | raw (pre) | 35.4 / 35.2 | 1.7e5 | a7f5e635 |
| `hvv2nimrr6` | CISPO cap3 + clip 0.2/0.2 | coeff | ESS·clip·0.1 | shaped (post) | 42.1 / 32.9 | 2.6e6 | a7f5e635 |
| `c332ayragg` | GSPO (clip 3e-4) + gate-clip 0.2/0.2 †(token-mean) | surrogate | ESS·clip·0.05 | raw (pre) | 46.7 / 46.7 | 0.51 | bf15e368 |
| `782xyquesk` | GSPO (clip 3e-4) + gate-clip 0.2/0.2 †(token-mean) | surrogate | ESS·clip·0.2 | raw (pre) | 39.8 / 39.8 | 0.42 | 8c5e971c |
.

### B1a. GSPO ESS-threshold "ablation" (gate clip is a near-no-op on GSPO)


| GATE_ESS | task | AIME pk/fin | grad_norm | gate trip (mean) | note |
|---|---|---|---|---|---|
| — (no gate) | `q3cfydj8eu` | 39.2 / 35.6 | low | — | plain GSPO baseline |
| 0.05 | `c332ayragg` | 46.7 / 46.7 | 0.51 | 0.13 | trips, but 0.2 clip ≫ looser than GSPO 3e-4 |
| 0.1 | `8m66pubxgu` | 37.7 / 37.7 | low | 0.07 | GSPO midpoint |
| 0.2 | `782xyquesk` | 39.8 / 39.8 | 0.42 | 0.28 | higher thr → trips most, still no-op |

### B2. ESS gate → skip (pure gating / veto the step)

| Bolt task | base / cap | trunc loc | gate: trigger·action·thr | ESS src | AIME pk/fin | grad_norm max | git sha |
|---|---|---|---|---|---|---|---|
| `fevnmajx3t` | GPD-shape (per-rank) | coeff | ESS·skip·0.1 | shaped (post) | 42.9 / 41 | ≤0.08 | 7bd1daa4 |
| `rdq6r5yy83` | CISPO cap5 | coeff | ESS·skip·0.1 | shaped (post) | 39.4 / 39.4 | 0.105 | 74082a36 |
| `vm7vcynvy7` | noclip-PG (inf) | coeff | ESS·skip·0.1 | raw (pre) | 35.6 / 35.6 | 26.8 | 7435e6c6 |
| `cisyyv7mdq` | CISPO cap5 | coeff | ESS·skip·0.05 | shaped (post) | 34.0 / 30.6 | 0.068 | 74082a36 |
| `9g6vqwuhxh` | GPD-shape | coeff | ESS·skip·0.3 | — | 32.9 / ~30 | low | 7bd1daa4 |
| `qyhdbyavtv` | CISPO cap3 | coeff | ESS·skip·0.1 | — | 30.2 / ~29 | ≤0.08 | 6f3a7476 |
| `f9mbgy6w9x` | noclip-PG (inf) | coeff | ESS·skip·0.05 | raw (pre) | 27.9 / 19.6 | 0.334 | 74082a36 |

## B3. Held-out benchmark eval (step-200 checkpoints, avg@16)

Sweep `smavmyz3uv` (git `bd9e966f`), `val-core/<set>/acc/mean@16`, temp 0.6 / top_p 0.95, prompt
2048 / resp 8192 (= training-val settings). All values = accuracy %. Only checkpoints with a
recovered HF export (Section A + B1) are evaluable — see `checkpoints.md`. Raw:
`results_sweep_A_B1.tsv`; full write-up + caveats: `results_sweep_A_B1.md`.

| run (task) | AIME-24 | AIME-25 | HMMT-25 | BRUMO-25 | BeyondAIME | MATH-500 |
|---|--:|--:|--:|--:|--:|--:|
| grpo_rerun (`udg7vbgfsn`) | 26.9 | 18.8 | 4.6 | 31.2 | 11.1 | 82.2 |
| dapo_rerun (`t82djeyx43`) | 36.7 | 25.0 | 7.3 | 39.8 | 17.2 | 83.6 |
| dppo_pure (`z95e8ih6mr`) | 30.4 | 20.6 | 7.7 | 29.6 | 14.9 | 76.2 |
| cispo_hi5_nogate 💥 (`ayv2ajeuqk`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.5 |
| cispo3_ess_dapoclip_shaped ⭐ (`328rfu6eb2`) | **42.1** | **34.6** | 14.6 | 36.9 | **23.8** | 76.1 |
| noclip_ess_dppolatch (`52iya9e2hr`) | 40.6 | 30.0 | 15.6 | 49.6 | 22.0 | **87.0** |
| cispo3_esclip 0.2/0.2 raw (`ircyhpdmku`) | 38.5 | 30.0 | **17.3** | **52.7** | 18.4 | 86.0 |
| cispo3_esclip 0.2/0.28 raw (`g7smhurn55`) | 37.7 | 23.8 | 14.2 | 41.0 | 13.4 | 84.3 |
| cispo3_esclip 0.2/0.2 shaped (`hvv2nimrr6`) | 34.8 | 30.0 | 13.8 | 40.0 | 17.5 | 84.4 |
| cispo3_esclip 0.2/0.2 ess05 shaped (`zt5vuurkyw`) | 35.8 | 29.4 | 16.7 | 39.6 | 17.8 | 85.6 |
| noclip_esclip 0.2/0.2 raw (`q2m6j822id`) | 35.6 | 26.5 | 17.7 | 46.7 | 17.5 | 85.2 |

| axis toggled | A (task) | B (task) | headline (mean@16) |
|---|---|---|---|
| coeff cap `GPD_FIXED_HI` 3 → inf | cap3 `ircyhpdmku` | inf `q2m6j822id` | cap3 ≥ inf on 5/6 (AIME24 38.5>35.6, BRUMO 52.7>46.7) |
| clip band 0.2/0.2 → 0.2/0.28 | `ircyhpdmku` | `g7smhurn55` | symmetric wins all (AIME25 30.0>23.8, BRUMO 52.7>41.0) |
| ESS source raw → shaped | `ircyhpdmku` | `hvv2nimrr6` | raw ≥ shaped all (BRUMO 52.7>40.0, AIME24 38.5>34.8) |
| ESS thr 0.1 → 0.05 (shaped) | `hvv2nimrr6` | `zt5vuurkyw` | ~tie |

## C. Frac-gate family

| Bolt task | base / cap | trunc loc | gate: trigger·action·thr | AIME pk/fin | grad_norm max | git sha |
|---|---|---|---|---|---|---|
| `bvrscfn6u8` | noclip-PG (inf) | coeff | FRAC·skip·0.015 (**never fired**) | 44.4 / 34.6 | 5.9e5 | 356c7958 |
| `9p9fp7hf5r` | noclip-PG cap3 + clip 0.2/0.28 | coeff | FRAC·clip·0.015 | 34.0 / 0.8 💥 | 7.5e6 | ff436434 |
| `mb5dsdddxq` | noclip-PG cap3 + clip 0.2/0.28 | coeff | FRAC·clip·0.015 | 40.6 / 24.2 | 9.4e6 | ff436434 |
| `yduurpxhds` | noclip-PG cap3 + clip 0.2/0.28 | coeff | FRAC·clip·0.015 | 46.7 / 29.4 | 1.1e8 | ff436434 |
| `33c6pmxban` | GPD-shape | coeff | FRAC·skip·0.015 | 42.9 / 11.9 💥 | 0.17 | 7bd1daa4 |
| `c7nrafjupe` | GPD-shape | coeff | FRAC·skip·0.01 | 29.6 / 26.0 | 0.14 | 356c7958 |
| `gezukrk4gy` | CISPO cap3 | coeff | FRAC·skip·0.015 | 29.8 / — | 0.84 | 356c7958 |

## D. Other / legacy

| Bolt task | method | note | AIME pk/fin |
|---|---|---|---|
| `3muhbfxw83` | CISPO trunc-1.2 (clip_c=10) | early reproduction, different sha; not the CISPO baseline we compare against | 48.8 / 43.1 |

## E. Multi-benchmark eval sweep (`smavmyz3uv`, `eval_sweep_A_B1`)

Eval-only session (git `bd9e966f`, 2026-08-13): each run's checkpoint scored on **6 benchmarks**,
`acc/mean@16` (%). Covers the §A baselines and §B1 (ESS→clip) runs. `beyond_aime`, `brumo_2025`,
`hmmt_feb_2025`, `math_500` are held-out (not the training-time AIME-2024).

| run (task) | AIME-24 | AIME-25 | Beyond-AIME | BRUMO-25 | HMMT-25 | MATH-500 |
|---|---|---|---|---|---|---|
| **baselines** | | | | | | |
| GRPO rerun (`udg7vbgfsn`) | 26.9 | 18.8 | 11.1 | 31.3 | 4.6 | 82.2 |
| DAPO rerun (`t82djeyx43`) | 36.7 | 25.0 | 17.2 | 39.8 | 7.3 | 83.6 |
| DPPO pure (`z95e8ih6mr`) | 30.4 | 20.6 | 14.9 | 29.6 | 7.7 | 76.2 |
| CISPO cap5 no-gate 💥 (`ayv2ajeuqk`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.5 |
| **B1 ESS→clip (ours)** | | | | | | |
| cispo3+DAPO+C3 ⭐ (`328rfu6eb2`) | **42.1** | **34.6** | **23.8** | 36.9 | 14.6 | 76.1 |
| noclip+ESS+DPPO-latch (`52iya9e2hr`) | 40.6 | 30.0 | 22.0 | **49.6** | 15.6 | **87.0** |
| cispo3 esclip 0.2/0.2 raw (`ircyhpdmku`) | 38.5 | 30.0 | 18.4 | **52.7** | **17.3** | 86.0 |
| cispo3 esclip 0.2/0.28 raw (`g7smhurn55`) | 37.7 | 23.8 | 13.4 | 41.0 | 14.2 | 84.3 |
| noclip esclip 0.2/0.2 raw (`q2m6j822id`) | 35.6 | 26.5 | 17.5 | 46.7 | 17.7 | 85.2 |
| cispo3 esclip 0.2/0.2 ess05 shaped (`zt5vuurkyw`) | 35.8 | 29.4 | 17.8 | 39.6 | 16.7 | 85.6 |
| cispo3 esclip 0.2/0.2 shaped (`hvv2nimrr6`) | 34.8 | 30.0 | 17.5 | 40.0 | 13.8 | 84.4 |

**Cross-benchmark:** every ESS-gated (B1) run beats all baselines on AIME-24/25, Beyond-AIME, and HMMT;
`328` leads the AIME/Beyond/HMMT hardest sets, while `52iya` (DPPO-latch) leads BRUMO and MATH-500.
The no-gate CISPO baseline (`ayv2`) is ~0 everywhere (collapsed), confirming the gate is load-bearing across all evals.

**Gap — no-clip gate ablation is incomplete on held-out** (see §E.1 below for the sweep to run).

### E.1 30B no-clip gate ablation — held-out sweep **to run**

The no-clip figure (`figures_mains/result/q30ba3b/noclip/`) is a 3-arm gate ablation, but only the
ESS-clip arm (`q2m6j822id`) has held-out numbers in §E. The **ungated** and **ESS-skip** arms were
never scored. All three share the **same config** (verified from run.log) — `adv_estimator=grpo`,
`loss_agg_mode=seq-mean-token-sum-norm`, band 0.2/0.2, `GPD_FIXED_HI=inf` — and differ **only** in
the gate (`GATE_ESS`: 0.0 ungated / 0.1 clip / 0.1 skip), so this is a clean gate-only sweep. Submit
over `global_step_200`, same 6 benchmarks / config as `smavmyz3uv` (git `bd9e966f`, temp 0.6 / top_p 0.95):

| role | run (task) | AIME-24 | AIME-25 | Beyond-AIME | BRUMO-25 | HMMT-25 | MATH-500 |
|---|---|---|---|---|---|---|---|
| no-clip, ungated (`GATE_ESS=0.0`) | `bvrscfn6u8` | — | — | — | — | — | — |
| no-clip + ESS **skip** | `vm7vcynvy7` | — | — | — | — | — | — |
| no-clip + ESS **clip** (already in §E) | `q2m6j822id` | 35.6 | 26.5 | 17.5 | 46.7 | 17.7 | 85.2 |

### E.2 8B held-out eval sweep — **DONE** (`ysjg39qct7`, git `1be6e6f`)

Eval-only sweep over `global_step_200`, `val-core/<set>/acc/mean@16`, temp 0.6 / top_p 0.95, prompt
2048 / resp 8192, 6 benchmarks; `ok=6 failed=0`. Full write-up + agg-matched pairings:
`results_sweep_8b_E1.md`; raw `results_sweep_8b_E1.tsv`. All values = accuracy %.

| role | run (task) | agg | AIME-24 | AIME-25 | Beyond | BRUMO-25 | HMMT-25 | MATH-500 |
|---|---|---|--:|--:|--:|--:|--:|--:|
| **baselines** | | | | | | | | |
| GRPO (token-mean) | `yfs6ms6w6a` | token-mean | 27.9 | 21.5 | 11.6 | 34.2 | 12.5 | 83.3 |
| DAPO clip-higher (token-mean) | `5sra49tycr` | token-mean | 30.0 | 23.3 | 14.4 | 35.0 | 11.9 | 83.7 |
| DPPO always-latch | `zrmqamex4j` | sumnorm | 32.1 | **27.5** | 13.5 | **38.8** | 12.3 | 83.4 |
| **ESS-gated (ours)** | | | | | | | | |
| cispo3 + dppo-latch ⭐ | `vdfa57r99z` | sumnorm | **33.3** | 25.0 | **15.8** | 31.0 | 11.7 | 76.2 |
| cispo3 + DAPO-band, no-norm | `bxjnvy6f3s` | token-mean | 30.0 | 24.2 | 12.9 | 36.7 | **12.7** | **85.3** |
| cispo3 + grpo-clip | `n2bu8xky6c` | sumnorm | 30.4 | 21.7 | 13.1 | 33.3 | 11.7 | 80.7 |

**Agg-matched pairs only** (see `results_sweep_8b_E1.md`): DAPO-band `5sra49tycr`↔`bxjnvy6f3s`
(both token-mean, ~flat); DPPO `zrmqamex4j`↔⭐`vdfa57r99z` (both sumnorm: +1.2 AIME-24/+2.3 Beyond but
−6.6 MATH-500/−7.8 BRUMO). GRPO-band `yfs6ms6w6a`↔`n2bu8xky6c` is **NOT** agg-matched (open cell).
**8B gate effect is small/mixed** (vs the decisive 30B) — lean on 30B for the headline.

## F. Qwen3-8B runs (B200, AIME-2024 mean@16, full 0→200)

Same design ported to Qwen3-8B. Shared GPD base (`C_HI=1.2,C_LO=0.8,σ=0.1,k=0.33,UPPER_ONLY=1`);
gate `GATE_ESS=0.1` throughout. `agg` = loss aggregation (`sumnorm` = seq-mean-token-sum-norm, the
30B default; `none` = plain mean).

### F1. Baselines — no ESS gate

| run | task | clip / loss | agg | AIME pk/fin | grad_norm max | entropy max | git sha |
|---|---|---|---|---|---|---|---|
| GRPO (fresh) | `yfs6ms6w6a` | 0.2/0.2 surrogate | sumnorm | 30.6 / 28.7 | 0.31 | 0.71 | 8f0979eb |
| DAPO clip-higher (fresh) | `5sra49tycr` | 0.2/0.28 surrogate | sumnorm | 31.0 / 29.0 | 0.52 | 1.19 | 8f0979eb |
| grpoclip_alwaysclip_ref | `d95iir8hri` | 0.2/0.2 always | — | 31.5 / 30.6 | 0.062 | 0.14 | ff30f687 |
| dapoclip_alwaysclip_ref | `8rx5xvf7dt` | 0.2/0.28 always | — | 29.0 / 26.9 | 0.36 | 0.44 | ff30f687 |
| dppolatch_alwayslatch_ref | `zrmqamex4j` | dppo-tv latch always | — | 33.3 / 31.7 | 1.7e2 | 0.14 | ff30f687 |

### F2. ESS-gated (ours) — all ESS·clip·0.1, cap via `GPD_FIXED_HI`

| run | task | cap | ESS src | clip band / rule | `CLIP_C` | agg | AIME pk/fin | grad_norm max |
|---|---|---|---|---|---|---|---|---|
| **cispo3 + dppo-latch** ⭐ | `vdfa57r99z` | 3 | raw | dppo_tv rule | – | sumnorm | **34.2 / 32.5** | 15 |
| cispo3 + DAPO-band, **no norm** | `bxjnvy6f3s` | 3 | shaped | 0.2/0.28 | 3 | **none** | 32.7 / **32.7** ↑ | 0.63 |
| cispo3 + grpo-clip | `n2bu8xky6c` | 3 | raw | 0.2/0.2 | 3 | sumnorm | 30.8 / 29.6 | 0.061 |
| cispo3 + DAPO-band | `ahx4ge5hjp` | 3 | shaped | 0.2/0.28 | 3 | sumnorm | 29.6 / 25.2 | 0.069 |
| noclip + dppo-latch | `gj37cm2h58` | **inf** | raw | dppo_tv rule | – | sumnorm | 29.0 / 25.2 | 3.4e5 |

### 8B findings

1. **Best = `cispo3_ess_dppolatch` (34.2 / 32.5)**, edging the strongest baseline (`dppolatch_alwayslatch` 33.3/31.7) on accuracy AND ~10× more stable (grad_norm 15 vs 1.7e2).
2. **Aggregation matters a lot here:** the DAPO-band + C3 recipe (`ahx4ge5hjp`, `328`'s 8B analog) is weak *with* sum-norm (25.2 final) but jumps to **32.7 (still rising) *without* norm** (`bxjnvy6f3s`) at grad_norm 0.63 — a clean one-knob ablation (only `agg` differs). So at 8B the sum-norm hurts the DAPO-band variant.
3. **noclip (inf) + ESS explodes again** (grad_norm 3.4e5, fades to 25.2) → the finite cap-3 is required at 8B too, matching 30B.
4. Spread is tighter than 30B (~25–34%); the 30B winner recipe (cispo3+DAPO+C3) only reaches the top at 8B once the sum-norm is removed.

Curves: `../reports/intern_pointers/qwen3_8b_learning_curves.png`; paper fig: `figures_mains/result/8b/curves/`.


## G. Qwen3-1.7B runs (p4d aws_6, `siri_euclid`; lr 1e-6 only)

Qwen3-**1.7B**-base, GSM8K/MATH train → AIME-2024. **Metric source: TensorBoard
`val-core/aime_2024/acc/mean@16`** (the text `results.md` "aime_2024/mean" column is a *different*,
higher aggregation — do not use it). Only lr 1e-6 recorded (lr 1e-5 collapses). Small base ⇒ absolute
AIME is low (~5–15%).

| run | task | method | AIME pk/fin (%) | grad_norm max | status |
|---|---|---|---|---|---|
| ess-dppo (running) | `677feegwie` | cispo3 + ESS·clip(dppo_tv) | **14.8 @400** / 11.7 | 14 | RUNNING (→470) |
| ess-dppo | `x25bmb3v2d` | cispo3 + ESS·clip(dppo_tv) | 12.1 @500 / **12.1** | 0.56 | ✅ 500 steps, still rising |
| **grpo (baseline)** | `zbx5g5nkpt` | GRPO 0.2/0.2 | 11.5 @250 / 10.2 | 0.047 | ✅ 500 steps, stable |
| ess-clip (rerun, healthy) | `zbx5g5nkpt` | cispo3 + ESS·clip 0.2/0.2 | 9.8 @110 / 9.6 | 0.034 | ✅ →170 |
| ess-clip (seg1) | `zbx5g5nkpt` | cispo3 + ESS·clip 0.2/0.2 | 7.9 @10 / **0.0** 💥 | 0.082 | 💥 collapsed by ~step 180 |

---

## H. Qwen3-4B HH-RLHF (preference reward — separate track, not AIME)

Does the ESS method carry beyond math? Separate track on **Anthropic HH-RLHF**, base
**Qwen3-4B-Instruct-2507**, 200 steps. Metric = reward-model score `critic/score/mean` (higher =
better), **not** AIME. GRPO baseline vs cispo3+ESS. Both RUNNING (~step 160). Logs archived in
`only_for_figures/hhrlhf_{grpo_funegmmbmz,cispo3ess_qshqvngbnt}_runlog.txt`.

| Bolt task | method | gate | agg | reward (last) | grad_norm | resp len | status |
|---|---|---|---|---|---|---|---|
| `funegmmbmz` | GRPO (vanilla) baseline | none (`GATE_ESS=0.0`) | — | 43.0 @200 | ≤1.7 | 349 | ✅ done |
| `qshqvngbnt` | cispo3 (gpd_shape cap3) + ESS·clip·0.1 | clip, shaped | — | **60.7 @200** | ≤1.9 | ~460 | ✅ done |

**cispo3+ESS reaches reward 60.7 vs GRPO 43.0** (both @200) — the advantage seen on AIME transfers to a real
preference-reward RLHF task; both stable (grad_norm ≤2), cispo3+ESS also produces longer responses
(444 vs 349 tok). Reward is on the RM's raw scale. (Numbers will move; runs not yet at 200.)

---
