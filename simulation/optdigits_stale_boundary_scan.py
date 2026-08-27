"""Fine scan near the Raw/PPO stale-rollout transition boundary."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale
from optdigits_stale_test_scan import load_splits

SEED_START = 21600826


def run(method, initial, train_x, train_y, test_x, test_y, rollout, order, batch, lr, eps, stride=20):
    cfg = base.Config(training_learning_rate=lr, ppo_epsilon=eps)
    w = initial.copy(); q = initial.copy()
    out = [(0, base.population_value(w, test_x, test_y), 1.0)]
    batches = stale.chunks(order, batch)
    for k, idx in enumerate(batches, 1):
        g, _, _ = base.estimate_gradients(w, rollout, idx, cfg)
        w = w + lr * g[method]
        if k % stride == 0 or k == len(batches):
            out.append((k, base.population_value(w, test_x, test_y), base.population_rho(w, q, test_x)))
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    train_x, train_y, test_x, test_y = load_splits(root)
    _, _, eta_max = stale.global_smoothness_bound(train_x)
    eps = 0.20
    fitted = base.fit_initial_policy(train_x, train_y, base.Config(initialization_scale=1.0))
    rows = []
    for scale in (0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25):
        initial = fitted * scale
        for batch in (4, 8, 16):
            for epochs in (1, 2):
                for lr in (0.10, 0.14, min(0.17, 0.98 * eta_max)):
                    raws = []; ppos = []
                    for rep in range(5):
                        rng = np.random.default_rng(SEED_START + rep)
                        rollout = base.collect_rollout(
                            initial, train_x, train_y, np.arange(len(train_x)), rng.random(len(train_x))
                        )
                        order = np.concatenate([rng.permutation(len(train_x)) for _ in range(epochs)])
                        raws.append(run('raw', initial, train_x, train_y, test_x, test_y, rollout, order, batch, lr, eps))
                        ppos.append(run('ppo', initial, train_x, train_y, test_x, test_y, rollout, order, batch, lr, eps))
                    updates = [z[0] for z in raws[0]]
                    raw = np.mean([[z[1] for z in r] for r in raws], axis=0)
                    ppo = np.mean([[z[1] for z in r] for r in ppos], axis=0)
                    rr = np.mean([[z[2] for z in r] for r in raws], axis=0)
                    pr = np.mean([[z[2] for z in r] for r in ppos], axis=0)
                    gap = raw - ppo
                    # Early window is first 15 percent of recorded trajectory, at least 3 points.
                    early_end = max(3, int(np.ceil(0.15 * len(gap))))
                    early_slice = gap[1:early_end]
                    max_early = float(np.max(early_slice))
                    max_idx = int(np.argmax(early_slice) + 1)
                    cross = -1
                    for j in range(max_idx + 1, len(gap)):
                        if np.all(gap[j:] < 0.0):
                            cross = int(updates[j]); break
                    final_gap = float(gap[-1])
                    transition = float(max_early > 0.001 and final_gap < -0.002 and cross > 0)
                    rows.append({
                        'initialization_scale': scale,
                        'batch_size': batch,
                        'epochs': epochs,
                        'updates': updates[-1],
                        'learning_rate': lr,
                        'eta_max': eta_max,
                        'initial_test_value': float(raw[0]),
                        'raw_final_test': float(raw[-1]),
                        'ppo_final_test': float(ppo[-1]),
                        'final_raw_minus_ppo': final_gap,
                        'max_early_raw_advantage': max_early,
                        'max_early_update': updates[max_idx],
                        'persistent_crossover_update': cross,
                        'minimum_raw_rho': float(np.min(rr)),
                        'minimum_raw_effective_count': batch * float(np.min(rr)),
                        'minimum_ppo_rho': float(np.min(pr)),
                        'raw_peak_drop': float(np.max(raw) - raw[-1]),
                        'transition_visible': transition,
                        'score': 100.0 * max_early + 100.0 * max(0.0, -final_gap) + 2.0 * transition,
                    })
                    print(rows[-1])
    rows.sort(key=lambda r: r['score'], reverse=True)
    path = root / 'simulation' / 'results' / 'optdigits_stale_boundary_scan.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    main()
