"""Targeted coarse scan for a Raw-to-PPO transition under one fixed rollout."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale


SEED_START = 21100826


def run_curve(method, initial_weights, features, labels, rollout, long_order, batch_size, lr, eps, stride=10):
    config = base.Config(training_learning_rate=lr, ppo_epsilon=eps)
    weights = initial_weights.copy()
    rollout_weights = initial_weights.copy()
    values = [(0, base.population_value(weights, features, labels), 1.0)]
    batches = stale.chunks(long_order, batch_size)
    for update, indices in enumerate(batches, start=1):
        gradients, _, _ = base.estimate_gradients(weights, rollout, indices, config)
        weights = weights + lr * gradients[method]
        if update % stride == 0 or update == len(batches):
            values.append((update, base.population_value(weights, features, labels), base.population_rho(weights, rollout_weights, features)))
    return values


def main():
    root = Path(__file__).resolve().parents[1]
    features, labels = stale.load_training_split(root)
    _, _, eta_max = stale.global_smoothness_bound(features)
    lr = min(0.17, 0.98 * eta_max)
    eps = 0.20
    rows = []
    for init_scale in (0.05, 0.10):
        config = base.Config(initialization_scale=init_scale, training_learning_rate=lr, ppo_epsilon=eps)
        initial_weights = base.fit_initial_policy(features, labels, config)
        for batch_size in (16, 32):
            for epochs in (4, 8):
                raw_runs = []
                ppo_runs = []
                for rep in range(3):
                    rng = np.random.default_rng(SEED_START + rep)
                    rollout = base.collect_rollout(initial_weights, features, labels, np.arange(len(features)), rng.random(len(features)))
                    order = np.concatenate([rng.permutation(len(features)) for _ in range(epochs)])
                    raw_runs.append(run_curve('raw', initial_weights, features, labels, rollout, order, batch_size, lr, eps))
                    ppo_runs.append(run_curve('ppo', initial_weights, features, labels, rollout, order, batch_size, lr, eps))
                updates = [x[0] for x in raw_runs[0]]
                raw_mean = np.mean([[x[1] for x in run] for run in raw_runs], axis=0)
                ppo_mean = np.mean([[x[1] for x in run] for run in ppo_runs], axis=0)
                raw_rho = np.mean([[x[2] for x in run] for run in raw_runs], axis=0)
                ppo_rho = np.mean([[x[2] for x in run] for run in ppo_runs], axis=0)
                gap = raw_mean - ppo_mean
                early_points = max(2, len(gap)//3)
                max_early = float(np.max(gap[1:early_points]))
                max_early_index = int(np.argmax(gap[1:early_points])+1)
                persistent = -1
                for j in range(max_early_index+1, len(gap)):
                    if np.all(gap[j:] < 0):
                        persistent = int(updates[j]); break
                rows.append({
                    'initialization_scale': init_scale,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'updates': updates[-1],
                    'learning_rate': lr,
                    'eta_max': eta_max,
                    'raw_final': float(raw_mean[-1]),
                    'ppo_final': float(ppo_mean[-1]),
                    'final_raw_minus_ppo': float(gap[-1]),
                    'max_early_raw_advantage': max_early,
                    'max_early_update': updates[max_early_index],
                    'persistent_crossover_update': persistent,
                    'minimum_raw_rho': float(np.min(raw_rho)),
                    'minimum_ppo_rho': float(np.min(ppo_rho)),
                    'transition_visible': float(max_early > .003 and gap[-1] < -.003 and persistent > 0),
                })
                print(rows[-1])
    rows.sort(key=lambda r: (r['transition_visible'], r['max_early_raw_advantage']-r['final_raw_minus_ppo'], -r['minimum_raw_rho']), reverse=True)
    path = root/'simulation'/'results'/'optdigits_stale_fast_scan.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    main()
