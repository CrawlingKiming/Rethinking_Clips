"""Stress test one fixed Optdigits rollout from weak or anti-classifier policies."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale

SEED_START = 21200826


def run(method, initial_weights, features, labels, rollout, order, batch_size, lr, eps, stride=10):
    config = base.Config(training_learning_rate=lr, ppo_epsilon=eps)
    weights = initial_weights.copy(); rollout_weights = initial_weights.copy()
    out=[(0, base.population_value(weights,features,labels),1.0)]
    batches=stale.chunks(order,batch_size)
    for k,idx in enumerate(batches,1):
        grads,_,_=base.estimate_gradients(weights,rollout,idx,config)
        weights=weights+lr*grads[method]
        if k%stride==0 or k==len(batches):
            out.append((k,base.population_value(weights,features,labels),base.population_rho(weights,rollout_weights,features)))
    return out


def main():
    root=Path(__file__).resolve().parents[1]
    features,labels=stale.load_training_split(root)
    _,_,eta_max=stale.global_smoothness_bound(features)
    lr=min(.17,.98*eta_max); eps=.2
    rows=[]
    # fit full classifier once, then scale it manually, including negative scales
    fit_cfg=base.Config(initialization_scale=1.0)
    fitted=base.fit_initial_policy(features,labels,fit_cfg)
    for scale in (-0.10,-0.05,0.0,0.05):
      initial=fitted*scale
      for batch in (32,64):
       for epochs in (4,8):
        raw_runs=[]; ppo_runs=[]
        for rep in range(3):
          rng=np.random.default_rng(SEED_START+rep)
          rollout=base.collect_rollout(initial,features,labels,np.arange(len(features)),rng.random(len(features)))
          order=np.concatenate([rng.permutation(len(features)) for _ in range(epochs)])
          raw_runs.append(run('raw',initial,features,labels,rollout,order,batch,lr,eps))
          ppo_runs.append(run('ppo',initial,features,labels,rollout,order,batch,lr,eps))
        updates=[x[0] for x in raw_runs[0]]
        raw=np.mean([[x[1] for x in r] for r in raw_runs],0); ppo=np.mean([[x[1] for x in r] for r in ppo_runs],0)
        rr=np.mean([[x[2] for x in r] for r in raw_runs],0); pr=np.mean([[x[2] for x in r] for r in ppo_runs],0)
        gap=raw-ppo; early=max(2,len(gap)//3); maxe=float(np.max(gap[1:early])); maxi=int(np.argmax(gap[1:early])+1)
        cross=-1
        for j in range(maxi+1,len(gap)):
          if np.all(gap[j:]<0): cross=int(updates[j]); break
        row={'initialization_scale':scale,'batch_size':batch,'epochs':epochs,'updates':updates[-1],'learning_rate':lr,'eta_max':eta_max,
             'initial_value':float(raw[0]),'raw_final':float(raw[-1]),'ppo_final':float(ppo[-1]),'final_raw_minus_ppo':float(gap[-1]),
             'max_early_raw_advantage':maxe,'max_early_update':updates[maxi],'persistent_crossover_update':cross,
             'minimum_raw_rho':float(np.min(rr)),'minimum_ppo_rho':float(np.min(pr)),
             'transition_visible':float(maxe>.003 and gap[-1]<-.003 and cross>0)}
        rows.append(row); print(row)
    rows.sort(key=lambda r:(r['transition_visible'], r['max_early_raw_advantage']-r['final_raw_minus_ppo'], -r['minimum_raw_rho']), reverse=True)
    path=root/'simulation'/'results'/'optdigits_stale_low_support_scan.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
