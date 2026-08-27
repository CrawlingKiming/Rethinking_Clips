"""Scan near-well-fit initial policies where the population gradient signal is small."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale
from optdigits_stale_test_scan import load_splits

SEED_START=21500826


def run(method,initial,train_x,train_y,test_x,test_y,rollout,order,batch,lr,eps,stride=20):
    cfg=base.Config(training_learning_rate=lr,ppo_epsilon=eps); w=initial.copy(); q=initial.copy()
    out=[(0,base.population_value(w,test_x,test_y),1.0)]
    batches=stale.chunks(order,batch)
    for k,idx in enumerate(batches,1):
        g,_,_=base.estimate_gradients(w,rollout,idx,cfg); w=w+lr*g[method]
        if k%stride==0 or k==len(batches): out.append((k,base.population_value(w,test_x,test_y),base.population_rho(w,q,test_x)))
    return out


def main():
    root=Path(__file__).resolve().parents[1]; train_x,train_y,test_x,test_y=load_splits(root)
    _,_,eta_max=stale.global_smoothness_bound(train_x); lr=min(.17,.98*eta_max); eps=.2; rows=[]
    fitted=base.fit_initial_policy(train_x,train_y,base.Config(initialization_scale=1.0))
    for scale in (.60,.80,1.00,1.25):
      initial=fitted*scale
      _,g0=base.population_value_and_gradient(initial,test_x,test_y); signal=float(np.sum(g0**2))
      for batch in (4,8,16,32):
       for epochs in (1,2,4):
        raws=[];ppos=[]
        for rep in range(4):
          rng=np.random.default_rng(SEED_START+rep); rollout=base.collect_rollout(initial,train_x,train_y,np.arange(len(train_x)),rng.random(len(train_x)))
          order=np.concatenate([rng.permutation(len(train_x)) for _ in range(epochs)])
          raws.append(run('raw',initial,train_x,train_y,test_x,test_y,rollout,order,batch,lr,eps));ppos.append(run('ppo',initial,train_x,train_y,test_x,test_y,rollout,order,batch,lr,eps))
        updates=[z[0] for z in raws[0]]; raw=np.mean([[z[1] for z in r] for r in raws],0); ppo=np.mean([[z[1] for z in r] for r in ppos],0)
        rr=np.mean([[z[2] for z in r] for r in raws],0); pr=np.mean([[z[2] for z in r] for r in ppos],0);gap=raw-ppo
        early=max(2,len(gap)//3); maxe=float(np.max(gap[1:early])); maxi=int(np.argmax(gap[1:early])+1);cross=-1
        for j in range(maxi+1,len(gap)):
          if np.all(gap[j:]<0):cross=int(updates[j]);break
        rawpeak=float(np.max(raw)); rawdrop=rawpeak-float(raw[-1]); minrho=float(np.min(rr))
        row={'initialization_scale':scale,'batch_size':batch,'epochs':epochs,'updates':updates[-1],'learning_rate':lr,'eta_max':eta_max,'initial_test_value':float(raw[0]),'initial_gradient_signal_sq':signal,
             'raw_final_test':float(raw[-1]),'ppo_final_test':float(ppo[-1]),'final_raw_minus_ppo':float(gap[-1]),'max_early_raw_advantage':maxe,'max_early_update':updates[maxi],
             'persistent_crossover_update':cross,'raw_peak_drop':rawdrop,'minimum_raw_rho':minrho,'minimum_raw_effective_count':batch*minrho,'minimum_ppo_rho':float(np.min(pr)),
             'transition_visible':float(maxe>.001 and gap[-1]<-.002 and cross>0)}
        rows.append(row);print(row)
    rows.sort(key=lambda r:(r['transition_visible'],max(0,-r['final_raw_minus_ppo'])+r['raw_peak_drop']+r['max_early_raw_advantage'], -r['minimum_raw_effective_count']),reverse=True)
    p=root/'simulation'/'results'/'optdigits_stale_low_signal_scan.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
