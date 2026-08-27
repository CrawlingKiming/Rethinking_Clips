from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale
from optdigits_stale_test_scan import load_splits

SEED=21700826

def run(method,initial,tx,ty,vx,vy,rollout,order,lr,eps):
    cfg=base.Config(training_learning_rate=lr,ppo_epsilon=eps);w=initial.copy();out=[(0,base.population_value(w,vx,vy))]
    batches=stale.chunks(order,4)
    for k,idx in enumerate(batches,1):
        g,_,_=base.estimate_gradients(w,rollout,idx,cfg);w=w+lr*g[method]
        if k%20==0 or k==len(batches):out.append((k,base.population_value(w,vx,vy)))
    return out

def main():
    root=Path(__file__).resolve().parents[1];tx,ty,vx,vy=load_splits(root);_,_,emax=stale.global_smoothness_bound(tx)
    fitted=base.fit_initial_policy(tx,ty,base.Config(initialization_scale=1.0));rows=[]
    for scale in (1.05,1.10,1.15,1.20):
      initial=fitted*scale
      for lr in (.10,.14,min(.17,.98*emax)):
        raws=[];ppos=[]
        for rep in range(5):
          rng=np.random.default_rng(SEED+rep);roll=base.collect_rollout(initial,tx,ty,np.arange(len(tx)),rng.random(len(tx)));order=rng.permutation(len(tx))
          raws.append(run('raw',initial,tx,ty,vx,vy,roll,order,lr,.2));ppos.append(run('ppo',initial,tx,ty,vx,vy,roll,order,lr,.2))
        updates=[z[0] for z in raws[0]];r=np.mean([[z[1] for z in q] for q in raws],0);p=np.mean([[z[1] for z in q] for q in ppos],0);gap=r-p
        early=min(10,len(gap));mx=float(np.max(gap[1:early]));mi=int(np.argmax(gap[1:early])+1);cross=-1
        for j in range(mi+1,len(gap)):
          if np.all(gap[j:]<0):cross=updates[j];break
        rows.append({'scale':scale,'lr':lr,'eta_max':emax,'updates':updates[-1],'initial':float(r[0]),'raw_final':float(r[-1]),'ppo_final':float(p[-1]),'final_gap':float(gap[-1]),'max_early_raw':mx,'max_early_update':updates[mi],'crossover':cross,'transition':float(mx>.0005 and gap[-1]<-.001 and cross>0),'score':100*mx+100*max(0,-float(gap[-1]))})
        print(rows[-1])
    rows.sort(key=lambda x:(x['transition'],x['score']),reverse=True)
    with (root/'simulation'/'results'/'optdigits_stale_quick_boundary.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
