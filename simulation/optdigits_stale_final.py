"""Final stale-rollout Optdigits validation on the selected pilot setting.

The official 3,823-image training split is rolled out exactly once. One action
is sampled per image from Q and the fixed data are consumed once in disjoint
minibatches of four. No fresh rollout is collected. Performance is evaluated on
the official 1,797-image test split.

The selected setting (initialization scale 1.20, minibatch size 4, certified
learning rate 0.98/L) was chosen on separate pilot seeds. Validation seeds choose
an ESS threshold solely by agreement with the exact-MSE oracle. A disjoint final
seed block then reports Raw, PPO, exact-MSE oracle, and the frozen ESS gate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import optdigits_categorical_theory as base
import optdigits_stale_rollout as stale
from optdigits_stale_test_scan import load_splits


RAW_COLOR = "#35618F"
PPO_COLOR = "#D27A2C"
ORACLE_COLOR = "#2F8F78"
ESS_COLOR = "#7A5195"
LIGHT_GRID = "#D9DEE8"

BATCH_SIZE = 4
INITIALIZATION_SCALE = 1.20
PPO_EPSILON = 0.20
ESS_THRESHOLDS = (0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95)
VALIDATION_SEED_START = 21800826
FINAL_SEED_START = 21900826


def se(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def full_rollout_sample_rho(weights, rollout):
    probs = base.softmax(base.logits(weights, rollout["features"]))
    selected = probs[np.arange(len(probs)), rollout["actions"]]
    ratios = selected / np.maximum(rollout["old_action_probabilities"], 1e-12)
    return float((np.sum(ratios) ** 2) / (len(ratios) * np.sum(ratios**2) + 1e-16))


def run_method(method, threshold, initial, train_x, train_y, test_x, test_y, rollout, order, lr, replication, record_oracle=False):
    cfg = base.Config(training_learning_rate=lr, ppo_epsilon=PPO_EPSILON)
    weights = initial.copy(); rollout_weights = initial.copy()
    initial_test = base.population_value(weights, test_x, test_y)
    path = []
    choices = []
    batches = stale.chunks(order, BATCH_SIZE)
    path.append({
        "replication": replication, "method": method, "update": 0,
        "test_value": initial_test, "relative_improvement": 0.0,
        "population_rho": 1.0, "sample_rho": 1.0, "selected_ppo": 0.0,
        "raw_risk": float("nan"), "ppo_risk": float("nan"),
    })
    for update, idx in enumerate(batches, 1):
        gradients, _, _ = base.estimate_gradients(weights, rollout, idx, cfg)
        sample_rho = full_rollout_sample_rho(weights, rollout)
        raw_risk = float("nan"); ppo_risk = float("nan")
        if method == "raw":
            selected = "raw"
        elif method == "ppo":
            selected = "ppo"
        elif method == "ess":
            selected = "ppo" if sample_rho < threshold else "raw"
        elif method == "oracle":
            if np.allclose(gradients["raw"], gradients["ppo"], rtol=1e-12, atol=1e-14):
                selected = "raw"
            else:
                risks = base.exact_estimator_risks(weights, rollout_weights, train_x, train_y, len(idx), cfg)
                raw_risk = float(risks["raw_risk"]); ppo_risk = float(risks["ppo_risk"])
                selected = "ppo" if ppo_risk < raw_risk else "raw"
        else:
            raise ValueError(method)
        choices.append((sample_rho, selected == "ppo"))
        weights = weights + lr * gradients[selected]
        test_value = base.population_value(weights, test_x, test_y)
        population_rho = base.population_rho(weights, rollout_weights, train_x)
        path.append({
            "replication": replication, "method": method, "update": update,
            "test_value": test_value,
            "relative_improvement": (test_value - initial_test) / max(1.0 - initial_test, 1e-12),
            "population_rho": population_rho, "sample_rho": sample_rho,
            "selected_ppo": float(selected == "ppo"),
            "raw_risk": raw_risk, "ppo_risk": ppo_risk,
        })
    return path, choices


def make_rollout(initial, train_x, train_y, seed):
    rng = np.random.default_rng(seed)
    rollout = base.collect_rollout(initial, train_x, train_y, np.arange(len(train_x)), rng.random(len(train_x)))
    order = rng.permutation(len(train_x))
    return rollout, order


def choose_threshold(initial, train_x, train_y, test_x, test_y, lr, validation_reps):
    agreement = {tau: [] for tau in ESS_THRESHOLDS}
    oracle_fraction = []
    validation_rows = []
    for rep in range(validation_reps):
        rollout, order = make_rollout(initial, train_x, train_y, VALIDATION_SEED_START + rep)
        _, choices = run_method("oracle", None, initial, train_x, train_y, test_x, test_y, rollout, order, lr, rep)
        oracle_fraction.append(np.mean([choice for _, choice in choices]))
        for tau in ESS_THRESHOLDS:
            matches = [((rho < tau) == oracle_ppo) for rho, oracle_ppo in choices]
            agreement[tau].append(np.mean(matches))
    for tau in ESS_THRESHOLDS:
        validation_rows.append({
            "threshold": tau,
            "mean_oracle_agreement": float(np.mean(agreement[tau])),
            "se_oracle_agreement": se(agreement[tau]),
            "mean_oracle_ppo_fraction": float(np.mean(oracle_fraction)),
        })
    best = max(ESS_THRESHOLDS, key=lambda tau: np.mean(agreement[tau]))
    return best, validation_rows


def aggregate(paths, methods):
    out=[]
    for method in methods:
        rows=[r for r in paths if r["method"]==method]
        updates=sorted({int(r["update"]) for r in rows})
        for update in updates:
            s=[r for r in rows if int(r["update"])==update]
            vals=np.asarray([r["test_value"] for r in s]); rel=np.asarray([r["relative_improvement"] for r in s]); rho=np.asarray([r["population_rho"] for r in s]); sr=np.asarray([r["sample_rho"] for r in s])
            out.append({"method":method,"update":update,"mean_test_value":float(np.mean(vals)),"se_test_value":se(vals),"mean_relative_improvement":float(np.mean(rel)),"se_relative_improvement":se(rel),"mean_population_rho":float(np.mean(rho)),"mean_sample_rho":float(np.mean(sr))})
    return out


def method_summary(paths, methods):
    out=[]
    reps=sorted({int(r["replication"]) for r in paths})
    for method in methods:
        finals=[]; rels=[]; minrho=[]; ppof=[]
        for rep in reps:
            s=sorted([r for r in paths if r["method"]==method and int(r["replication"])==rep], key=lambda r:r["update"])
            finals.append(s[-1]["test_value"]); rels.append(s[-1]["relative_improvement"]); minrho.append(min(r["population_rho"] for r in s)); ppof.append(np.mean([r["selected_ppo"] for r in s[1:]]))
        out.append({"method":method,"replications":len(reps),"mean_final_test_value":float(np.mean(finals)),"se_final_test_value":se(finals),"mean_final_relative_improvement":float(np.mean(rels)),"se_final_relative_improvement":se(rels),"mean_min_population_rho":float(np.mean(minrho)),"mean_ppo_fraction":float(np.mean(ppof))})
    return out


def make_figure(curve, methods, output):
    plt.rcParams.update({"font.size":9.8,"axes.titlesize":10.8,"axes.labelsize":9.8,"legend.fontsize":8.5,"xtick.labelsize":8.7,"ytick.labelsize":8.7,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":LIGHT_GRID,"grid.linewidth":0.6,"grid.alpha":0.7})
    styles={"raw":("Unmodified",RAW_COLOR,"o"),"ppo":("PPO",PPO_COLOR,"s"),"oracle":("Exact MSE oracle",ORACLE_COLOR,"D")}
    for method in methods:
        if method.startswith("ess_"): styles[method]=(f"ESS gate ({method.split('_')[1]})",ESS_COLOR,"^")
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    for method in methods:
        s=sorted([r for r in curve if r["method"]==method],key=lambda r:r["update"]); x=np.asarray([r["update"] for r in s]); y=100*np.asarray([r["mean_relative_improvement"] for r in s]); e=100*np.asarray([r["se_relative_improvement"] for r in s]); label,color,marker=styles[method]
        ax.plot(x,y,color=color,linewidth=2.1,label=label,marker=marker,markevery=max(1,len(x)//12),markersize=4.0); ax.fill_between(x,y-1.96*e,y+1.96*e,color=color,alpha=.11,linewidth=0)
    ax.set_xlabel("Stale-rollout minibatch update"); ax.set_ylabel("Relative improvement toward perfect policy (\%)"); ax.set_title("One rollout, increasingly stale updates"); ax.legend(frameon=False); fig.tight_layout(); output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(output.with_suffix('.png'),dpi=260,bbox_inches='tight'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--validation-reps',type=int,default=8); parser.add_argument('--final-reps',type=int,default=30); args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; train_x,train_y,test_x,test_y=load_splits(root); _,_,eta_max=stale.global_smoothness_bound(train_x); lr=.98*eta_max
    initial=base.fit_initial_policy(train_x,train_y,base.Config(initialization_scale=1.0))*INITIALIZATION_SCALE
    threshold,validation_rows=choose_threshold(initial,train_x,train_y,test_x,test_y,lr,args.validation_reps)
    gate=f"ess_{threshold:.2f}"; paths=[]; methods=['raw','ppo','oracle',gate]
    for rep in range(args.final_reps):
        rollout,order=make_rollout(initial,train_x,train_y,FINAL_SEED_START+rep)
        for method in methods:
            if method==gate: path,_=run_method('ess',threshold,initial,train_x,train_y,test_x,test_y,rollout,order,lr,rep)
            else: path,_=run_method(method,None,initial,train_x,train_y,test_x,test_y,rollout,order,lr,rep)
            for row in path: row['method']=method
            paths.extend(path)
    curve=aggregate(paths,methods); summary=method_summary(paths,methods); result=root/'simulation'/'results'; write_csv(result/'optdigits_stale_threshold_validation.csv',validation_rows); write_csv(result/'optdigits_stale_final_paths.csv',paths); write_csv(result/'optdigits_stale_final_curve.csv',curve); write_csv(result/'optdigits_stale_final_summary.csv',summary); make_figure(curve,methods,root/'figures'/'optdigits_stale_rollout')
    lines=[f"training_examples={len(train_x)}",f"test_examples={len(test_x)}",f"batch_size={BATCH_SIZE}",f"updates={math.ceil(len(train_x)/BATCH_SIZE)}",f"learning_rate={lr:.8f}",f"certified_eta_max={eta_max:.8f}",f"initialization_scale={INITIALIZATION_SCALE:.2f}",f"selected_ess_threshold={threshold:.2f}",f"validation_reps={args.validation_reps}",f"final_reps={args.final_reps}"]
    for row in summary:
        m=row['method']; lines += [f"{m}_final_test={row['mean_final_test_value']:.8f}",f"{m}_final_test_se={row['se_final_test_value']:.8f}",f"{m}_relative_improvement={row['mean_final_relative_improvement']:.8f}",f"{m}_ppo_fraction={row['mean_ppo_fraction']:.8f}"]
    (result/'optdigits_stale_final_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
