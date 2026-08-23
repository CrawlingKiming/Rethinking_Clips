#!/usr/bin/env python
"""Extract clean per-metric CSVs for every run in runlog.RUNS into only_for_figures/data/.

Each `<key>.csv` has a `step` column plus one column per metric that the run actually logged
(validation `eval`/`eval_math500`, `reward`, `entropy`, `length`, `grad_norm`, `ess`, gate stats…).
These CSVs are the data the plot code reads (runlog.series reads them first). Regenerate after
adding a run or refreshing a log:  python coding/build_figure_data.py
"""
import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RUNS, METRIC, TR, ex

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "only_for_figures", "data")
os.makedirs(OUT, exist_ok=True)

built, skipped = [], []
for key, (root, fname) in RUNS.items():
    if not os.path.exists(os.path.join(TR[root], fname)):
        skipped.append((key, "source missing"))
        continue
    data = {}                                   # metric name -> {step: value}
    for mname, mkey in METRIC.items():
        d = ex(root, fname, mkey)
        if not d and mname == "ess":
            d = ex(root, fname, "actor/gate/ess_norm")   # some runs log only the unsuffixed key
        if d:
            data[mname] = d
    if not data:
        skipped.append((key, "no metrics parsed"))
        continue
    cols = [m for m in METRIC if m in data]     # keep METRIC order, only present metrics
    steps = sorted(set().union(*[set(d) for d in data.values()]))
    with open(os.path.join(OUT, f"{key}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + cols)
        for s in steps:
            w.writerow([s] + [data[m].get(s, "") for m in cols])
    built.append((key, len(steps), cols))

print(f"built {len(built)} CSVs into {OUT}")
for k, n, cols in built:
    print(f"  {k:22s} {n:4d} steps  [{', '.join(cols)}]")
if skipped:
    print("skipped:")
    for k, why in skipped:
        print(f"  {k}: {why}")
