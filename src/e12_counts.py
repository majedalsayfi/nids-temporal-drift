"""E12: per-day confusion counts for every E1 test set (same seeds, caps and models as E1).
Used for (a) volume-weighted pooled metrics from confusion counts and (b) the hierarchical
seed x weekly-block bootstrap. Output: results_prop/e12_daycounts.csv"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import *

NTR, NTE, NTM = 150_000, 200_000, 50_000
models = os.environ.get("MODELS", ",".join(MODELS)).split(",")
seeds = range(int(os.environ.get("NSEEDS", "5")))
df = load("data/luflow_prop.parquet")
main = df[df.month.isin(MONTHS)]
T0 = main[main.month.isin(["2020-06", "2020-07"])]
later = [m for m in MONTHS if m not in ("2020-06", "2020-07")] + [FAR]
out = []
for seed in seeds:
    tr_all, te_all = train_test_split(main, test_size=0.3, stratify=main.y, random_state=seed)
    tr_t0, te_t0 = train_test_split(T0, test_size=0.3, stratify=T0.y, random_state=seed)
    tr_all, te_all = cap(tr_all, NTR, seed), cap(te_all, NTE, seed)
    tr_t0c, te_t0 = cap(tr_t0, NTR, seed), cap(te_t0, NTE, seed)
    tr_temp = cap(T0, NTR, seed)
    tests = {m: cap(df[df.month == m], NTM, seed) for m in later}
    tests["ALL-LATER"] = cap(df[df.month.isin(later[:-1])], NTE, seed)
    for name in models:
        jobs = [("RAND-ALL", tr_all, {"RAND-ALL": te_all}), ("RAND-T0", tr_t0c, {"RAND-T0": te_t0}),
                ("TEMP", tr_temp, tests)]
        for proto, tr, tt in jobs:
            m = make_model(name, seed).fit(tr[FEATURES], tr.y)
            for k, t in tt.items():
                p = m.predict_proba(t[FEATURES])[:, 1] >= 0.5
                y = t.y.values == 1
                g = pd.DataFrame(dict(day=t.day.values, tp=p & y, fp=p & ~y, fn=~p & y, tn=~p & ~y))
                g = g.groupby("day").sum().reset_index()
                g["protocol"], g["test"], g["model"], g["seed"] = proto, k, name, seed
                out.append(g)
        print(seed, name, flush=True)
    pd.concat(out).to_csv("results_prop/e12_daycounts.csv", index=False)
