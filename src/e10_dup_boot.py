"""E10: (a) exact-duplicate flows between training and test sets; (b) day-block bootstrap
confidence intervals for the random-split FPR, the worst-month FPR and their ratio.
Reproduces the E1 splits exactly (same seeds, same caps) on the volume-proportional sample."""
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import *

NTR, NTE, NTM = 150_000, 200_000, 50_000
models = os.environ.get("MODELS", ",".join(MODELS)).split(",")
seeds = range(int(os.environ.get("NSEEDS", "5")))
df = load("data/luflow_prop.parquet")
main = df[df.month.isin(MONTHS)]
T0 = main[main.month.isin(["2020-06", "2020-07"])]
later = MONTHS[2:]
key = lambda d: pd.util.hash_pandas_object(d[FEATURES], index=False).values
dup_rows, day_rows, chk = [], [], []
for seed in seeds:
    tr_all, te_all = train_test_split(main, test_size=0.3, stratify=main.y, random_state=seed)
    tr_all, te_all = cap(tr_all, NTR, seed), cap(te_all, NTE, seed)
    tr_temp = cap(T0, NTR, seed)
    tests = {m: cap(df[df.month == m], NTM, seed) for m in later}
    k_tr_all, k_tr_t0 = set(key(tr_all)), set(key(tr_temp))
    te_all = te_all.assign(dup=np.isin(key(te_all), list(k_tr_all)))
    for m in later:
        tests[m] = tests[m].assign(dup=np.isin(key(tests[m]), list(k_tr_t0)))
    for nm, t in [("RAND-ALL", te_all)] + [(m, tests[m]) for m in later]:
        dup_rows.append(dict(seed=seed, test=nm, dup_all=t.dup.mean(), dup_benign=t.dup[t.y == 0].mean(),
                             dup_mal=t.dup[t.y == 1].mean()))
    for name in models:
        m1 = make_model(name, seed).fit(tr_all[FEATURES], tr_all.y)
        m2 = make_model(name, seed).fit(tr_temp[FEATURES], tr_temp.y)
        for nm, t, mm in [("RAND-ALL", te_all, m1)] + [(m, tests[m], m2) for m in later]:
            p = mm.predict_proba(t[FEATURES])[:, 1] >= 0.5
            b = t.y.values == 0
            g = pd.DataFrame(dict(day=t.day.values, fp=p & b, ben=b, dup=t.dup.values))
            agg = g.groupby("day").agg(fp=("fp", "sum"), ben=("ben", "sum"))
            agg_u = g[~g.dup].groupby("day").agg(fp_u=("fp", "sum"), ben_u=("ben", "sum"))
            agg = agg.join(agg_u, how="left").fillna(0).reset_index()
            agg["model"], agg["seed"], agg["test"] = name, seed, nm
            day_rows.append(agg)
            chk.append(dict(model=name, seed=seed, test=nm, fpr=(p & b).sum() / b.sum(),
                            fpr_nodup=(p & b & ~t.dup.values).sum() / (b & ~t.dup.values).sum()))
        print(seed, name, flush=True)
    pd.DataFrame(dup_rows).to_csv("results_prop/e10_dup.csv", index=False)
    pd.concat(day_rows).to_csv("results_prop/e10_daily.csv", index=False)
    pd.DataFrame(chk).to_csv("results_prop/e10_check.csv", index=False)
