"""E15: label-free monitors for several seeds and settings (reviewer request).
For each model and seed: train on T0 (cap 150,000, as E7/E8), reference = T0 flows not used for training.
Per day (Aug-2020 .. Feb-2021): benign count and false positives (to define burst days afterwards),
alert rate, uncertainty for three bands ([0.1,0.9], [0.2,0.8], [0.3,0.7]), port surge, mean score,
and the Kolmogorov-Smirnov statistic of the day's scores against the reference scores.
Output: results/e15_monitors.csv"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp
from common import *

models = sys.argv[1].split(","); seeds = [int(s) for s in sys.argv[2].split(",")]
df = load("data/luflow_sample.parquet")
T0all = df[df.month.isin(["2020-06", "2020-07"])]
days = sorted(df[(df.day >= "2020-08-01") & (df.day < "2022-01-01")].day.unique())
bydays = {d: g for d, g in df[df.day >= "2020-08-01"].groupby("day")}
rows = []
for name in models:
    for seed in seeds:
        train = cap(T0all, 150_000, seed); ref = T0all.drop(train.index)
        m = make_model(name, seed).fit(train[FEATURES], train.y)
        rs = m.predict_proba(ref[FEATURES])[:, 1]
        ref_port = ref[rs >= 0.5].dest_port.value_counts(normalize=True)
        for d in days:
            D = bydays[d]; s = m.predict_proba(D[FEATURES])[:, 1]; p = s >= 0.5; ben = D.y.values == 0
            dp = D[p].dest_port.value_counts(normalize=True)
            rows.append(dict(model=name, seed=seed, day=d, n=len(D), n_benign=int(ben.sum()),
                             fp=int((p & ben).sum()), tp=int((p & ~ben).sum()),
                             alert_rate=float(p.mean()),
                             unc_10_90=float(((s >= .1) & (s <= .9)).mean()),
                             unc_20_80=float(((s >= .2) & (s <= .8)).mean()),
                             unc_30_70=float(((s >= .3) & (s <= .7)).mean()),
                             port_surge=float((dp - ref_port.reindex(dp.index, fill_value=0)).max()) if len(dp) else 0.0,
                             mean_score=float(s.mean()), ks_score=float(ks_2samp(s, rs).statistic)))
        print(name, seed, flush=True)
        pd.DataFrame(rows).to_csv("results/e15_monitors.csv", index=False)
