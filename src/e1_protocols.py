"""E1: random-split vs temporal-split evaluation of the same models on LUFlow.
Protocols
  RAND-ALL : stratified random 70/30 split over Jun-2020..Feb-2021 (common practice)
  RAND-T0  : stratified random 70/30 split inside the training period T0 (Jun-Jul 2020)
  TEMP     : train on T0, test on each later month (Aug-2020..Feb-2021) and on Jun-2022
Training sets are capped to the same size for all protocols so only the split differs."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import *

data, out = sys.argv[1], sys.argv[2]
policy = sys.argv[3] if len(sys.argv) > 3 else "drop"
feats = FEATURES if (len(sys.argv) <= 4 or sys.argv[4] != "noports") else \
    [f for f in FEATURES if f not in ("src_port", "dest_port")]
import os
seeds = list(range(int(os.environ.get("NSEEDS", "5"))))
NTR, NTE, NTM = 150_000, 200_000, 50_000

df = load(data, policy)
main = df[df.month.isin(MONTHS)]
T0 = main[main.month.isin(["2020-06", "2020-07"])]
later = [m for m in MONTHS if m not in ("2020-06", "2020-07")] + [FAR]
rows = []
for seed in seeds:
    tr_all, te_all = train_test_split(main, test_size=0.3, stratify=main.y, random_state=seed)
    tr_t0, te_t0 = train_test_split(T0, test_size=0.3, stratify=T0.y, random_state=seed)
    tr_all, te_all = cap(tr_all, NTR, seed), cap(te_all, NTE, seed)
    tr_t0c, te_t0 = cap(tr_t0, NTR, seed), cap(te_t0, NTE, seed)
    tr_temp = cap(T0, NTR, seed)
    temp_tests = {m: cap(df[df.month == m], NTM, seed) for m in later}
    temp_tests["ALL-LATER"] = cap(df[df.month.isin(later[:-1])], NTE, seed)
    for name in MODELS:
        r = fit_eval(name, seed, tr_all[feats], tr_all.y, {"RAND-ALL": (te_all[feats], te_all.y)})
        rows.append(dict(protocol="RAND-ALL", test="RAND-ALL", model=name, seed=seed, **r["RAND-ALL"]))
        r = fit_eval(name, seed, tr_t0c[feats], tr_t0c.y, {"RAND-T0": (te_t0[feats], te_t0.y)})
        rows.append(dict(protocol="RAND-T0", test="RAND-T0", model=name, seed=seed, **r["RAND-T0"]))
        r = fit_eval(name, seed, tr_temp[feats], tr_temp.y,
                     {k: (v[feats], v.y) for k, v in temp_tests.items()})
        for k, v in r.items():
            rows.append(dict(protocol="TEMP", test=k, model=name, seed=seed, **v))
        print(seed, name, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
