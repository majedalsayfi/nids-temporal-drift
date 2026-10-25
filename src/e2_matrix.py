"""E2: train-month x test-month matrix (single month training, every month tested).
Same-month cells use a held-out 30% split of that month."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import *

data, out = sys.argv[1], sys.argv[2]
NTR, NTE = 60_000, 30_000
seed = 0
df = load(data)
allm = MONTHS + [FAR]
parts = {}
for m in allm:
    d = df[df.month == m]
    tr, te = train_test_split(d, test_size=0.3, stratify=d.y, random_state=seed)
    parts[m] = (cap(tr, NTR, seed), cap(te, NTE, seed))
rows = []
for mi in MONTHS:
    tr = parts[mi][0]
    tests = {mj: (parts[mj][1][FEATURES], parts[mj][1].y) for mj in allm}
    for name in MODELS:
        r = fit_eval(name, seed, tr[FEATURES], tr.y, tests)
        for mj, v in r.items():
            rows.append(dict(train=mi, test=mj, model=name, **v))
        print(mi, name, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
