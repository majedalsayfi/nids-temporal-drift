"""E3: update strategies under temporal evaluation, tested on Sep-2020..Feb-2021.
  STATIC    : trained once on T0 (Jun-Jul 2020)
  SLIDING   : retrained each month on the previous month only
  EXPANDING : retrained each month on all previous months (equal-size cap)"""
import sys, numpy as np, pandas as pd
from common import *

data, out = sys.argv[1], sys.argv[2]
seeds = [0, 1, 2]
NTR, NTM = 150_000, 50_000
df = load(data)
tests = MONTHS[3:]  # Sep-2020 .. Feb-2021
rows = []
for seed in seeds:
    for name in MODELS:
        static = make_model(name, seed)
        T0 = cap(df[df.month.isin(["2020-06", "2020-07"])], NTR, seed)
        static.fit(T0[FEATURES], T0.y)
        for m in tests:
            idx = MONTHS.index(m)
            te = cap(df[df.month == m], NTM, seed)
            prev = MONTHS[idx - 1]
            sl = cap(df[df.month == prev], NTR, seed)
            ex = cap(df[df.month.isin(MONTHS[:idx])], NTR, seed)
            for strat, model in [("STATIC", static), ("SLIDING", None), ("EXPANDING", None)]:
                if model is None:
                    model = make_model(name, seed)
                    tr = sl if strat == "SLIDING" else ex
                    model.fit(tr[FEATURES], tr.y)
                s = model.predict_proba(te[FEATURES])[:, 1]
                rows.append(dict(strategy=strat, model=name, seed=seed, test=m,
                                 **metrics(te.y.values, (s >= .5).astype(int), s)))
        print(seed, name, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
