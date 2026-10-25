"""Kyoto KE1: random vs temporal evaluation, monthly test periods.
T0 = Nov-2006 .. Dec-2007 (first 14 months). TEMP tests every month 2008-01 .. 2015-12."""
import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import make_model, metrics, MODELS
from kyoto_common import load_kyoto

out = sys.argv[1]
seeds = list(range(int(os.environ.get("NSEEDS", "3"))))
models = os.environ.get("MODELS", ",".join(MODELS)).split(",")
NTR, NTE, NTM = 150_000, 200_000, 20_000
X, M = load_kyoto()
print("loaded", X.shape, M.y.mean(), flush=True)
t0 = M.month <= "2007-12"
months = sorted(M.month[~t0].unique())
idx_all = np.arange(len(M)); y = M.y.values


def capi(ix, n, seed):
    return ix if len(ix) <= n else np.random.default_rng(seed).choice(ix, n, replace=False)


rows = []
for seed in seeds:
    tr_all, te_all = train_test_split(idx_all, test_size=0.3, stratify=y, random_state=seed)
    tr_all, te_all = capi(tr_all, NTR, seed), capi(te_all, NTE, seed)
    i0 = idx_all[t0.values]
    tr_t0, te_t0 = train_test_split(i0, test_size=0.3, stratify=y[i0], random_state=seed)
    tr_t0c, te_t0 = capi(tr_t0, NTR, seed), capi(te_t0, NTE, seed)
    tr_temp = capi(i0, NTR, seed)
    mt = {m: capi(idx_all[(M.month == m).values], NTM, seed) for m in months}
    pooled = np.concatenate(list(mt.values()))
    for name in models:
        for proto, tr, tests in [("RAND-ALL", tr_all, {"RAND-ALL": te_all}),
                                 ("RAND-T0", tr_t0c, {"RAND-T0": te_t0}),
                                 ("TEMP", tr_temp, {**mt, "ALL-LATER": pooled})]:
            m = make_model(name, seed).fit(X.values[tr], y[tr])
            for k, te in tests.items():
                s = m.predict_proba(X.values[te])[:, 1]
                r = metrics(y[te], (s >= .5).astype(int), s)
                r["unc"] = float(((s >= .2) & (s <= .8)).mean())
                rows.append(dict(protocol=proto, test=k, model=name, seed=seed, **r))
        print(seed, name, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
