"""E16: targeted tests of the port explanations (reviewer request).
Setup as E1 TEMP: train on T0 (cap 150,000), test on 50,000 flows of every later month; seeds 0-2.
Feature variants (applied to training and test data):
  RAW        the 11 features of the main study
  EPH-GROUP  every source or destination port in the Linux ephemeral range 32768-60999 is replaced by 32768;
             all other port values are kept, so service ports still carry their information
Test-time perturbations of the RAW model (all other features unchanged):
  EPH->T0    benign test flows with an ephemeral destination port receive a destination port drawn from the
             ephemeral destination ports of benign T0 training flows
  9092->T0   benign test flows to destination port 9092 receive a destination port drawn from the destination
             ports of benign T0 training flows
Output: results_prop/e16_ports.csv (per model, seed, variant, month: FPR overall, FPR on ephemeral-port benign
flows, FPR on port-9092 benign flows, recall, MCC)"""
import os, numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef
from common import *

models = os.environ.get("MODELS", "DT,RF,XGB,LGBM").split(",")
seeds = range(int(os.environ.get("NSEEDS", "3")))
LO, HI = 32768, 60999
df = load("data/luflow_prop.parquet")
T0 = df[df.month.isin(["2020-06", "2020-07"])]
later = MONTHS[2:]
eph = lambda v: (v >= LO) & (v <= HI)


def group(X):
    X = X.copy()
    for c in ("src_port", "dest_port"):
        X.loc[eph(X[c]), c] = LO
    return X


def score(m, X, t, label):
    p = m.predict_proba(X)[:, 1] >= 0.5; y = t.y.values; b = y == 0
    e = b & eph(t.dest_port.values); k = b & (t.dest_port.values == 9092)
    return dict(fpr=(p & b).sum() / b.sum(), fpr_eph=(p & e).sum() / max(e.sum(), 1), n_eph=int(e.sum()),
                fpr_9092=(p & k).sum() / max(k.sum(), 1), n_9092=int(k.sum()),
                rec=(p & ~b).sum() / max((~b).sum(), 1), mcc=matthews_corrcoef(y, p), variant=label)


rows = []
for seed in seeds:
    tr = cap(T0, 150_000, seed); rng = np.random.default_rng(seed)
    tests = {m: cap(df[df.month == m], 50_000, seed) for m in later}
    trb = tr[tr.y == 0]
    t0_eph_ports = trb.dest_port[eph(trb.dest_port)].values
    t0_ports = trb.dest_port.values
    for name in models:
        raw = make_model(name, seed).fit(tr[FEATURES], tr.y)
        grp = make_model(name, seed).fit(group(tr[FEATURES]), tr.y)
        for mth, t in tests.items():
            X = t[FEATURES]
            out = [score(raw, X, t, "RAW"), score(grp, group(X), t, "EPH-GROUP")]
            b = t.y.values == 0
            Xp = X.copy(); sel = b & eph(t.dest_port.values)
            Xp.loc[sel, "dest_port"] = rng.choice(t0_eph_ports, sel.sum())
            out.append(score(raw, Xp, t, "EPH->T0"))
            Xq = X.copy(); sel = b & (t.dest_port.values == 9092)
            if sel.any():
                Xq.loc[sel, "dest_port"] = rng.choice(t0_ports, sel.sum())
            out.append(score(raw, Xq, t, "9092->T0"))
            for o in out:
                rows.append(dict(model=name, seed=seed, month=mth, **o))
        print(seed, name, flush=True)
        pd.DataFrame(rows).to_csv("results_prop/e16_ports.csv", index=False)
