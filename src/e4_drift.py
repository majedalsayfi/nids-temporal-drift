"""E4: quantify drift of each later month relative to T0 (Jun-Jul 2020).
  PSI per feature (decile bins of T0), mean PSI
  Domain-classifier AUC (adversarial validation): can a model tell T0 from month m?
    computed on all flows, on benign only, and on malicious only
  Label prior (malicious share)."""
import sys, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from common import *

data, out = sys.argv[1], sys.argv[2]
N = 20_000
df = load(data)
T0 = df[df.month.isin(["2020-06", "2020-07"])]
later = [m for m in MONTHS if m not in ("2020-06", "2020-07")] + [FAR]


def psi(a, b, bins=10):
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(np.unique(a)) <= 20 or len(qs) < 3:  # discrete/degenerate feature: value-level bins
        qs = np.unique(np.concatenate([a, b]))
        pa = pd.Series(a).value_counts(normalize=True)
        pb = pd.Series(b).value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        pa, pb = pa.reindex(idx, fill_value=0).values, pb.reindex(idx, fill_value=0).values
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        pa = np.histogram(a, qs)[0] / len(a)
        pb = np.histogram(b, qs)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def dom_auc(A, B, seed=0, feats=FEATURES):
    A, B = cap(A, N, seed), cap(B, N, seed)
    X = pd.concat([A[feats], B[feats]])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=2, verbose=-1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


rows = []
for m in later:
    M = df[df.month == m]
    r = dict(month=m, mal_rate=float(M.y.mean()), t0_mal_rate=float(T0.y.mean()))
    a, b = cap(T0, 100_000, 0), cap(M, 100_000, 0)
    for f in FEATURES:
        r["psi_" + f] = psi(a[f].values, b[f].values)
    r["psi_mean"] = float(np.mean([r["psi_" + f] for f in FEATURES]))
    r["dom_auc_all"] = dom_auc(T0, M)
    r["dom_auc_benign"] = dom_auc(T0[T0.y == 0], M[M.y == 0])
    r["dom_auc_malicious"] = dom_auc(T0[T0.y == 1], M[M.y == 1])
    NP = [f for f in FEATURES if f not in ("src_port", "dest_port")]
    r["dom_auc_benign_noports"] = dom_auc(T0[T0.y == 0], M[M.y == 0], feats=NP)
    r["dom_auc_malicious_noports"] = dom_auc(T0[T0.y == 1], M[M.y == 1], feats=NP)
    rows.append(r)
    print(m, {k: round(v, 3) for k, v in r.items() if k in ("psi_mean", "dom_auc_all", "mal_rate")}, flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
