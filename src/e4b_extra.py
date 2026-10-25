"""E4b: proto PSI with value-level bins, and per-class domain AUC without port features."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_score
from common import *

data, out = sys.argv[1], sys.argv[2]
df = load(data)
T0 = df[df.month.isin(["2020-06", "2020-07"])]
NP = [f for f in FEATURES if f not in ("src_port", "dest_port")]


def dom_auc(A, B, feats, seed=0, N=20_000):
    A, B = cap(A, N, seed), cap(B, N, seed)
    X = pd.concat([A[feats], B[feats]])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=1, verbose=-1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def psi_discrete(a, b):
    pa = pd.Series(a).value_counts(normalize=True)
    pb = pd.Series(b).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa = np.clip(pa.reindex(idx, fill_value=0).values, 1e-4, None)
    pb = np.clip(pb.reindex(idx, fill_value=0).values, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


rows = []
for m in [x for x in MONTHS if x not in ("2020-06", "2020-07")] + [FAR]:
    M = df[df.month == m]
    r = dict(month=m,
             psi_proto_discrete=psi_discrete(cap(T0, 100_000, 0).proto.values, cap(M, 100_000, 0).proto.values),
             dom_auc_benign_noports=dom_auc(T0[T0.y == 0], M[M.y == 0], NP),
             dom_auc_malicious_noports=dom_auc(T0[T0.y == 1], M[M.y == 1], NP))
    rows.append(r)
    print(r, flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
