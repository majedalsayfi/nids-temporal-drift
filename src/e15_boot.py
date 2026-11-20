"""Hierarchical bootstrap intervals for monitor ROC-AUC (burst days, K=5): seeds, then circular 7-day blocks
of evaluation days (Sep-2020..Feb-2021). Output: results/e15_auc_ci.csv"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
m = pd.read_csv("results/e15_monitors.csv")
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
m["est_fa"] = m.fp / m.n_benign.clip(lower=1) * m.day.map(cnt.benign)
rng = np.random.default_rng(3); B = 1000
MON = ["alert_rate", "unc_20_80", "unc_30_70", "port_surge", "ks_score"]
out = []
for model, gm in m.groupby("model"):
    if model in ("DT", "MLP"): pass
    D = {}
    for s, g in gm.groupby("seed"):
        g = g.sort_values("day"); cal = g[g.day < "2020-09-01"]; ev = g[g.day >= "2020-09-01"]
        D[s] = (ev, (ev.est_fa > 5 * cal.est_fa.median()).values.astype(int))
    seeds = sorted(D)
    for mon in MON:
        pt = np.mean([roc_auc_score(D[s][1], D[s][0][mon]) for s in seeds])
        draws = []
        for b in range(B):
            vals = []
            for s in rng.choice(seeds, len(seeds)):
                ev, y = D[s]; n = len(y); st = rng.integers(0, n, int(np.ceil(n / 7)))
                idx = ((st[:, None] + np.arange(7)).ravel()[:n]) % n
                if y[idx].min() == y[idx].max(): continue
                vals.append(roc_auc_score(y[idx], ev[mon].values[idx]))
            if vals: draws.append(np.mean(vals))
        out.append(dict(model=model, monitor=mon, auc=pt, lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5)))
O = pd.DataFrame(out); O.to_csv("results/e15_auc_ci.csv", index=False); print(O.round(3).to_string())
