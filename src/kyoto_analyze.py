import numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option("display.width", 250)
d = pd.read_csv("results_kyoto/ke1.csv")
O = ["LR", "DT", "RF", "XGB", "LGBM", "MLP"]
t = d[(d.protocol == "TEMP") & (d.test != "ALL-LATER")].copy(); t["year"] = t.test.str[:4]
months = sorted(t.test.unique())
def aut(v): v = np.asarray(v, float); return float(np.sum((v[1:] + v[:-1]) / 2) / (len(v) - 1))
rows = []
for (m, s), g in t.groupby(["model", "seed"]):
    g = g.set_index("test").loc[months]
    ra = d[(d.protocol == "RAND-ALL") & (d.model == m) & (d.seed == s)].iloc[0]
    r0 = d[(d.protocol == "RAND-T0") & (d.model == m) & (d.seed == s)].iloc[0]
    pl = d[(d.test == "ALL-LATER") & (d.model == m) & (d.seed == s)].iloc[0]
    e = g[g.index < "2013-01"]
    rows.append(dict(model=m, seed=s, rand_mcc=ra.mcc, t0_mcc=r0.mcc, pool_mcc=pl.mcc, aut_mcc=aut(g.mcc),
                     rand_fpr=ra.fpr, t0_fpr=r0.fpr, pool_fpr=pl.fpr, worst_fpr=g.fpr.max(), med_fpr=g.fpr.median(),
                     fpr_0812=e.fpr.mean(), fpr_1415=g[g.index >= "2014-01"].fpr.mean(),
                     rand_rec=ra.rec, pool_rec=pl.rec, min_rec_year=g.groupby(g.index.str[:4]).rec.mean().min(),
                     rand_f1=ra.f1, aut_f1=aut(g.f1)))
s = pd.DataFrame(rows); s.to_csv("results_kyoto/ke1_perseed.csv", index=False)
g = s.groupby("model").mean(numeric_only=True).drop(columns="seed").reindex(O)
print(g.round(4).to_string())
print("std max", s.groupby("model").std(numeric_only=True).drop(columns="seed").max().round(4).to_dict())
for a, b in [("rand_mcc", "aut_mcc"), ("rand_f1", "aut_f1")]:
    w = wilcoxon(s[a], s[b]); print(a, b, "n", len(s), "all>", (s[a] > s[b]).sum(), "median diff", round(np.median(s[a] - s[b]), 4), "p", f"{w.pvalue:.2e}")
w = wilcoxon(s.rand_fpr, s.fpr_0812); print("FPR 2008-12 vs rand: pairs higher", (s.fpr_0812 > s.rand_fpr).sum(), "median ratio", round(np.median(s.fpr_0812 / s.rand_fpr), 2), "p", f"{w.pvalue:.2e}")
print("rand vs t0 mcc median diff", round(np.median(s.rand_mcc - s.t0_mcc), 4), (s.t0_mcc > s.rand_mcc).sum())
y = t.groupby(["year", "model"])[["fpr", "rec"]].mean().unstack()
print((y["fpr"][O] * 100).round(1).to_string()); print(y["rec"][O].round(3).to_string())
