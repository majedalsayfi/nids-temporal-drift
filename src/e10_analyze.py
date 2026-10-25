"""Summarise E10: duplicate rates, FPR on non-duplicate flows, day-block bootstrap CIs (2,000 resamples)."""
import numpy as np, pandas as pd
from common import MODELS
B = 2000
rng = np.random.default_rng(2026)
d = pd.read_csv("results_prop/e10_daily.csv")
dup = pd.read_csv("results_prop/e10_dup.csv")
later = ["2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02"]
pd.set_option("display.width", 200)
print("Duplicate share of test flows that also occur in the training set (mean over seeds)")
print(dup.groupby("test")[["dup_all", "dup_benign", "dup_mal"]].mean().round(4).to_string())


def boot_sum(g, col_fp="fp", col_b="ben"):
    fp, bn = g[col_fp].values, g[col_b].values
    idx = rng.integers(0, len(fp), size=(B, len(fp)))
    return fp[idx].sum(1) / np.maximum(bn[idx].sum(1), 1)


rows = []
for m in [x for x in MODELS if x in d.model.unique()]:
    dm = d[d.model == m]; seeds = sorted(dm.seed.unique())
    rand_pt, worst_pt, rand_bs, worst_bs, ratio_seed = [], [], [], [], []
    rand_u = []
    for s in seeds:
        ds = dm[dm.seed == s]
        r = ds[ds.test == "RAND-ALL"]; rand_pt.append(r.fp.sum() / r.ben.sum()); rand_bs.append(boot_sum(r))
        rand_u.append(r.fp_u.sum() / r.ben_u.sum())
        mon = {k: ds[ds.test == k] for k in later}
        w = max(g.fp.sum() / g.ben.sum() for g in mon.values()); worst_pt.append(w)
        worst_bs.append(np.max(np.vstack([boot_sum(g) for g in mon.values()]), axis=0))
        ratio_seed.append(w / rand_pt[-1])
    rb, wb = np.mean(rand_bs, 0), np.mean(worst_bs, 0)
    ratio_b = wb / rb
    q = lambda a: np.percentile(a, [2.5, 97.5])
    rows.append(dict(model=m, fpr_rand=np.mean(rand_pt) * 100, rand_lo=q(rb)[0] * 100, rand_hi=q(rb)[1] * 100,
                     fpr_rand_nodup=np.mean(rand_u) * 100,
                     fpr_worst=np.mean(worst_pt) * 100, worst_lo=q(wb)[0] * 100, worst_hi=q(wb)[1] * 100,
                     ratio=np.mean(worst_pt) / np.mean(rand_pt), ratio_lo=q(ratio_b)[0], ratio_hi=q(ratio_b)[1],
                     ratio_seed_min=min(ratio_seed), ratio_seed_max=max(ratio_seed)))
r = pd.DataFrame(rows)
print(r.round(3).to_string()); r.to_csv("results_prop/e10_summary.csv", index=False)
