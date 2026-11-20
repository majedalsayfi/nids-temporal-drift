"""Kyoto KE8 summary over seeds with hierarchical bootstrap (seeds, then circular 3-month blocks)."""
import glob, numpy as np, pandas as pd
files = ["results_kyoto/ke8_s0.csv", "results_kyoto/ke8_s12.csv", "results_kyoto/ke8_x1.csv", "results_kyoto/ke8_x2.csv"]
e = pd.concat([pd.read_csv(f) for f in files if glob.glob(f)]).drop_duplicates(["model", "seed", "strategy", "month"])
rng = np.random.default_rng(21); B = 2000


def mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = map(float, (tp, fp, fn, tn)); d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


rows = []
for (m, s, st), g in e.groupby(["model", "seed", "strategy"]):
    g = g.sort_values("month"); f = g.fp / (g.fp + g.tn); y = g.month.str[:4].astype(int)
    rows.append(dict(model=m, seed=s, strategy=st, fpr=g.fp.sum() / (g.fp.sum() + g.tn.sum()), rec=g.tp.sum() / (g.tp.sum() + g.fn.sum()),
                     mcc=mcc(g.tp.sum(), g.fp.sum(), g.fn.sum(), g.tn.sum()), fpr_0812=f[y <= 2012].mean(), fpr_1315=f[y >= 2013].mean(),
                     worst=f.max(), labels=g.labels.max(), updates=g.updates.max()))
S = pd.DataFrame(rows); S.to_csv("results_kyoto/ke8_seeds_summary.csv", index=False)
out = []
for m, gm in e.groupby("model"):
    seeds = sorted(gm.seed.unique())
    A = {(s, st): g.sort_values("month")[["fp", "tn"]].values for (s, st), g in gm.groupby(["seed", "strategy"])}
    n = len(next(iter(A.values())))
    for st in sorted(gm.strategy.unique()):
        have = [s for s in seeds if (s, st) in A]
        D = []
        for b in range(B):
            ss = rng.choice(have, len(have)); fr, fr0 = [], []
            for s in ss:
                k = int(np.ceil(n / 3)); idx = ((rng.integers(0, n, k)[:, None] + np.arange(3)).ravel()[:n]) % n
                x = A[(s, st)][idx]; x0 = A[(s, "STATIC")][idx]
                fr.append(x[:, 0].sum() / x.sum()); fr0.append(x0[:, 0].sum() / x0.sum())
            D.append((np.mean(fr), 1 - np.sum(fr) / np.sum(fr0)))
        D = np.array(D); g = S[(S.model == m) & (S.strategy == st)]; g0 = S[(S.model == m) & (S.strategy == "STATIC")]
        out.append(dict(model=m, strategy=st, n_seeds=len(have), labels=g.labels.mean(), labels_min=g.labels.min(), labels_max=g.labels.max(),
                        fpr=g.fpr.mean(), fpr_lo=np.percentile(D[:, 0], 2.5), fpr_hi=np.percentile(D[:, 0], 97.5),
                        reduction=1 - g.fpr.mean() / g0.fpr.mean(), red_lo=np.percentile(D[:, 1], 2.5), red_hi=np.percentile(D[:, 1], 97.5),
                        rec=g.rec.mean(), mcc=g.mcc.mean(), fpr_0812=g.fpr_0812.mean(), fpr_1315=g.fpr_1315.mean(), worst_min=g.worst.min()))
O = pd.DataFrame(out); O.to_csv("results_kyoto/ke8_table.csv", index=False)
pd.set_option("display.width", 250); print(O.round(4).to_string())
print(S[S.seed == 0].round(4).to_string())
