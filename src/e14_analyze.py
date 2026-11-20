"""E14 analysis: volume-scaled metrics, label accounting and hierarchical bootstrap intervals.
Label accounting: initial = 150,000 labelled T0 flows (reused by every retrain), new = flows labelled during the
stream (counted once), reuse = labelled flows that enter later retrains again (not new cost).
Bootstrap: 2,000 draws; seeds resampled with replacement, then days resampled in circular 7-day blocks
within the episode (18 Nov 2020 - 7 Jan 2021) and within the whole stream. Reported: mean episode false alerts/day,
reduction relative to STATIC (paired by seed and resampled days) and stream FPR, with 95% percentile intervals."""
import sys, numpy as np, pandas as pd
files = sys.argv[1:]
e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["model", "seed", "strategy", "day"])
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
e["wb"] = e.day.map(cnt.benign) / (e.fp + e.tn).replace(0, np.nan)
e["wm"] = e.day.map(cnt.malicious) / (e.tp + e.fn).replace(0, np.nan)
e["FP"] = e.fp * e.wb.fillna(0); e["TN"] = e.tn * e.wb.fillna(0)
e["TP"] = e.tp * e.wm.fillna(0); e["FN"] = e.fn * e.wm.fillna(0)
EP = (e.day >= "2020-11-18") & (e.day <= "2021-01-07")
rng = np.random.default_rng(11); B, BL = 2000, 7


def mcc(tp, fp, fn, tn):
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (tp * tn - fp * fn) / den if den > 0 else 0


rows = []
for (m, s, st), g in e.groupby(["model", "seed", "strategy"]):
    tp, fp, fn, tn = g.TP.sum(), g.FP.sum(), g.FN.sum(), g.TN.sum(); ge = g[EP.loc[g.index]]
    rows.append(dict(model=m, seed=s, strategy=st, fpr=fp / (fp + tn), recall=tp / (tp + fn), mcc=mcc(tp, fp, fn, tn),
                     fa_day=fp / g.day.nunique(), episode_fa_day=ge.FP.sum() / ge.day.nunique(),
                     new_labels=g.new_labels.max(), updates=g.updates.max()))
S = pd.DataFrame(rows); S.to_csv(files[0].replace(".csv", "_summary.csv"), index=False)


def blocks(n):
    k = int(np.ceil(n / BL)); st = rng.integers(0, n, k)
    return (st[:, None] + np.arange(BL)).ravel()[:n] % n


out = []
for m, gm in e.groupby("model"):
    seeds = sorted(gm.seed.unique())
    arr = {}
    for (s, st), g in gm.groupby(["seed", "strategy"]):
        g = g.sort_values("day"); arr[(s, st)] = (g.FP.values, g.TN.values, EP.loc[g.index].values)
    strats = sorted(gm.strategy.unique())
    nd = len(next(iter(arr.values()))[0]); ep_idx = np.where(next(iter(arr.values()))[2])[0]
    draws = {st: [] for st in strats}
    for b in range(B):
        ss = rng.choice(seeds, len(seeds)); ie = ep_idx[blocks(len(ep_idx))]; ia = blocks(nd)
        for st in strats:
            fa, fa0, fr = [], [], []
            for s in ss:
                if (s, st) not in arr: continue
                FP, TN, _ = arr[(s, st)]; FP0 = arr[(s, "STATIC")][0]
                fa.append(FP[ie].mean()); fa0.append(FP0[ie].mean())
                fr.append(FP[ia].sum() / (FP[ia].sum() + TN[ia].sum()))
            draws[st].append((np.mean(fa), 1 - np.sum(fa) / max(np.sum(fa0), 1e-9), np.mean(fr)))
    for st in strats:
        D = np.array(draws[st]); g = S[(S.model == m) & (S.strategy == st)]
        g0 = S[(S.model == m) & (S.strategy == "STATIC")].set_index("seed").episode_fa_day
        red_pt = 1 - g.episode_fa_day.mean() / g0.mean()
        out.append(dict(model=m, strategy=st, n_seeds=len(g), new_labels=g.new_labels.mean(),
                        new_labels_min=g.new_labels.min(), new_labels_max=g.new_labels.max(), updates=g.updates.mean(),
                        fpr=g.fpr.mean(), fpr_lo=np.percentile(D[:, 2], 2.5), fpr_hi=np.percentile(D[:, 2], 97.5),
                        recall=g.recall.mean(), mcc=g.mcc.mean(), fa_day=g.fa_day.mean(),
                        episode_fa=g.episode_fa_day.mean(), episode_fa_lo=np.percentile(D[:, 0], 2.5),
                        episode_fa_hi=np.percentile(D[:, 0], 97.5), episode_fa_min=g.episode_fa_day.min(),
                        episode_fa_max=g.episode_fa_day.max(), reduction=red_pt,
                        reduction_lo=np.percentile(D[:, 1], 2.5), reduction_hi=np.percentile(D[:, 1], 97.5)))
O = pd.DataFrame(out); O.to_csv(files[0].replace(".csv", "_table.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
print(O.round(4).to_string())
