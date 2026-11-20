"""E12 analysis.
(1) Pooled TEMP metrics from volume-weighted confusion counts versus the uniform ALL-LATER sample and
    versus the unweighted mean of monthly metrics.
(2) Hierarchical bootstrap (2,000 draws): seeds resampled with replacement, then days resampled in
    circular blocks of 7 consecutive calendar days within each test month (RAND-ALL: within its test period).
    Worst month is re-selected inside every draw. Reports 95% percentile intervals for RAND-ALL FPR,
    worst-month FPR, their ratio, RAND-ALL MCC, AUT(MCC) and the gap RAND-ALL MCC - AUT(MCC)."""
import numpy as np, pandas as pd
from common import MODELS
B, BL = 2000, 7
rng = np.random.default_rng(7)
d = pd.read_csv("results_prop/e12_daycounts.csv")
prop = pd.read_parquet("data/luflow_prop.parquet", columns=["day", "label"])
prop = prop[prop.label.isin(["benign", "malicious"])]
prop["month"] = prop.day.str[:7]
later = ["2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02"]


def mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def aut(v):  # v: (..., 7) monthly values -> trapezoid normalised
    return (v[..., 1:] + v[..., :-1]).sum(-1) / 2 / (v.shape[-1] - 1)


# ---------- (1) weighted pooled ----------
nprop = prop.groupby("month").size()
rows = []
for (m, s), g in d[(d.protocol == "TEMP")].groupby(["model", "seed"]):
    mon = g[g.test.isin(later)].groupby("test")[["tp", "fp", "fn", "tn"]].sum()
    ntest = mon.sum(1)
    w = (nprop.reindex(mon.index) / ntest)  # relative weights (inverse inclusion probability up to the 1% factor)
    W = mon.mul(w, axis=0).sum()
    A = g[g.test == "ALL-LATER"][["tp", "fp", "fn", "tn"]].sum()
    permon = dict(fpr=mon.fp / (mon.fp + mon.tn), rec=mon.tp / (mon.tp + mon.fn),
                  mcc=pd.Series(mcc(mon.tp.values, mon.fp.values, mon.fn.values, mon.tn.values), index=mon.index))
    rows.append(dict(model=m, seed=s,
                     fpr_weighted=W.fp / (W.fp + W.tn), fpr_alllater=A.fp / (A.fp + A.tn), fpr_meanmonths=permon["fpr"].mean(),
                     rec_weighted=W.tp / (W.tp + W.fn), rec_alllater=A.tp / (A.tp + A.fn), rec_meanmonths=permon["rec"].mean(),
                     mcc_weighted=float(mcc(W.tp, W.fp, W.fn, W.tn)), mcc_alllater=float(mcc(A.tp, A.fp, A.fn, A.tn)),
                     mcc_meanmonths=permon["mcc"].mean()))
P = pd.DataFrame(rows)
pw = P.groupby("model").mean(numeric_only=True).drop(columns="seed").reindex(MODELS)
pd.set_option("display.width", 220)
print("Pooled TEMP metrics (mean over seeds)"); print(pw.round(5).to_string())
pw.to_csv("results_prop/e12_pooled.csv")


# ---------- (2) hierarchical bootstrap ----------
def block_idx(n, rng):
    """circular 7-day block bootstrap indices for a series of n consecutive days"""
    k = int(np.ceil(n / BL)); starts = rng.integers(0, n, k)
    return (starts[:, None] + np.arange(BL)[None, :]).ravel()[:n] % n


def calendar(g):
    """day-indexed counts on a full calendar (missing capture days carry zero counts)"""
    g = g.set_index(pd.to_datetime(g.day))[["tp", "fp", "fn", "tn"]]
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    return g.reindex(full, fill_value=0).values


out = []
seeds = sorted(d.seed.unique())
for m in [x for x in MODELS if x in d.model.unique()]:
    dm = d[d.model == m]
    R = {s: calendar(dm[(dm.seed == s) & (dm.test == "RAND-ALL")]) for s in seeds}
    T = {s: {mo: calendar(dm[(dm.seed == s) & (dm.test == mo)]) for mo in later} for s in seeds}
    stats = []
    for b in range(B):
        ss = rng.choice(seeds, len(seeds), replace=True)
        fr, fw, mr, au = [], [], [], []
        for s in ss:
            r = R[s][block_idx(len(R[s]), rng)].sum(0)
            fr.append(r[1] / (r[1] + r[3])); mr.append(float(mcc(*r)))
            mon = np.array([T[s][mo][block_idx(len(T[s][mo]), rng)].sum(0) for mo in later])
            fpr_m = mon[:, 1] / (mon[:, 1] + mon[:, 3]); fw.append(fpr_m.max())
            au.append(aut(mcc(mon[:, 0], mon[:, 1], mon[:, 2], mon[:, 3])))
        fr, fw, mr, au = map(np.mean, (fr, fw, mr, au))
        stats.append((fr, fw, fw / fr, mr, au, mr - au))
    S = np.array(stats)
    pt = []  # point estimates on the original data
    for s in seeds:
        r = R[s].sum(0); mon = np.array([T[s][mo].sum(0) for mo in later])
        f = mon[:, 1] / (mon[:, 1] + mon[:, 3])
        pt.append((r[1] / (r[1] + r[3]), f.max(), float(mcc(*r)), aut(mcc(*mon.T))))
    pt = np.mean(pt, 0)
    lo, hi = np.percentile(S, 2.5, 0), np.percentile(S, 97.5, 0)
    out.append(dict(model=m, fpr_rand=pt[0], fpr_rand_lo=lo[0], fpr_rand_hi=hi[0], fpr_worst=pt[1], fpr_worst_lo=lo[1],
                    fpr_worst_hi=hi[1], ratio=pt[1] / pt[0], ratio_lo=lo[2], ratio_hi=hi[2], mcc_rand=pt[2],
                    mcc_rand_lo=lo[3], mcc_rand_hi=hi[3], aut=pt[3], aut_lo=lo[4], aut_hi=hi[4],
                    gap=pt[2] - pt[3], gap_lo=lo[5], gap_hi=hi[5]))
    print(m, flush=True)
O = pd.DataFrame(out); print(O.round(5).to_string()); O.to_csv("results_prop/e12_hboot.csv", index=False)
