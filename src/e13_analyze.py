"""E13 analysis: matched comparison on identical future records, and group-aware random split.
Pooled metrics use month weights = (flows of the month in the proportional pool) / (test flows of the month).
Intervals: hierarchical bootstrap (seeds, then circular 7-day blocks within months), 2,000 draws, paired across
conditions (same resampled seeds and days)."""
import numpy as np, pandas as pd
from common import MODELS
d = pd.read_csv("results_prop/e13_daycounts.csv")
prop = pd.read_parquet("data/luflow_prop.parquet", columns=["day", "label"])
prop = prop[prop.label.isin(["benign", "malicious"])]; nprop = prop.groupby(prop.day.str[:7]).size()
later = ["2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02"]
rng = np.random.default_rng(13); B, BL = 2000, 7


def mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(x, float) for x in (tp, fp, fn, tn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0)


def cal(g):
    g = g.set_index(pd.to_datetime(g.day))[["tp", "fp", "fn", "tn"]]
    return g.reindex(pd.date_range(g.index.min(), g.index.max()), fill_value=0).values


def blk(n):
    st = rng.integers(0, n, int(np.ceil(n / BL))); return ((st[:, None] + np.arange(BL)).ravel()[:n]) % n


def summarize(mon_counts):
    """mon_counts: array (7 months, 4) -> pooled weighted FPR, MCC, worst-month FPR, AUT(MCC)"""
    w = np.array([nprop[m] for m in later]) / mon_counts.sum(1)
    W = (mon_counts * w[:, None]).sum(0)
    f = mon_counts[:, 1] / (mon_counts[:, 1] + mon_counts[:, 3]); mm = mcc(*mon_counts.T)
    return W[1] / (W[1] + W[3]), float(mcc(*W)), f.max(), ((mm[1:] + mm[:-1]) / 2).sum() / (len(mm) - 1)


F = d[d.test == "F"].copy(); F["month"] = F.day.str[:7]
conds = ["PAST", "PAST+FUTURE", "PAST+FUTURE-G"]
rows = []
for m in [x for x in MODELS if x in F.model.unique()]:
    fm = F[F.model == m]; seeds = sorted(fm.seed.unique())
    C = {(s, c): {mo: cal(fm[(fm.seed == s) & (fm.condition == c) & (fm.month == mo)]) for mo in later} for s in seeds for c in conds}
    pt = {c: np.mean([summarize(np.array([C[(s, c)][mo].sum(0) for mo in later])) for s in seeds], 0) for c in conds}
    draws = {c: [] for c in conds}
    for b in range(B):
        ss = rng.choice(seeds, len(seeds))
        idx = {(s, mo): blk(len(C[(s, conds[0])][mo])) for s in set(ss) for mo in later}
        for c in conds:
            draws[c].append(np.mean([summarize(np.array([C[(s, c)][mo][idx[(s, mo)]].sum(0) for mo in later])) for s in ss], 0))
    D = {c: np.array(draws[c]) for c in conds}
    for c in conds:
        r = dict(model=m, condition=c, fpr=pt[c][0], mcc=pt[c][1], worst_fpr=pt[c][2], aut=pt[c][3])
        for k, j in [("fpr", 0), ("mcc", 1), ("worst_fpr", 2)]:
            r[k + "_lo"], r[k + "_hi"] = np.percentile(D[c][:, j], [2.5, 97.5])
        if c != "PAST":
            ratio = D["PAST"][:, 0] / D[c][:, 0]; dm = D[c][:, 1] - D["PAST"][:, 1]
            r["fpr_ratio_past"] = pt["PAST"][0] / pt[c][0]; r["ratio_lo"], r["ratio_hi"] = np.percentile(ratio, [2.5, 97.5])
            r["dmcc"] = pt[c][1] - pt["PAST"][1]; r["dmcc_lo"], r["dmcc_hi"] = np.percentile(dm, [2.5, 97.5])
        rows.append(r)
O = pd.DataFrame(rows); O.to_csv("results_prop/e13_summary.csv", index=False)
pd.set_option("display.width", 250); print(O.round(5).to_string())
# group-aware random split versus the ordinary random split (E12 RAND-ALL)
G = d[d.condition == "RAND-ALL-G"].groupby(["model", "seed"])[["tp", "fp", "fn", "tn"]].sum()
e12 = pd.read_csv("results_prop/e12_daycounts.csv"); R = e12[e12.test == "RAND-ALL"].groupby(["model", "seed"])[["tp", "fp", "fn", "tn"]].sum()
out = []
for (m, s), g in G.iterrows():
    r = R.loc[(m, s)]
    out.append(dict(model=m, seed=s, fpr_rand=r.fp / (r.fp + r.tn), fpr_rand_group=g.fp / (g.fp + g.tn),
                    mcc_rand=float(mcc(*r)), mcc_rand_group=float(mcc(*g))))
Q = pd.DataFrame(out).groupby("model").mean().drop(columns="seed").reindex(MODELS)
print(Q.round(5).to_string()); Q.to_csv("results_prop/e13_groupsplit.csv")
