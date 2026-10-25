"""Volume-weighted evaluation of E8 strategies.
Per-day sampled confusion counts are scaled to the day's true benign / malicious volume,
so pooled metrics reflect the real traffic stream, not one-sample-per-day."""
import sys, numpy as np, pandas as pd

e8 = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
e8["ben_s"] = e8.fp + e8.tn; e8["mal_s"] = e8.tp + e8.fn
e8["wb"] = e8.day.map(cnt.benign) / e8.ben_s.replace(0, np.nan)
e8["wm"] = e8.day.map(cnt.malicious) / e8.mal_s.replace(0, np.nan)
for c in ["fp", "tn"]: e8[c + "_w"] = e8[c] * e8.wb.fillna(0)
for c in ["tp", "fn"]: e8[c + "_w"] = e8[c] * e8.wm.fillna(0)
EP = (e8.day >= "2020-11-18") & (e8.day <= "2021-01-07")  # port-9092 episode


def summ(g):
    tp, fp, fn, tn = g.tp_w.sum(), g.fp_w.sum(), g.fn_w.sum(), g.tn_w.sum()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    e = g[EP.loc[g.index]]
    days = g.day.nunique()
    return pd.Series(dict(fpr=fp / (fp + tn), recall=tp / (tp + fn), mcc=mcc,
                          false_alerts_per_day=fp / days,
                          episode_fpr=e.fp_w.sum() / (e.fp_w.sum() + e.tn_w.sum()),
                          episode_false_alerts_per_day=e.fp_w.sum() / max(e.day.nunique(), 1),
                          labels=g.labels_used.max(), updates=g.updates.max()))


out = e8.groupby(["model", "seed", "strategy"]).apply(summ).reset_index()
out.to_csv("results/e8_summary.csv", index=False)
agg = out.groupby(["model", "strategy"]).mean(numeric_only=True).drop(columns="seed")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(agg.sort_values(["model", "episode_fpr"]).round(5).to_string())
