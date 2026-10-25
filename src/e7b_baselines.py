"""E7b: generic label-free drift detectors as baselines for the E7 monitors (RF, seed 0).
Per day, from the T0-trained RF and unlabelled flows only:
  mean_score   mean malicious score of the day's flows
  ks_score     Kolmogorov-Smirnov statistic between the day's scores and the reference scores
  ph_score     two-sided Page-Hinkley statistic on the daily mean score (delta = 0.005), run from 1 Aug 2020
Supervised detectors such as DDM, EDDM or ADWIN-on-error need the error stream, i.e. labels, and are not comparable.
Scored exactly like E7: alarm = value above its August maximum; burst day = estimated false alerts > 5x August median."""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from common import *

df = load("data/luflow_sample.parquet")
T0all = df[df.month.isin(["2020-06", "2020-07"])]
train = cap(T0all, 150_000, 0); ref = T0all.drop(train.index)
m = make_model("RF", 0).fit(train[FEATURES], train.y)
ref_s = m.predict_proba(ref[FEATURES])[:, 1]
rows = []
for d, g in df[(df.day >= "2020-08-01") & (df.day < "2022-01-01")].groupby("day"):
    s = m.predict_proba(g[FEATURES])[:, 1]
    rows.append(dict(day=d, mean_score=s.mean(), ks_score=ks_2samp(s, ref_s).statistic))
r = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
x = r.mean_score.values; delta = 0.005
up = dn = 0.0; mu = 0.0; ph = []
cu = cd = 0.0; mn_u = mx_d = 0.0
for t, v in enumerate(x, 1):
    mu += (v - mu) / t
    cu += v - mu - delta; mn_u = min(mn_u, cu)
    cd += v - mu + delta; mx_d = max(mx_d, cd)
    ph.append(max(cu - mn_u, mx_d - cd))
r["ph_score"] = ph
e7 = pd.read_csv("results/e7_rf.csv")[["day", "fpr"]]
r = r.merge(e7, on="day")
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
r["false_alerts"] = r.fpr * r.day.map(cnt.benign)
r.to_csv("results/e7b_rf.csv", index=False)
cal = r[r.day < "2020-09-01"]; ev = r[r.day >= "2020-09-01"]
burst = (ev.false_alerts > 5 * cal.false_alerts.median()).astype(int)
out = []
for k in ["mean_score", "ks_score", "ph_score"]:
    al = ev[k] > cal[k].max()
    det = ev[al & (ev.day >= "2020-11-18")].day.min() if al.any() else None
    out.append(dict(monitor=k, roc_auc=roc_auc_score(burst, ev[k]), burst_days=int(burst.sum()),
                    alarm_days=int(al.sum()), alarm_tp=int((al & (burst == 1)).sum()),
                    delay=(pd.Timestamp(det) - pd.Timestamp("2020-11-18")).days if det else None,
                    before_onset=int((al & (ev.day < "2020-11-18")).sum())))
o = pd.DataFrame(out); print(o.round(3).to_string()); o.to_csv("results/e7b_scores.csv", index=False)
