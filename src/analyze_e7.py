"""Score label-free monitors as early warning of FPR-burst days."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

files = sys.argv[1:]
d = pd.concat([pd.read_csv(f) for f in files])
d = d.groupby(["model", "day"]).first().reset_index()  # merge dom_* (RF file) with cheap monitors
MON = ["alert_rate", "uncertainty", "internal_alert_share", "port_surge", "dom_all", "dom_alerts", "dom_nonalerts"]
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
d["false_alerts"] = d.fpr * d.day.map(cnt.benign)
rows = []
for model, g in d.groupby("model"):
    g = g.copy(); g[[c for c in MON if c in g]] = g[[c for c in MON if c in g]].fillna(g[[c for c in MON if c in g]].median())
    cal = g[g.day < "2020-09-01"]; ev = g[(g.day >= "2020-09-01") & (g.day < "2022-01-01")]
    base = cal.false_alerts.median()
    for K in [5, 10]:
        tau = K
        burst = (ev.false_alerts > K * base).astype(int)
        first_burst = ev.day[burst == 1].min() if burst.any() else None
        for k in MON:
            if k not in g.columns or g[k].isna().all():
                continue
            thr = cal[k].max()
            al = ev[k] > thr
            tp = int((al & (burst == 1)).sum()); fp = int((al & (burst == 0)).sum()); fn = int((~al & (burst == 1)).sum())
            first_alarm = ev.day[al].min() if al.any() else None
            # first alarm on or before the first burst day, within the 14 preceding days
            pre = ev[(al) & (ev.day <= first_burst)] if first_burst else ev.iloc[0:0]
            lead = (pd.Timestamp(first_burst) - pd.Timestamp(pre.day.max())).days if len(pre) and first_burst else np.nan
            onset = "2020-11-18"
            det = ev[(al) & (ev.day >= onset)].day.min() if al.any() else None
            delay = (pd.Timestamp(det) - pd.Timestamp(onset)).days if det else np.nan
            fa_pre = int((al & (ev.day < onset)).sum())
            rows.append(dict(model=model, K=K, base_false_alerts=base, detect_day=det, delay_days=delay, alarms_before_onset=fa_pre, monitor=k, n_days=len(ev), burst_days=int(burst.sum()),
                             roc_auc=roc_auc_score(burst, ev[k]) if 0 < burst.sum() < len(burst) else np.nan,
                             ap=average_precision_score(burst, ev[k]) if burst.sum() else np.nan,
                             base_rate=float(burst.mean()),
                             spearman_fa=spearmanr(ev[k], ev.false_alerts).statistic,
                             alarm_days=int(al.sum()), alarm_tp=tp, alarm_fp=fp, alarm_recall=tp / max(tp + fn, 1),
                             alarm_precision=tp / max(tp + fp, 1), first_burst=first_burst, first_alarm=first_alarm,
                             cal_threshold=thr))
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.drop(columns=["first_alarm", "first_burst", "cal_threshold", "n_days"]).round(3).to_string())
r.to_csv("results/e7_scores.csv", index=False)
