"""Kyoto monitor evaluation: bad month = FPR > K x median FPR of the calibration months (Oct-Dec 2007);
alarm = monitor > max over calibration months. Spearman correlation with monthly FPR, ROC-AUC for bad months,
fraction of bad months alarmed and of good months alarmed. Hierarchical bootstrap of AUC (seeds, 3-month blocks)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
m = pd.read_csv("results_kyoto/kyoto_monitors.csv"); rng = np.random.default_rng(4)
MON = ["alert_rate", "unc_10_90", "unc_20_80", "unc_30_70", "service_surge", "mean_score", "ks_score"]
rows = []
for (mod, s), g in m.groupby(["model", "seed"]):
    cal = g[g.month < "2008-01"]; ev = g[g.month >= "2008-01"].sort_values("month")
    for K in (3, 5):
        bad = (ev.fpr > K * cal.fpr.median()).values
        for mon in MON:
            al = ev[mon].values > cal[mon].max()
            rows.append(dict(model=mod, seed=s, K=K, monitor=mon, rho=spearmanr(ev[mon], ev.fpr).statistic,
                             auc=roc_auc_score(bad, ev[mon]) if 0 < bad.sum() < len(bad) else np.nan,
                             bad_months=int(bad.sum()), hit=al[bad].mean() if bad.any() else np.nan,
                             false_alarm=al[~bad].mean() if (~bad).any() else np.nan))
R = pd.DataFrame(rows); R.to_csv("results_kyoto/kyoto_monitors_eval.csv", index=False)
pd.set_option("display.width", 220)
print(R[R.K == 5].groupby(["model", "monitor"]).agg(rho=("rho", "mean"), auc=("auc", "mean"), auc_min=("auc", "min"),
      bad=("bad_months", "mean"), hit=("hit", "mean"), fa=("false_alarm", "mean")).round(3).to_string())
print(R[R.K == 3].groupby(["model", "monitor"]).agg(auc=("auc", "mean")).round(3).unstack(0).to_string())
