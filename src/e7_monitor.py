"""E7: label-free daily drift monitors as early warning of false-positive bursts.

Model trained on T0 (Jun-Jul 2020). Reference data = T0 flows NOT used for training.
Calibration period = Aug 2020 (alarm thresholds = max monitor value over Aug days).
Evaluation period = Sep 2020 .. Feb 2021. Labels are used only to score the monitors.

Monitors (computed from one day's flows, no labels):
  alert_rate     share of flows predicted malicious
  uncertainty    share of flows with score in [0.2, 0.8]
  port_surge     max over destination ports of (day alert share - reference alert share)
  dom_all        domain-classifier AUC, day vs reference, all flows, port-free features
  dom_alerts     same, restricted to predicted-malicious flows
  dom_nonalerts  same, restricted to predicted-benign flows
  internal_alert_share  share of alerts whose source is inside the home network (AS 786)
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_score
from common import *

data, out = sys.argv[1], sys.argv[2]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["RF"]
full_dom = len(sys.argv) > 4 and sys.argv[4] == "dom"
NP = [f for f in FEATURES if f not in ("src_port", "dest_port")]
ND = 5000
df = load(data)
df["src_ip_as"] = pd.to_numeric(df["src_ip"], errors="coerce") if "src_ip" in df.columns else np.nan
HOME = 786  # home AS of the monitored network (known to the operator; not used as a model feature)
T0all = df[df.month.isin(["2020-06", "2020-07"])]
train = cap(T0all, 150_000, 0)
ref = T0all.drop(train.index)
days = sorted(df[(df.day >= "2020-08-01") & (df.day < "2022-01-01")].day.unique())


def dom_auc(A, B, seed=0):
    if len(A) < 200 or len(B) < 200:
        return np.nan
    A, B = cap(A, ND, seed), cap(B, ND, seed)
    X = pd.concat([A[NP], B[NP]]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=2, verbose=-1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())


rows = []
for name in models:
    m = make_model(name, 0).fit(train[FEATURES], train.y)
    rs = m.predict_proba(ref[FEATURES])[:, 1]
    ref_alerts = ref[rs >= 0.5]
    ref_port = ref_alerts.dest_port.value_counts(normalize=True)
    ref_dom_all = cap(ref, ND, 1)
    for d in days:
        D = df[df.day == d]
        s = m.predict_proba(D[FEATURES])[:, 1]
        p = s >= 0.5
        ben = D.y.values == 0
        r = dict(model=name, day=d, n=len(D),
                 fpr=float(p[ben].mean()) if ben.any() else np.nan,
                 rec=float(p[~ben].mean()) if (~ben).any() else np.nan,
                 alert_rate=float(p.mean()),
                 uncertainty=float(((s >= 0.2) & (s <= 0.8)).mean()))
        r["internal_alert_share"] = float((D.src_ip_as[p] == HOME).mean()) if p.any() else 0.0
        dp = D[p].dest_port.value_counts(normalize=True)
        r["port_surge"] = float((dp - ref_port.reindex(dp.index, fill_value=0)).max()) if len(dp) else 0.0
        if full_dom:
            r["dom_all"] = dom_auc(ref_dom_all, D)
            r["dom_alerts"] = dom_auc(ref_alerts, D[p])
            r["dom_nonalerts"] = dom_auc(ref[rs < 0.5], D[~p])
        rows.append(r)
        print(name, d, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
