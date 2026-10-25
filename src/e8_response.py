"""E8: update strategies on a daily stream, with their labelling cost.

Stream = every day from 2020-09-01 to 2021-02-17. A model is trained on T0 (Jun-Jul 2020).
Labels obtained on day d can only be used from day d+1.

Strategies
  STATIC                  never updated, threshold 0.5
  MONTHLY-EXPANDING       retrained on the 1st of each month on all past data (fully labelled)
  WEEKLY-RECAL-n          every 7 days: label n random flows of that day, re-pick the threshold
  WEEKLY-RETRAIN-n        every 7 days: label n flows, retrain on T0 + all labels so far
  TRIG-RECAL-n[mon]       label n flows only on days the monitor alarms, re-pick the threshold
  TRIG-RETRAIN-n[mon]     label n flows on alarm days (cool-down 3 days), retrain
Monitor alarm = monitor value above its maximum over the Aug-2020 calibration days.
Threshold re-pick = threshold maximising MCC on the last 4 labelled samples (ties -> closest to 0.5).
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef
from common import *

data, mon_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
models = sys.argv[4].split(",") if len(sys.argv) > 4 else ["RF"]
seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0]
import os
NS = [int(x) for x in os.environ.get("NS", "200,1000").split(",")]
MON_ONLY = os.environ.get("MONITORS")
HOME = 786
df = load(data)
df["src_ip_as"] = pd.to_numeric(df["src_ip"], errors="coerce")
T0all = df[df.month.isin(["2020-06", "2020-07"])]
days = sorted(df[(df.day >= "2020-09-01") & (df.day < "2022-01-01")].day.unique())
cal_days = sorted(df[(df.day >= "2020-08-01") & (df.day < "2020-09-01")].day.unique())
bydays = {d: g for d, g in df[df.day >= "2020-08-01"].groupby("day")}
mon_rf = pd.read_csv(mon_csv).set_index("day")  # dom_* monitors from E7 (RF, seed 0)
GRID = np.r_[np.linspace(0.05, 0.95, 91), 0.96, 0.97, 0.98, 0.99, 0.995]


def pick_threshold(samples):
    s = np.concatenate([x[0] for x in samples]); y = np.concatenate([x[1] for x in samples])
    if y.min() == y.max():
        return 0.5
    best, bt = -2, 0.5
    for t in GRID:
        v = matthews_corrcoef(y, s >= t)
        if v > best + 1e-9 or (abs(v - best) <= 1e-9 and abs(t - 0.5) < abs(bt - 0.5)):
            best, bt = v, t
    return bt


def cheap_monitors(m, D, ref_port):
    s = m.predict_proba(D[FEATURES])[:, 1]; p = s >= 0.5
    dp = D[p].dest_port.value_counts(normalize=True)
    return dict(alert_rate=float(p.mean()),
                uncertainty=float(((s >= .2) & (s <= .8)).mean()),
                internal_alert_share=float((D.src_ip_as[p] == HOME).mean()) if p.any() else 0.0,
                port_surge=float((dp - ref_port.reindex(dp.index, fill_value=0)).max()) if len(dp) else 0.0)


rows, events = [], []
for name in models:
    for seed in seeds:
        train0 = cap(T0all, 150_000, seed)
        ref = T0all.drop(train0.index)
        base = make_model(name, seed).fit(train0[FEATURES], train0.y)
        rp = base.predict_proba(ref[FEATURES])[:, 1] >= 0.5
        ref_port = ref[rp].dest_port.value_counts(normalize=True)
        # monitor values for calibration + stream (cheap ones from this model, dom_* from E7)
        mon = {}
        for d in cal_days + days:
            v = cheap_monitors(base, bydays[d], ref_port)
            for k in ["dom_all", "dom_alerts", "dom_nonalerts"]:
                v[k] = float(mon_rf.loc[d, k]) if d in mon_rf.index and k in mon_rf.columns else np.nan
            mon[d] = v
        monitors = ["alert_rate", "uncertainty", "internal_alert_share", "port_surge", "dom_all", "dom_alerts", "dom_nonalerts"]
        thr = {k: np.nanmax([mon[d][k] for d in cal_days]) for k in monitors}
        alarm = {k: {d: bool(mon[d][k] > thr[k]) for d in days} for k in monitors}

        strategies = ["STATIC", "MONTHLY-EXPANDING"]
        use_mons = MON_ONLY.split(",") if MON_ONLY else monitors
        for n in NS:
            strategies += [f"WEEKLY-RECAL-{n}", f"WEEKLY-RETRAIN-{n}"]
            for k in use_mons:
                strategies += [f"TRIG-RECAL-{n}[{k}]", f"TRIG-RETRAIN-{n}[{k}]"]
        for strat in strategies:
            rng = np.random.default_rng(1000 * seed + 7)
            model, t = base, 0.5
            labelled, pool_X, pool_y = [], [], []
            labels_used, updates, last_retrain = 0, 0, -99
            mon_key = strat[strat.index("[") + 1:-1] if "[" in strat else None
            n = int(strat.split("-")[2].split("[")[0]) if strat.startswith(("WEEKLY", "TRIG")) else 0
            month_seen = days[0][:7]
            for i, d in enumerate(days):
                D = bydays[d]
                if strat == "MONTHLY-EXPANDING" and d[:7] != month_seen:
                    month_seen = d[:7]
                    past = df[df.day < d]
                    tr = cap(past, 150_000, seed)
                    model = make_model(name, seed).fit(tr[FEATURES], tr.y)
                    labels_used = len(df[(df.day >= "2020-08-01") & (df.day < d)])
                    updates += 1
                s = model.predict_proba(D[FEATURES])[:, 1]
                pred = s >= t
                y = D.y.values
                rows.append(dict(model=name, seed=seed, strategy=strat, day=d, n=len(D),
                                 tp=int((pred & (y == 1)).sum()), fp=int((pred & (y == 0)).sum()),
                                 fn=int((~pred & (y == 1)).sum()), tn=int((~pred & (y == 0)).sum()),
                                 threshold=t, labels_used=labels_used, updates=updates))
                # end-of-day update decision (takes effect tomorrow)
                fire = False
                if strat.startswith("WEEKLY") and i % 7 == 6:
                    fire = True
                if strat.startswith("TRIG") and alarm[mon_key][d]:
                    fire = True
                    if "RETRAIN" in strat and i - last_retrain < 3:
                        fire = False
                if fire:
                    idx = rng.choice(len(D), min(n, len(D)), replace=False)
                    L = D.iloc[idx]
                    labels_used += len(L)
                    updates += 1
                    if "RECAL" in strat:
                        labelled.append((model.predict_proba(L[FEATURES])[:, 1], L.y.values))
                        labelled = labelled[-4:]
                        t = pick_threshold(labelled)
                    else:
                        pool_X.append(L[FEATURES]); pool_y.append(L.y)
                        Xtr = pd.concat([train0[FEATURES]] + pool_X); ytr = pd.concat([train0.y] + pool_y)
                        w = np.r_[np.ones(len(train0)), np.full(sum(len(x) for x in pool_X), 10.0)]
                        model = make_model(name, seed)
                        try:
                            if name in ("LR", "MLP"):
                                model.fit(Xtr, ytr)
                            else:
                                model.fit(Xtr, ytr, sample_weight=w)
                        except TypeError:
                            model.fit(Xtr, ytr)
                        last_retrain = i
                    events.append(dict(model=name, seed=seed, strategy=strat, day=d, threshold=t))
            print(name, seed, strat, "labels", labels_used, "updates", updates, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.DataFrame(events).to_csv(out.replace(".csv", "_events.csv"), index=False)
        pd.DataFrame([dict(model=name, seed=seed, monitor=k, threshold=thr[k]) for k in monitors]).to_csv(
            out.replace(".csv", f"_thr_{name}_{seed}.csv"), index=False)
