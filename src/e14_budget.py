"""E14: label accounting, matched labelling budgets, active learning, label delay and label noise
on the LUFlow daily stream (1 Sep 2020 .. 17 Feb 2021; same model, monitors and calibration as E8).

Every update retrains on the initial T0 training set (150,000 labelled flows, reused) plus all newly
labelled flows so far (weight 10). Newly acquired labels are counted once; reused labels are not recounted.
Strategies (budget per update in brackets):
  STATIC                 no update
  MONTHLY-CAPPED         1st of each month: retrain on a fresh random sample of 150,000 labelled past flows
                         (the practical form of full retraining; new labels = sampled flows not labelled before)
  MONTHLY-RAND-680       1st of each month: label 680 random flows of the previous day          (~4,080 labels)
  WEEKLY-RAND-170        every 7th day: label 170 random flows of that day                         (~4,080 labels)
  WEEKLY-AL-170          every 7th day: label the 170 flows with the most uncertain scores (active learning)
  TRIG-UNC-200           on uncertainty alarms (cool-down 3 days): 200 random flows   (as E8)
  TRIG-UNC-AL-200        on uncertainty alarms: the 200 most uncertain flows
  TRIG-PORT-200          on port-surge alarms: 200 random flows                       (as E8)
  TRIG-UNC-200-DELAY3/7  as TRIG-UNC-200, labels arrive 3 or 7 days after the alarm
  TRIG-UNC-200-NOISE5/10 as TRIG-UNC-200, 5% or 10% of acquired labels are flipped
"""
import os, sys, numpy as np, pandas as pd
from common import *

out = sys.argv[1]
models = sys.argv[2].split(",")
seeds = [int(s) for s in sys.argv[3].split(",")]
STRATS = os.environ.get("STRATS")
df = load("data/luflow_sample.parquet")
T0all = df[df.month.isin(["2020-06", "2020-07"])]
days = sorted(df[(df.day >= "2020-09-01") & (df.day < "2022-01-01")].day.unique())
cal_days = sorted(df[(df.day >= "2020-08-01") & (df.day < "2020-09-01")].day.unique())
bydays = {d: g for d, g in df[df.day >= "2020-08-01"].groupby("day")}
ALL = ["STATIC", "MONTHLY-CAPPED", "MONTHLY-RAND-680", "WEEKLY-RAND-170", "WEEKLY-AL-170", "TRIG-UNC-200",
       "TRIG-UNC-AL-200", "TRIG-PORT-200", "TRIG-UNC-200-DELAY3", "TRIG-UNC-200-DELAY7",
       "TRIG-UNC-200-NOISE5", "TRIG-UNC-200-NOISE10"]
strategies = STRATS.split(",") if STRATS else ALL


def monitors(m, D, ref_port):
    s = m.predict_proba(D[FEATURES])[:, 1]; p = s >= 0.5
    dp = D[p].dest_port.value_counts(normalize=True)
    return dict(uncertainty=float(((s >= .2) & (s <= .8)).mean()),
                port_surge=float((dp - ref_port.reindex(dp.index, fill_value=0)).max()) if len(dp) else 0.0)


def fit(name, seed, X, y, w):
    m = make_model(name, seed)
    if name in ("LR", "MLP"):
        return m.fit(X, y)
    return m.fit(X, y, sample_weight=w)


rows = []
for name in models:
    for seed in seeds:
        train0 = cap(T0all, 150_000, seed); ref = T0all.drop(train0.index)
        base = make_model(name, seed).fit(train0[FEATURES], train0.y)
        rp = base.predict_proba(ref[FEATURES])[:, 1] >= 0.5
        ref_port = ref[rp].dest_port.value_counts(normalize=True)
        mon = {d: monitors(base, bydays[d], ref_port) for d in cal_days + days}
        thr = {k: max(mon[d][k] for d in cal_days) for k in ["uncertainty", "port_surge"]}
        for strat in strategies:
            rng = np.random.default_rng(1000 * seed + 7)
            model = base; new_X, new_y = [], []; labelled_ids = set(train0.index)
            new_labels, updates, last = 0, 0, -99; pending = []  # (available_day_index, X, y)
            delay = int(strat.split("DELAY")[1]) if "DELAY" in strat else 1
            noise = int(strat.split("NOISE")[1]) / 100 if "NOISE" in strat else 0.0
            month_seen = days[0][:7]
            for i, d in enumerate(days):
                # labels that became available today
                arrived = [p for p in pending if p[0] <= i]; pending = [p for p in pending if p[0] > i]
                if arrived:
                    for _, X, y in arrived:
                        new_X.append(X); new_y.append(y)
                    Xtr = pd.concat([train0[FEATURES]] + new_X); ytr = pd.concat([train0.y] + new_y)
                    w = np.r_[np.ones(len(train0)), np.full(len(Xtr) - len(train0), 10.0)]
                    model = fit(name, seed, Xtr, ytr, w); updates += 1
                if strat == "MONTHLY-CAPPED" and d[:7] != month_seen:
                    month_seen = d[:7]
                    tr = cap(df[df.day < d], 150_000, seed + updates + 1)
                    new_labels += len(set(tr.index) - labelled_ids); labelled_ids |= set(tr.index)
                    model = make_model(name, seed).fit(tr[FEATURES], tr.y); updates += 1
                D = bydays[d]
                s = model.predict_proba(D[FEATURES])[:, 1]; pred = s >= 0.5; y = D.y.values
                rows.append(dict(model=name, seed=seed, strategy=strat, day=d,
                                 tp=int((pred & (y == 1)).sum()), fp=int((pred & (y == 0)).sum()),
                                 fn=int((~pred & (y == 1)).sum()), tn=int((~pred & (y == 0)).sum()),
                                 new_labels=new_labels, updates=updates,
                                 unc=float(((s >= .2) & (s <= .8)).mean())))
                if strat in ("STATIC", "MONTHLY-CAPPED"):
                    continue
                n = 200 if strat.startswith("TRIG-UNC-AL") else int(strat.split("-")[2])
                fire = False
                if strat.startswith("WEEKLY") and i % 7 == 6: fire = True
                if strat.startswith("MONTHLY-RAND") and (i + 1 < len(days) and days[i + 1][:7] != d[:7]): fire = True
                if strat.startswith("TRIG"):
                    # monitors use the initial model and the fixed T0 reference, as in E8
                    key = "port_surge" if "PORT" in strat else "uncertainty"
                    fire = mon[d][key] > thr[key] and i - last >= 3
                if not fire: continue
                if "-AL-" in strat:
                    idx = np.argsort(np.abs(s - 0.5))[:n]
                else:
                    idx = rng.choice(len(D), min(n, len(D)), replace=False)
                L = D.iloc[idx]; yl = L.y.values.copy()
                if noise > 0:
                    flip = rng.random(len(yl)) < noise; yl[flip] = 1 - yl[flip]
                yl = pd.Series(yl, index=L.index)
                new_labels += len(L); last = i
                pending.append((i + delay, L[FEATURES], yl))
            print(name, seed, strat, "new labels", new_labels, "updates", updates, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
