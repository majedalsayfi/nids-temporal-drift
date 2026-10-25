"""Kyoto KE8: update strategies on a monthly stream, Jan-2008 .. Dec-2015.
Initial model trained on Nov-2006 .. Sep-2007; Oct-Dec 2007 calibrates the uncertainty alarm
(threshold = max monthly uncertainty over the calibration months). Labels from month m are used from m+1.
  STATIC              never updated
  EXPANDING           retrained every month on all past months (fully labelled)
  SLIDING             retrained every month on the previous month only (fully labelled)
  MONTHLY-200         every month: label 200 random flows, retrain on initial data + all labels (weight 10)
  TRIG-200[unc]       as MONTHLY-200, but only in months whose uncertainty exceeds the alarm threshold
"""
import sys, os, numpy as np, pandas as pd
from common import make_model
from kyoto_common import load_kyoto

out = sys.argv[1]
models = os.environ.get("MODELS", "RF,XGB").split(",")
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
NTR, NTM, NL = 150_000, 20_000, 200
X, M = load_kyoto(); Xv = X.values; y = M.y.values
months = sorted(M.month.unique())
init_m = [m for m in months if m <= "2007-09"]; cal_m = ["2007-10", "2007-11", "2007-12"]
stream = [m for m in months if m >= "2008-01"]
byM = {m: np.where((M.month == m).values)[0] for m in months}


def capi(ix, n, seed):
    return ix if len(ix) <= n else np.random.default_rng(seed).choice(ix, n, replace=False)


def unc(model, ix):
    s = model.predict_proba(Xv[ix])[:, 1]
    return float(((s >= .2) & (s <= .8)).mean()), s


rows = []
for name in models:
    for seed in (seeds if name == "RF" else seeds[:1]):
        tests = {m: capi(byM[m], NTM, seed + 1) for m in months}
        init = capi(np.concatenate([byM[m] for m in init_m]), NTR, seed)
        base = make_model(name, seed).fit(Xv[init], y[init])
        thr = max(unc(base, tests[m])[0] for m in cal_m)
        for strat in ["STATIC", "EXPANDING", "SLIDING", "MONTHLY-200", "TRIG-200[unc]"]:
            model, labels, extra, n_upd = base, 0, [], 0
            rng = np.random.default_rng(100 + seed)
            for i, m in enumerate(stream):
                if strat == "EXPANDING" and i > 0:
                    past = capi(np.concatenate([byM[k] for k in months if k < m]), NTR, seed)
                    model = make_model(name, seed).fit(Xv[past], y[past]); n_upd += 1
                if strat == "SLIDING" and i > 0:
                    prev = capi(byM[stream[i - 1]], NTR, seed)
                    model = make_model(name, seed).fit(Xv[prev], y[prev]); n_upd += 1
                te = tests[m]
                u, s = unc(model, te)
                p = s >= .5; yt = y[te]
                rows.append(dict(model=name, seed=seed, strategy=strat, month=m, unc=u, thr=thr,
                                 tp=int((p & (yt == 1)).sum()), fp=int((p & (yt == 0)).sum()),
                                 fn=int((~p & (yt == 1)).sum()), tn=int((~p & (yt == 0)).sum()),
                                 labels=labels, updates=n_upd))
                fire = strat == "MONTHLY-200" or (strat == "TRIG-200[unc]" and u > thr)
                if fire:
                    lab = rng.choice(byM[m], min(NL, len(byM[m])), replace=False)
                    extra.append(lab); labels += len(lab); n_upd += 1
                    allx = np.concatenate([init] + extra)
                    w = np.r_[np.ones(len(init)), np.full(len(allx) - len(init), 10.0)]
                    model = make_model(name, seed)
                    if name in ("LR", "MLP"):
                        model.fit(Xv[allx], y[allx])
                    else:
                        model.fit(Xv[allx], y[allx], sample_weight=w)
            print(name, seed, strat, "labels", labels, "updates", n_upd, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
