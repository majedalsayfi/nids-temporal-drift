"""Kyoto 2006+: label-free monitors on the monthly stream as additional drift events (reviewer request).
Static model trained on Nov-2006..Sep-2007 (cap 150,000), as in KE8; reference = initial-period flows not used for
training; calibration months Oct-Dec 2007. Per month (Jan-2008..Dec-2015): FPR, alert rate, uncertainty for three
bands, service surge (largest increase of any service's share among alerts relative to the reference alerts),
mean score and KS statistic against the reference scores. Seeds 0-2, RF and XGB.
Output: results_kyoto/kyoto_monitors.csv"""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from common import make_model
from kyoto_common import load_kyoto
X, M = load_kyoto(); Xv = X.values; y = M.y.values
months = sorted(M.month.unique())
init_m = [m for m in months if m <= "2007-09"]
byM = {m: np.where((M.month == m).values)[0] for m in months}
cap = lambda ix, n, s: ix if len(ix) <= n else np.random.default_rng(s).choice(ix, n, replace=False)
rows = []
for name in ["RF", "XGB"]:
    for seed in (0, 1, 2):
        init = np.concatenate([byM[m] for m in init_m]); tr = cap(init, 150_000, seed)
        ref = np.setdiff1d(init, tr)
        mdl = make_model(name, seed).fit(Xv[tr], y[tr])
        rs = mdl.predict_proba(Xv[ref])[:, 1]
        ref_svc = M.service.values[ref][rs >= .5]; ref_svc = pd.Series(ref_svc).value_counts(normalize=True)
        for m in months:
            if m < "2007-10": continue
            ix = cap(byM[m], 20_000, seed + 1); s = mdl.predict_proba(Xv[ix])[:, 1]; p = s >= .5; b = y[ix] == 0
            sv = pd.Series(M.service.values[ix][p]).value_counts(normalize=True)
            rows.append(dict(model=name, seed=seed, month=m, fpr=(p & b).sum() / max(b.sum(), 1), n_benign=int(b.sum()),
                             alert_rate=p.mean(), unc_10_90=((s >= .1) & (s <= .9)).mean(),
                             unc_20_80=((s >= .2) & (s <= .8)).mean(), unc_30_70=((s >= .3) & (s <= .7)).mean(),
                             service_surge=float((sv - ref_svc.reindex(sv.index, fill_value=0)).max()) if len(sv) else 0.0,
                             mean_score=s.mean(), ks_score=ks_2samp(s, rs).statistic))
        print(name, seed, flush=True)
pd.DataFrame(rows).to_csv("results_kyoto/kyoto_monitors.csv", index=False)
