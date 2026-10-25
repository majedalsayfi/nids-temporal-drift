"""Kyoto KE5: which normal traffic does the T0 model flag? RF trained on T0 (seed 0).
Per year: FPR, and share of false positives by service and by flag, vs the service mix of normal flows in T0."""
import numpy as np, pandas as pd
from common import make_model
from kyoto_common import load_kyoto

X, M = load_kyoto(); Xv = X.values; y = M.y.values
t0 = np.where((M.month <= "2007-12").values)[0]
rng = np.random.default_rng(0)
tr = rng.choice(t0, min(150_000, len(t0)), replace=False)
m = make_model("RF", 0).fit(Xv[tr], y[tr])
ben = M[M.y == 0]
t0b = ben[ben.month <= "2007-12"]
rows = []
for yr in range(2008, 2016):
    ix = np.where(((M.year == yr) & (M.y == 0)).values)[0]
    ix = rng.choice(ix, min(60_000, len(ix)), replace=False)
    p = m.predict(Xv[ix]) == 1
    b = M.iloc[ix]
    fp = b[p]
    for key in ["service", "flag"]:
        share_fp = fp[key].value_counts(normalize=True)
        share_b = b[key].value_counts(normalize=True)
        share_t0 = t0b[key].value_counts(normalize=True)
        fpr_by = b.assign(p=p).groupby(key).p.mean()
        for v in share_fp.index[:4]:
            rows.append(dict(year=yr, fpr=p.mean(), key=key, value=v, fp_share=share_fp[v],
                             benign_share_year=share_b.get(v, 0), benign_share_T0=share_t0.get(v, 0),
                             fpr_within=fpr_by.get(v, np.nan)))
r = pd.DataFrame(rows); r.to_csv("results_kyoto/ke5.csv", index=False)
pd.set_option("display.width", 200); print(r.round(4).to_string())
