"""E5: diagnose the false-positive episodes. Train RF on T0 (seed 0), score every later month,
and break down false positives by destination-port / protocol, compared with how often
those benign services appear in T0."""
import sys, numpy as np, pandas as pd
from common import *

data, out = sys.argv[1], sys.argv[2]
df = load(data)
T0 = cap(df[df.month.isin(["2020-06", "2020-07"])], 150_000, 0)
m = make_model("RF", 0).fit(T0[FEATURES], T0.y)
t0b = df[df.month.isin(["2020-06", "2020-07"]) & (df.y == 0)]
t0_port_share = t0b.dest_port.value_counts(normalize=True)
rows, days = [], []
for mo in [x for x in MONTHS if x not in ("2020-06", "2020-07")] + [FAR]:
    te = df[(df.month == mo)]
    p = m.predict(te[FEATURES])
    fp = te[(te.y == 0) & (p == 1)]
    ben = te[te.y == 0]
    vc = fp.dest_port.value_counts()
    for port, c in vc.head(5).items():
        rows.append(dict(month=mo, dest_port=int(port), fp=int(c), fp_share=c / max(len(fp), 1),
                         benign_share_month=float((ben.dest_port == port).mean()),
                         benign_share_T0=float(t0_port_share.get(port, 0.0))))
    # per-day FPR to locate episodes
    te = te.assign(pred=p)
    for d, g in te[te.y == 0].groupby("day"):
        days.append(dict(day=d, fpr=float(g.pred.mean()), n_benign=len(g)))
pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(days).to_csv(out.replace(".csv", "_daily.csv"), index=False)
print(pd.DataFrame(rows).to_string())
