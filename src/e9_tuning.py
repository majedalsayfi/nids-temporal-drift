"""E9: does hyper-parameter tuning remove the temporal gap?
For RF, XGB and LGBM: 20 random configurations, selected by MCC under
  (a) RANDOM validation  - stratified random 70/30 split of the T0 training sample
  (b) TEMPORAL validation - train on T0 days up to 2020-07-15, validate on 2020-07-16..31
The selected configuration is refit on the T0 training sample and evaluated with the same
protocols as E1 (RAND-ALL, TEMP per month, Jun-2022)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform, loguniform
import xgboost as xgb, lightgbm as lgb
from common import *

data, out = sys.argv[1], sys.argv[2]
seed = 0
df = load(data)
main = df[df.month.isin(MONTHS)]
T0 = main[main.month.isin(["2020-06", "2020-07"])]
later = [m for m in MONTHS if m not in ("2020-06", "2020-07")] + [FAR]
tr_t0 = cap(T0, 150_000, seed)
SPACE = {
    "RF": {"n_estimators": randint(50, 400), "max_depth": [None, 10, 20, 30], "min_samples_leaf": randint(1, 20),
           "max_features": ["sqrt", "log2", 0.5, None]},
    "XGB": {"n_estimators": randint(100, 600), "max_depth": randint(3, 12), "learning_rate": loguniform(0.02, 0.3),
            "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": loguniform(1, 50)},
    "LGBM": {"n_estimators": randint(100, 600), "num_leaves": randint(15, 255), "learning_rate": loguniform(0.02, 0.3),
             "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.5, 0.5), "min_child_samples": randint(5, 200)},
}


def build(name, p):
    if name == "RF":
        return RandomForestClassifier(n_jobs=2, random_state=seed, **p)
    if name == "XGB":
        return xgb.XGBClassifier(n_jobs=2, random_state=seed, tree_method="hist", eval_metric="logloss", **p)
    return lgb.LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, subsample_freq=1, **p)


def mcc_of(m, X, y):
    s = m.predict_proba(X)[:, 1]
    return metrics(y, (s >= .5).astype(int), s)["mcc"]


# validation splits
a_tr, a_va = train_test_split(tr_t0, test_size=0.3, stratify=tr_t0.y, random_state=seed)
b_tr = tr_t0[tr_t0.day <= "2020-07-15"]; b_va = tr_t0[tr_t0.day > "2020-07-15"]
# evaluation sets (as in E1)
r_tr, r_te = train_test_split(main, test_size=0.3, stratify=main.y, random_state=seed)
r_tr, r_te = cap(r_tr, 150_000, seed), cap(r_te, 200_000, seed)
tests = {m: cap(df[df.month == m], 50_000, seed) for m in later}
tests["ALL-LATER"] = cap(df[df.month.isin(later[:-1])], 200_000, seed)

rows = []
for name in ["RF", "XGB", "LGBM"]:
    cfgs = list(ParameterSampler(SPACE[name], n_iter=20, random_state=seed))
    scores = []
    for i, p in enumerate(cfgs):
        sa = mcc_of(build(name, p).fit(a_tr[FEATURES], a_tr.y), a_va[FEATURES], a_va.y)
        sb = mcc_of(build(name, p).fit(b_tr[FEATURES], b_tr.y), b_va[FEATURES], b_va.y)
        scores.append((sa, sb)); print(name, i, round(sa, 4), round(sb, 4), flush=True)
    for sel, idx in [("RANDOM-VAL", int(np.argmax([s[0] for s in scores]))),
                     ("TEMPORAL-VAL", int(np.argmax([s[1] for s in scores])))]:
        p = cfgs[idx]
        m_rand = build(name, p).fit(r_tr[FEATURES], r_tr.y)
        s = m_rand.predict_proba(r_te[FEATURES])[:, 1]
        rows.append(dict(model=name, selection=sel, cfg=str(p), test="RAND-ALL", **metrics(r_te.y, (s >= .5).astype(int), s)))
        m_t = build(name, p).fit(tr_t0[FEATURES], tr_t0.y)
        for k, v in tests.items():
            s = m_t.predict_proba(v[FEATURES])[:, 1]
            rows.append(dict(model=name, selection=sel, cfg=str(p), test=k, **metrics(v.y, (s >= .5).astype(int), s)))
        pd.DataFrame(rows).to_csv(out, index=False)
    print(name, "done", flush=True)
