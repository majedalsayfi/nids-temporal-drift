"""E13: isolating future exposure on identical future records (reviewer request).
For each seed, the future test set F = 50,000 uniformly sampled flows from every month Aug-2020..Feb-2021
(the same records for every training condition). Training sets have the same size (150,000) and the same
malicious share (35%) and never contain a record of F:
  PAST          drawn from T0 (Jun-Jul 2020) only
  PAST+FUTURE   drawn uniformly from Jun-2020..Feb-2021 (records of F excluded)
  PAST+FUTURE-G as PAST+FUTURE, but every record whose feature vector also occurs in F is excluded
                (duplicate groups are kept on one side of the split)
A fourth condition repeats the usual random split with duplicate groups kept together:
  RAND-ALL-G    70/30 split of Jun-2020..Feb-2021 by groups of identical feature vectors, evaluated on its own test set
Preprocessing (log transform and scaling for LR/MLP) is part of each model pipeline and is fitted on training data only.
Output: per-day confusion counts in results_prop/e13_daycounts.csv"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from common import *

N, SHARE, NTM, NTE = 150_000, 0.35, 50_000, 200_000
models = os.environ.get("MODELS", ",".join(MODELS)).split(",")
seeds = range(int(os.environ.get("NSEEDS", "5")))
df = load("data/luflow_prop.parquet")
main = df[df.month.isin(MONTHS)].copy()
main["g"] = pd.util.hash_pandas_object(main[FEATURES], index=False).values
later = MONTHS[2:]


def draw(pool, seed):
    """Sample N flows with a fixed malicious share."""
    nm = int(round(N * SHARE)); nb = N - nm
    b = pool[pool.y == 0].sample(n=nb, random_state=seed); m = pool[pool.y == 1].sample(n=nm, random_state=seed)
    return pd.concat([b, m])


def counts(model, t, cond, name, seed, test):
    p = model.predict_proba(t[FEATURES])[:, 1] >= 0.5; y = t.y.values == 1
    g = pd.DataFrame(dict(day=t.day.values, tp=p & y, fp=p & ~y, fn=~p & y, tn=~p & ~y)).groupby("day").sum().reset_index()
    g["condition"], g["model"], g["seed"], g["test"] = cond, name, seed, test
    return g


out = []
for seed in seeds:
    F = pd.concat([cap(main[main.month == m], NTM, seed) for m in later])
    rest = main.drop(F.index)
    Fg = set(F.g)
    pools = {"PAST": rest[rest.month.isin(["2020-06", "2020-07"])],
             "PAST+FUTURE": rest,
             "PAST+FUTURE-G": rest[~rest.g.isin(Fg)]}
    trains = {k: draw(v, seed) for k, v in pools.items()}
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    i_tr, i_te = next(gss.split(main, groups=main.g))
    trG, teG = cap(main.iloc[i_tr], N, seed), cap(main.iloc[i_te], NTE, seed)
    for name in models:
        for cond, tr in trains.items():
            m = make_model(name, seed).fit(tr[FEATURES], tr.y)
            out.append(counts(m, F, cond, name, seed, "F"))
        m = make_model(name, seed).fit(trG[FEATURES], trG.y)
        out.append(counts(m, teG, "RAND-ALL-G", name, seed, "RAND-ALL-G"))
        print(seed, name, flush=True)
    pd.concat(out).to_csv("results_prop/e13_daycounts.csv", index=False)
