import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score, recall_score,
                             confusion_matrix, accuracy_score, precision_score)
import lightgbm as lgb
import xgboost as xgb

FEATURES = ["avg_ipt", "bytes_in", "bytes_out", "dest_port", "entropy", "num_pkts_out",
            "num_pkts_in", "proto", "src_port", "total_entropy", "duration"]
MONTHS = ["2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12",
          "2021-01", "2021-02"]
FAR = "2022-06"


def load(path, outlier_policy="drop"):
    df = pd.read_parquet(path)
    df["month"] = df["day"].str[:7]
    if outlier_policy == "drop":
        df = df[df["label"].isin(["benign", "malicious"])]
        df["y"] = (df["label"] == "malicious").astype(np.int8)
    elif outlier_policy == "malicious":
        df = df[df["label"].isin(["benign", "malicious", "outlier"])]
        df["y"] = (df["label"] != "benign").astype(np.int8)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES)
    df[FEATURES] = df[FEATURES].clip(lower=0).astype(np.float64)
    return df.reset_index(drop=True)


def _log():
    return FunctionTransformer(np.log1p)


def make_model(name, seed):
    if name == "LR":
        return make_pipeline(_log(), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, min_samples_leaf=5)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed,
                                      min_samples_leaf=2)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                 random_state=seed, tree_method="hist", eval_metric="logloss")
    if name == "LGBM":
        import os
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  n_jobs=int(os.environ.get("LGBM_JOBS", "2")), random_state=seed, verbose=-1)
    if name == "MLP":
        return make_pipeline(_log(), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                           max_iter=200, random_state=seed))
    raise ValueError(name)


MODELS = ["LR", "DT", "RF", "XGB", "LGBM", "MLP"]


def metrics(y, p, s):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = dict(acc=accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
               rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
               mcc=matthews_corrcoef(y, p), fpr=fp / max(fp + tn, 1),
               mal_rate=float(np.mean(y)), n=len(y))
    try:
        out["auc"] = roc_auc_score(y, s)
    except ValueError:
        out["auc"] = np.nan
    return out


def fit_eval(name, seed, Xtr, ytr, tests):
    m = make_model(name, seed)
    m.fit(Xtr, ytr)
    res = {}
    for k, (Xte, yte) in tests.items():
        s = m.predict_proba(Xte)[:, 1]
        res[k] = metrics(yte, (s >= 0.5).astype(int), s)
    return res


def cap(df, n, seed):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)
