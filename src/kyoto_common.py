"""Kyoto 2006+ (AnoShift 'subset' release) loader.
AnoShift columns (Kyoto 2006+ order, IP/port fields removed by AnoShift):
 0 duration (log-bin)   1 service   2 src_bytes (log-bin)   3 dst_bytes (log-bin)   4 count
 5 same_srv_rate 6 serror_rate 7 srv_serror_rate 8 dst_host_count 9 dst_host_srv_count
10 dst_host_same_src_port_rate 11 dst_host_serror_rate 12 dst_host_srv_serror_rate 13 flag
14 start_time 15 ids_detection 16 malware_detection 17 ashula_detection 18 label 19 protocol
Excluded: 14 (time), 15-17 (outputs of signature detectors -> label leakage), 18 (label).
Label: 1 = normal -> y=0 ; -1 known attack, -2 unknown attack -> y=1."""
import glob, os, numpy as np, pandas as pd

NUM = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
BIN = ["0", "2", "3"]
CAT = ["1", "13", "19"]


def load_kyoto(raw=os.environ.get("KYOTO_RAW", "data/kyoto/raw"), per_month=25_000, seed=2026,
               cache=os.environ.get("KYOTO_CACHE", "data/kyoto/kyoto_sample.parquet")):
    """Uniform sample of up to `per_month` flows from every month of the AnoShift *_subset files
    (the *_subset_valid files are not used). Uniform within-month sampling keeps each month's class mix."""
    if cache and os.path.exists(cache):
        d = pd.read_parquet(cache)
    else:
        parts = []
        for f in sorted(glob.glob(os.path.join(raw, "*_subset.parquet"))):
            x = pd.read_parquet(f, columns=[str(i) for i in range(20) if i not in (15, 16, 17)])
            mo = x["14"].str[:6]
            keep = []
            rng = np.random.default_rng(seed)
            for m, ix in x.groupby(mo).indices.items():
                keep.append(ix if len(ix) <= per_month else rng.choice(ix, per_month, replace=False))
            parts.append(x.iloc[np.sort(np.concatenate(keep))])
            del x
        d = pd.concat(parts, ignore_index=True)
        if cache:
            d.to_parquet(cache)
    d["t"] = pd.to_datetime(d["14"], format="%Y%m%d %H:%M:%S")
    d["month"] = d.t.dt.strftime("%Y-%m")
    d["year"] = d.t.dt.year
    d["y"] = (d["18"].astype(int) != 1).astype(np.int8)
    d["unknown_attack"] = (d["18"].astype(int) == -2).astype(np.int8)
    X = pd.DataFrame(index=d.index)
    for c in NUM:
        X["f" + c] = pd.to_numeric(d[c], errors="coerce")
    for c in BIN:  # 'c<col><bin>' -> integer bin index
        X["f" + c] = pd.to_numeric(d[c].str[len(c) + 1:], errors="coerce")
    dummies = pd.get_dummies(d[CAT].rename(columns={"1": "svc", "13": "flag", "19": "proto"}), dtype=np.float32)
    X = pd.concat([X.astype(np.float32), dummies], axis=1).fillna(0)
    meta = d[["t", "month", "year", "y", "unknown_attack", "1", "13", "19"]].rename(
        columns={"1": "service", "13": "flag", "19": "proto"})
    return X, meta
