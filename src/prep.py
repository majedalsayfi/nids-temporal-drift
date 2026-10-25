"""Build a per-day uniform random sample of LUFlow (keeps natural daily class priors).
Also records the full per-day label counts for the dataset description table."""
import sys, glob, os, zipfile, io
import numpy as np, pandas as pd

RAW = sys.argv[1]           # path to LUFlow repo
OUT = sys.argv[2]           # output dir
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # >0: volume-proportional sampling
NAME = sys.argv[5] if len(sys.argv) > 5 else "luflow_sample"
os.makedirs(OUT, exist_ok=True)

cols = ["avg_ipt","bytes_in","bytes_out","dest_ip","dest_port","entropy","num_pkts_out",
        "num_pkts_in","proto","src_ip","src_port","time_end","time_start","total_entropy",
        "label","duration"]
from multiprocessing import Pool

def one(f):
    day = os.path.basename(f).replace(".zip", "").replace(".", "-")
    with zipfile.ZipFile(f) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        try:
            df = pd.read_csv(z.open(name), low_memory=False)
        except Exception as e:
            print("FAIL", f, e, flush=True); return None
    df = df[[c for c in cols if c in df.columns]]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    vc = df["label"].value_counts()
    cnt = {"day": day, "n": len(df), **{k: int(vc.get(k, 0)) for k in ["benign", "malicious", "outlier"]}}
    rng = np.random.default_rng(int(day.replace("-", "")))
    k = int(round(FRAC * len(df))) if FRAC > 0 else min(CAP, len(df))
    if k < len(df):
        df = df.iloc[rng.choice(len(df), k, replace=False)]
    df = df.copy(); df["day"] = day
    print(cnt, flush=True)
    return df, cnt

files = sorted(glob.glob(os.path.join(RAW, "20*/*/*.zip")))
with Pool(2) as p:
    res = [r for r in p.map(one, files) if r is not None]
samples = [r[0] for r in res]; counts = [r[1] for r in res]

S = pd.concat(samples, ignore_index=True)
S.to_parquet(os.path.join(OUT, NAME + ".parquet"))
pd.DataFrame(counts).to_csv(os.path.join(OUT, NAME + "_daily_counts.csv"), index=False)
print("rows", len(S), S["label"].value_counts().to_dict())
