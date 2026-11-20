"""Kyoto 2006+: class counts per year in the AnoShift *_subset.parquet release files and in our
monthly sample (up to 25,000 flows per month), before and after preprocessing.
Output: results_kyoto/kyoto_counts.csv"""
import glob, os, pandas as pd
from kyoto_common import load_kyoto
raw = os.environ.get("KYOTO_RAW", "data/kyoto/raw")
rows = []
for f in sorted(glob.glob(os.path.join(raw, "*_subset.parquet"))):
    x = pd.read_parquet(f, columns=["14", "18"])
    yr = int(os.path.basename(f)[:4]); lab = x["18"].astype(int)
    rows.append(dict(year=yr, file=os.path.basename(f), release_flows=len(x), release_normal=int((lab == 1).sum()),
                     release_known_attack=int((lab == -1).sum()), release_unknown_attack=int((lab == -2).sum()),
                     release_months=x["14"].str[:6].nunique()))
r = pd.DataFrame(rows).set_index("year")
X, M = load_kyoto()
s = M.groupby("year").agg(sample_flows=("y", "size"), sample_normal=("y", lambda v: int((v == 0).sum())),
                          sample_attack=("y", "sum"), sample_months=("month", "nunique"))
r = r.join(s); r["release_attack_share"] = 1 - r.release_normal / r.release_flows
r["sample_attack_share"] = r.sample_attack / r.sample_flows
print(r.to_string()); r.to_csv("results_kyoto/kyoto_counts.csv")
print("features after preprocessing:", X.shape[1], "rows dropped in preprocessing:", 0)
