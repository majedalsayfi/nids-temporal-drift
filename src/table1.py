"""Table 1: monthly composition of the volume-proportional LUFlow sample after cleaning (benign vs malicious)."""
from common import load
df = load("data/luflow_prop.parquet")
t = df.groupby("month").agg(days=("day", "nunique"), flows=("y", "size"), malicious_share=("y", "mean"))
print(t.round(3).to_string()); print("total", len(df), "benign", int((df.y == 0).sum()), "malicious", int(df.y.sum()))
t.to_csv("results_prop/table1.csv")
