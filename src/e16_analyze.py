"""E16 summary: FPR on ephemeral-port benign flows and on port-9092 benign flows per variant, mean over seeds."""
import pandas as pd
r = pd.read_csv("results_prop/e16_ports.csv")
pd.set_option("display.width", 250)
g = r.groupby(["model", "variant", "month"]).mean(numeric_only=True).drop(columns="seed")
for col in ["fpr", "fpr_eph", "fpr_9092", "mcc", "rec"]:
    t = (g[col].unstack("month") * (100 if col.startswith("fpr") else 1)).round(3)
    print("==", col); print(t.to_string())
agg = r.groupby(["model", "variant"]).agg(fpr=("fpr", "mean"), worst_fpr=("fpr", "max"), mcc=("mcc", "mean"), rec=("rec", "min"))
print(agg.round(5).to_string()); agg.to_csv("results_prop/e16_summary.csv")
