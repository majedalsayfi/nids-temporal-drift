"""Convert E8 boosted-model runs (three seeds) to the E14 format for joint analysis."""
import pandas as pd
m = {"STATIC": "STATIC", "WEEKLY-RETRAIN-200": "WEEKLY-RAND-200", "TRIG-RETRAIN-200[port_surge]": "TRIG-PORT-200",
     "TRIG-RETRAIN-200[uncertainty]": "TRIG-UNC-200", "MONTHLY-EXPANDING": "MONTHLY-CAPPED"}
e = pd.concat([pd.read_csv(f) for f in ["results/e8_xgb.csv", "results/e8_xgbseeds.csv", "results/e8_lgbm.csv", "results/e8_lgbmseeds.csv"]])
e = e[e.strategy.isin(m)].copy(); e["strategy"] = e.strategy.map(m)
e = e.rename(columns={"labels_used": "new_labels"})
e.loc[e.strategy == "MONTHLY-CAPPED", "new_labels"] = 558634  # fresh 150,000-flow samples (E14 accounting)
e[["model", "seed", "strategy", "day", "tp", "fp", "fn", "tn", "new_labels", "updates"]].to_csv("results/e14_boost_from_e8.csv", index=False)
print(e.groupby(["model", "strategy"]).seed.nunique())
