import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
o = pd.read_csv("results/e8_summary.csv"); o = o[o.model == "RF"]
lab = {"STATIC": "Static (no updates)", "WEEKLY-RECAL-200": "Weekly threshold re-pick", "TRIG-RECAL-200[uncertainty]": "Triggered threshold re-pick",
       "WEEKLY-RETRAIN-200": "Weekly retrain", "TRIG-RETRAIN-200[dom_all]": "Triggered retrain (domain AUC)",
       "TRIG-RETRAIN-200[port_surge]": "Triggered retrain (port surge)", "TRIG-RETRAIN-200[uncertainty]": "Triggered retrain (uncertainty)",
       "MONTHLY-EXPANDING": "Monthly expanding (all labels)"}
g = o[o.strategy.isin(lab)].groupby("strategy").agg(m=("episode_false_alerts_per_day", "mean"), lo=("episode_false_alerts_per_day", "min"),
                                                   hi=("episode_false_alerts_per_day", "max"), L=("labels", "mean"))
g = g.sort_values("m")
fig, ax = plt.subplots(figsize=(4.4, 2.6))
y = np.arange(len(g))
col = ["#2a78d6" if s == "TRIG-RETRAIN-200[uncertainty]" else "#9a9a9a" for s in g.index]
ax.barh(y, g.m, color=col, height=0.6)
ax.errorbar(g.m, y, xerr=[g.m - g.lo, g.hi - g.m], fmt="none", ecolor="#444", lw=0.8, capsize=2)
ax.set_yticks(y); ax.set_yticklabels([lab[s] for s in g.index], fontsize=7)
for i, (s, r) in enumerate(g.iterrows()):
    t = "all past flows" if s == "MONTHLY-EXPANDING" else f"{int(round(r.L, -2)):,} labels"
    ax.text(r.hi + 60, i, t, va="center", fontsize=6, color="#555")
ax.set_xlabel("False alerts per day in the episode (RF)", loc="left")
ax.set_xlim(0, 4600); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures_prop/fig_e8.png", dpi=300)
print(g.round(0))
