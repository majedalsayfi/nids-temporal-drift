"""Fig. 5: episode false alerts per day for RF under the update strategies of E14 (mean over three seeds;
whiskers = 95% hierarchical bootstrap interval), annotated with newly acquired labels."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
t = pd.read_csv("results/e14_rf_table.csv"); t = t[t.model == "RF"]
lab = {"STATIC": "Static (no updates)", "MONTHLY-CAPPED": "Monthly full retraining (150,000 flows)",
       "MONTHLY-RAND-680": "Monthly, 680 random labels", "WEEKLY-RAND-170": "Weekly, 170 random labels",
       "WEEKLY-AL-170": "Weekly, 170 most uncertain (AL)", "TRIG-UNC-200": "Triggered (uncertainty), 200 random",
       "TRIG-UNC-AL-200": "Triggered (uncertainty), 200 most uncertain", "TRIG-PORT-200": "Triggered (port surge), 200 random",
       "TRIG-UNC-200-DELAY7": "Triggered (uncertainty), labels 7 days late",
       "TRIG-UNC-200-NOISE10": "Triggered (uncertainty), 10% label errors"}
g = t[t.strategy.isin(lab)].set_index("strategy").sort_values("episode_fa")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
y = np.arange(len(g))
col = ["#2a78d6" if s.startswith("TRIG-UNC-200") and "DELAY" not in s and "NOISE" not in s else "#9a9a9a" for s in g.index]
ax.barh(y, g.episode_fa, color=col, height=0.6)
ax.errorbar(g.episode_fa, y, xerr=[g.episode_fa - g.episode_fa_lo, g.episode_fa_hi - g.episode_fa], fmt="none", ecolor="#444", lw=0.8, capsize=2)
ax.set_yticks(y); ax.set_yticklabels([lab[s] for s in g.index], fontsize=6.5)
for i, (s, r) in enumerate(g.iterrows()):
    ax.text(r.episode_fa_hi + 60, i, f"{int(round(r.new_labels, -2)):,} new labels", va="center", fontsize=5.8, color="#555")
ax.set_xlabel("RF false alerts per day, port-9092 episode")
ax.set_xlim(0, max(g.episode_fa_hi) * 1.45); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures_prop/fig_e8.png", dpi=300)
print(g[["episode_fa", "episode_fa_lo", "episode_fa_hi", "new_labels"]].round(0))
