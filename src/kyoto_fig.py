import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kyoto_common import load_kyoto
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})
d = pd.read_csv("results_kyoto/ke1.csv")
t = d[(d.protocol == "TEMP") & (d.test != "ALL-LATER")]
COL = {"RF": "#1baf7a", "XGB": "#eda100", "MLP": "#008300", "LR": "#2a78d6"}
MK = {"RF": "^", "XGB": "D", "MLP": "P", "LR": "o"}
fig, ax = plt.subplots(2, 1, figsize=(7.16, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
for m in ["LR", "RF", "XGB", "MLP"]:
    g = t[t.model == m].groupby("test").fpr.mean()
    x = pd.to_datetime(g.index + "-15")
    ax[0].plot(x, g.values, color=COL[m], lw=1.2, label=m)
    ra = d[(d.protocol == "RAND-ALL") & (d.model == m)].fpr.mean()
    ax[0].axhline(ra, color=COL[m], lw=0.7, ls=":")
ax[0].set_ylabel("False-positive rate"); ax[0].legend(ncol=4, fontsize=7, loc="upper left")
ax[0].set_title("(a) Monthly FPR of models trained on Nov-2006 to Dec-2007 (dotted: random split)", fontsize=8)
X, M = load_kyoto(); b = M[M.y == 0]
s = b.groupby("month").service.value_counts(normalize=True).unstack().fillna(0)
s = s[s.index >= "2008-01"]
parts = pd.DataFrame({"SMTP": s.get("smtp", 0), "DNS": s.get("dns", 0), "SSH": s.get("ssh", 0)})
parts["Other"] = 1 - parts.sum(axis=1)
x = pd.to_datetime(s.index + "-15")
cols = {"SMTP": "#2a78d6", "DNS": "#eb6834", "SSH": "#6250d6", "Other": "#bdbdbd"}
ax[1].stackplot(x, *[parts[c].values for c in cols], labels=list(cols), colors=list(cols.values()), lw=0)
ax[1].set_ylabel("Share of normal flows"); ax[1].set_ylim(0, 1)
ax[1].legend(ncol=4, fontsize=7, loc="lower right", bbox_to_anchor=(1, 1.0))
ax[1].set_title("(b) Service mix of normal traffic", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("figures_prop/fig_kyoto.png", dpi=300)
