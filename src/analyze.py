"""Turn result CSVs into paper tables (CSV + LaTeX-ish text) and figures (PNG, 300 dpi)."""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

R = os.environ.get("RES", "results"); F = os.environ.get("FIG", "figures"); os.makedirs(F, exist_ok=True)
COL = {"LR": "#2a78d6", "DT": "#eb6834", "RF": "#1baf7a", "XGB": "#eda100", "LGBM": "#e87ba4", "MLP": "#008300"}
MK = {"LR": "o", "DT": "s", "RF": "^", "XGB": "D", "LGBM": "v", "MLP": "P"}
ORDER = ["LR", "DT", "RF", "XGB", "LGBM", "MLP"]
LATER = ["2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02"]
LAB = {m: pd.Timestamp(m + "-01").strftime("%b-%y") for m in LATER + ["2022-06"]}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "lines.linewidth": 1.5, "legend.frameon": False})
out = open(os.path.join(R, "summary.txt"), "w")
def P(*a):
    print(*a); print(*a, file=out)


def aut(vals):
    v = np.asarray(vals, float)
    return float(np.sum((v[1:] + v[:-1]) / 2) / (len(v) - 1))


# ---------------- E1 -----------------
def e1_summary(path, tag):
    d = pd.read_csv(path)
    rows = []
    for (mdl, seed), g in d.groupby(["model", "seed"]):
        t = g[g.protocol == "TEMP"].set_index("test")
        r = dict(model=mdl, seed=seed)
        for met in ["f1", "mcc", "fpr", "rec", "auc", "prec"]:
            r[f"randall_{met}"] = g[g.protocol == "RAND-ALL"][met].iloc[0]
            r[f"randt0_{met}"] = g[g.protocol == "RAND-T0"][met].iloc[0]
            r[f"temppool_{met}"] = t.loc["ALL-LATER", met]
            monthly = t.loc[LATER, met].values
            r[f"tempmean_{met}"] = monthly.mean()
            r[f"tempworst_{met}"] = monthly.max() if met == "fpr" else monthly.min()
            r[f"aut_{met}"] = aut(monthly)
            r[f"far_{met}"] = t.loc["2022-06", met]
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(R, f"{tag}_perseed.csv"), index=False)
    agg = s.groupby("model").agg(["mean", "std"]).drop(columns="seed")
    agg = agg.reindex(ORDER)
    agg.to_csv(os.path.join(R, f"{tag}_summary.csv"))
    P(f"\n===== {tag} =====")
    for met in ["f1", "mcc", "fpr", "rec", "auc"]:
        cols = [f"randall_{met}", f"randt0_{met}", f"temppool_{met}", f"tempmean_{met}",
                f"aut_{met}", f"tempworst_{met}", f"far_{met}"]
        tab = pd.DataFrame({c: agg[(c, "mean")].map(lambda x: f"{x:.4f}") + " ± " +
                            agg[(c, "std")].map(lambda x: f"{x:.4f}") for c in cols})
        P(f"--- {met}"); P(tab.to_string())
    # paired tests across models x seeds: RAND-ALL vs worst month / AUT
    for met in ["f1", "mcc"]:
        a, b = s[f"randall_{met}"], s[f"aut_{met}"]
        w = wilcoxon(a, b)
        P(f"Wilcoxon RAND-ALL vs AUT {met}: n={len(a)}, median diff={np.median(a-b):.4f}, W={w.statistic:.1f}, p={w.pvalue:.2e}")
        a, b = s[f"randall_{met}"], s[f"tempworst_{met}"]
        w = wilcoxon(a, b)
        P(f"Wilcoxon RAND-ALL vs worst-month {met}: median diff={np.median(a-b):.4f}, p={w.pvalue:.2e}")
    a, b = s["randall_fpr"], s["tempworst_fpr"]
    w = wilcoxon(a, b)
    P(f"Wilcoxon RAND-ALL vs worst-month FPR: median ratio={np.median(b/np.maximum(a,1e-6)):.1f}x, p={w.pvalue:.2e}")
    a, b = s["randall_f1"], s["randt0_f1"]
    w = wilcoxon(a, b)
    P(f"Wilcoxon RAND-ALL vs RAND-T0 f1: median diff={np.median(a-b):.5f}, p={w.pvalue:.2e}")
    return d, s


d1, s1 = e1_summary(os.path.join(R, "e1.csv"), "e1")
for extra in ["e1_noports", "e1_outlier"]:
    if os.path.exists(os.path.join(R, extra + ".csv")):
        e1_summary(os.path.join(R, extra + ".csv"), extra)

# monthly table (mean over seeds)
tm = d1[d1.protocol == "TEMP"].groupby(["test", "model"])[["f1", "fpr", "rec", "auc", "mcc"]].mean()
P("\nE1 monthly (mean over seeds)")
for met in ["f1", "fpr", "rec", "auc", "mcc"]:
    P(f"--- {met}"); P(tm[met].unstack()[ORDER].reindex(LATER + ["2022-06", "ALL-LATER"]).round(4).to_string())

# Fig 1: monthly F1 and FPR under temporal protocol with random-split reference
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
x = np.arange(len(LATER) + 1); xl = [LAB[m] for m in LATER] + ["Jun-22"]
for mdl in ORDER:
    g = d1[(d1.protocol == "TEMP") & (d1.model == mdl)].groupby("test")
    mf, sf = g.f1.mean(), g.f1.std()
    mp = g.fpr.mean()
    ra = d1[(d1.protocol == "RAND-ALL") & (d1.model == mdl)]
    ax[0].plot(x, mf.reindex(LATER + ["2022-06"]).values, marker=MK[mdl], ms=4, color=COL[mdl], label=mdl)
    ax[1].plot(x, mp.reindex(LATER + ["2022-06"]).values, marker=MK[mdl], ms=4, color=COL[mdl], label=mdl)
    ax[1].axhline(ra.fpr.mean(), color=COL[mdl], lw=0.8, ls=":")
ax[0].set_ylabel("F1 (malicious)"); ax[1].set_ylabel("False-positive rate (log)")
ax[1].set_yscale("log")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(xl, rotation=45, ha="right"); a.axvline(len(LATER) - 0.5, color="#999", lw=0.6, ls="--")
ax[0].set_title("(a) F1 under temporal evaluation", fontsize=8)
ax[1].set_title("(b) FPR; dotted = same model, random split", fontsize=8)
ax[0].legend(ncol=3, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig_monthly.png"), dpi=300); plt.close(fig)

# ---------------- E2 -----------------
if os.path.exists(os.path.join(R, "e2.csv")):
    d2 = pd.read_csv(os.path.join(R, "e2.csv"))
    MON = ["2020-06", "2020-07"] + LATER
    allm = MON + ["2022-06"]
    lab = {m: pd.Timestamp(m + "-01").strftime("%b-%y") for m in allm}
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={"wspace": 0.35})
    for a, mdl in zip(axs, ["RF", "LGBM"]):
        M = d2[d2.model == mdl].pivot(index="train", columns="test", values="mcc").reindex(index=MON, columns=allm)
        im = a.imshow(M.values, cmap="Blues_r", vmin=0.9, vmax=1.0, aspect="auto")
        a.set_xticks(range(len(allm))); a.set_xticklabels([lab[m] for m in allm], rotation=60, ha="right", fontsize=6)
        a.set_yticks(range(len(MON))); a.set_yticklabels([lab[m] for m in MON], fontsize=6)
        a.set_xlabel("Test month"); a.set_ylabel("Training month"); a.set_title(f"MCC, {mdl}", fontsize=8); a.grid(False)
        for i in range(M.shape[0]):
            a.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="#222", lw=0.6))
            for j in range(M.shape[1]):
                v = M.values[i, j]
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.8, color="white" if v < 0.94 else "#222")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, extend="min"); cb.set_label("MCC (values < 0.90 share the darkest shade)", fontsize=6)
    fig.savefig(os.path.join(F, "fig_matrix.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    # decay vs gap (forward only, within MON)
    d2["gap"] = d2.apply(lambda r: (pd.Period(r.test, "M") - pd.Period(r.train, "M")).n, axis=1)
    fw = d2[(d2.gap >= 0) & (d2.test != "2022-06")]
    gp = fw.groupby(["model", "gap"])[["f1", "mcc", "fpr"]].mean().reset_index()
    P("\nE2 mean MCC by gap (months)"); P(gp.pivot(index="gap", columns="model", values="mcc")[ORDER].round(4).to_string())
    P("\nE2 mean FPR by gap"); P(gp.pivot(index="gap", columns="model", values="fpr")[ORDER].round(4).to_string())
    diag = d2[d2.gap == 0].groupby("model").mcc.mean(); off = fw[fw.gap > 0].groupby("model").mcc.mean()
    P("\nE2 same-month vs future-month MCC"); P(pd.DataFrame({"same": diag, "future": off}).reindex(ORDER).round(4).to_string())
    back = d2[(d2.gap < 0)].groupby("model").mcc.mean()
    P("E2 past-month (backward) MCC"); P(back.reindex(ORDER).round(4).to_string())
    sp = [spearmanr(g.gap, g.mcc) for _, g in fw[fw.gap > 0].groupby("model")]
    P("Spearman gap vs MCC per model:", {m: (round(s.statistic, 3), f"{s.pvalue:.1e}") for m, s in zip(sorted(fw.model.unique()), sp)})
    P("E2 2022-06 MCC by training month"); P(d2[d2.test == "2022-06"].pivot(index="train", columns="model", values="mcc")[ORDER].round(4).to_string())

# ---------------- E3 -----------------
if os.path.exists(os.path.join(R, "e3.csv")):
    d3 = pd.read_csv(os.path.join(R, "e3.csv"))
    g = d3.groupby(["strategy", "model"])[["f1", "mcc", "fpr", "rec"]].mean()
    P("\nE3 mean over Sep-20..Feb-21 and seeds"); P(g.round(4).to_string())
    a = d3.groupby(["strategy", "model", "seed"]).apply(lambda t: aut(t.set_index("test").loc[LATER[1:], "mcc"].values)).unstack("strategy")
    P("E3 AUT(MCC) mean over seeds"); P(a.groupby("model").mean().reindex(ORDER).round(4).to_string())
    for s_ in ["SLIDING", "EXPANDING"]:
        w = wilcoxon(a["STATIC"], a[s_]); P(f"Wilcoxon STATIC vs {s_} AUT(MCC): p={w.pvalue:.2e}, median diff={np.median(a[s_]-a['STATIC']):.4f}")
    w = wilcoxon(a["SLIDING"], a["EXPANDING"]); P(f"Wilcoxon SLIDING vs EXPANDING: p={w.pvalue:.2e}, median diff={np.median(a['EXPANDING']-a['SLIDING']):.4f}")
    mon = d3.groupby(["strategy", "test"])["fpr"].mean().unstack("strategy"); P("E3 FPR by month (mean over models)"); P(mon.round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4))
    sc = {"STATIC": "#2a78d6", "SLIDING": "#eb6834", "EXPANDING": "#1baf7a"}; smk = {"STATIC": "o", "SLIDING": "s", "EXPANDING": "^"}
    xs = LATER[1:]; xx = np.arange(len(xs))
    for s_ in sc:
        t = d3[(d3.strategy == s_) & (d3.model.isin(["RF", "XGB", "LGBM", "DT"]))].groupby("test")
        ax[0].plot(xx, t.mcc.mean().reindex(xs).values, marker=smk[s_], ms=4, color=sc[s_], label=s_.capitalize())
        ax[1].plot(xx, t.fpr.mean().reindex(xs).values, marker=smk[s_], ms=4, color=sc[s_], label=s_.capitalize())
    ax[0].set_ylabel("MCC"); ax[1].set_ylabel("FPR (log)"); ax[1].set_yscale("log")
    for a_ in ax: a_.set_xticks(xx); a_.set_xticklabels([LAB[m] for m in xs], rotation=45, ha="right")
    ax[0].legend(fontsize=7); ax[0].set_title("(a) MCC, mean of tree ensembles + DT", fontsize=8); ax[1].set_title("(b) FPR, same models", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_retrain.png"), dpi=300); plt.close(fig)

# ---------------- E4 -----------------
if os.path.exists(os.path.join(R, "e4.csv")):
    d4 = pd.read_csv(os.path.join(R, "e4.csv")).set_index("month")
    if os.path.exists(os.path.join(R, "e4b.csv")):
        xb = pd.read_csv(os.path.join(R, "e4b.csv")).set_index("month")
        d4["psi_proto"] = xb["psi_proto_discrete"]
        d4["dom_auc_benign_noports"] = xb["dom_auc_benign_noports"]; d4["dom_auc_malicious_noports"] = xb["dom_auc_malicious_noports"]
        d4["psi_mean"] = d4[["psi_" + f for f in ["avg_ipt", "bytes_in", "bytes_out", "dest_port", "entropy", "num_pkts_out", "num_pkts_in", "proto", "src_port", "total_entropy", "duration"]]].mean(axis=1)
        d4.to_csv(os.path.join(R, "e4_merged.csv"))
    P("\nE4 drift"); P(d4[[c for c in ["mal_rate", "psi_mean", "dom_auc_all", "dom_auc_benign", "dom_auc_malicious", "dom_auc_benign_noports", "dom_auc_malicious_noports"] if c in d4.columns]].round(3).to_string())
    P(d4[[c for c in d4.columns if c.startswith("psi_") and c != "psi_mean"]].round(3).to_string())
    tmm = tm.reset_index(); tmm = tmm[tmm.test.isin(LATER)]
    ra = d1[d1.protocol == "RAND-ALL"].groupby("model")[["mcc", "fpr"]].mean()
    for met in ["mcc", "fpr"]:
        for dv in [c for c in ["psi_mean", "dom_auc_all", "dom_auc_benign", "dom_auc_malicious", "dom_auc_benign_noports", "dom_auc_malicious_noports", "mal_rate"] if c in d4.columns]:
            xs_, ys_ = [], []
            for _, r in tmm.iterrows():
                xs_.append(d4.loc[r.test, dv]); ys_.append(r[met] - ra.loc[r.model, met])
            s = spearmanr(xs_, ys_)
            P(f"Spearman {dv} vs delta-{met} (n={len(xs_)}): rho={s.statistic:.3f}, p={s.pvalue:.2e}")
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4))
    xx = np.arange(len(LATER) + 1); xl = [LAB[m] for m in LATER] + ["Jun-22"]
    idx = LATER + ["2022-06"]
    ax[0].plot(xx, d4.loc[idx, "dom_auc_benign"], marker="o", ms=4, color="#2a78d6", ls=":", label="Benign, all features")
    ax[0].plot(xx, d4.loc[idx, "dom_auc_benign_noports"], marker="o", ms=4, color="#2a78d6", label="Benign, no ports")
    ax[0].plot(xx, d4.loc[idx, "dom_auc_malicious_noports"], marker="s", ms=4, color="#eb6834", label="Malicious, no ports")
    ax[0].set_ylabel("Domain-classifier AUC vs T0"); ax[0].set_ylim(0.55, 1.02); ax[0].legend(fontsize=6, loc="lower left", ncol=1); ax[0].set_title("(a) Drift per class", fontsize=8)
    ax[1].plot(xx, d4.loc[idx, "mal_rate"], marker="o", ms=4, color="#6250d6")
    ax[1].axhline(d4["t0_mal_rate"].iloc[0], color="#999", ls="--", lw=0.8); ax[1].text(5.0, d4["t0_mal_rate"].iloc[0] + 0.012, "T0 prior", fontsize=6, color="#666")
    ax[1].set_ylabel("Malicious share"); ax[1].set_title("(b) Label prior", fontsize=8)
    for a_ in ax: a_.set_xticks(xx); a_.set_xticklabels(xl, rotation=45, ha="right"); a_.axvline(len(LATER) - 0.5, color="#999", lw=0.6, ls="--")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_drift.png"), dpi=300); plt.close(fig)

# ---------------- E5 -----------------
# Daily view uses the one-sample-per-day file (precise per-day estimates), scaled to true daily volume.
if os.path.exists(os.path.join(R, "e5_daily.csv")) and os.path.exists("data/luflow_sample.parquet"):
    dd = pd.read_csv(os.path.join(R, "e5_daily.csv"))
    cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
    raw = pd.read_parquet("data/luflow_sample.parquet", columns=["day", "label", "dest_port"])
    raw = raw[raw.label == "benign"]; raw["p9092"] = (pd.to_numeric(raw.dest_port, errors="coerce") == 9092)
    sh = raw.groupby("day").p9092.mean()
    dd = dd[dd.day < "2022-01-01"].copy()
    dd["benign"] = dd.day.map(cnt.benign); dd["outage"] = dd.day.map(cnt.malicious) == 0
    dd["false_alerts"] = dd.fpr * dd.benign
    dd["svc_flows"] = dd.day.map(sh) * dd.benign
    dd.to_csv(os.path.join(R, "e5_daily_abs.csv"), index=False)
    dd["day"] = pd.to_datetime(dd.day)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
    ax[0].plot(dd.day, dd.false_alerts.clip(lower=10), color="#1baf7a", lw=1)
    ax[0].set_yscale("log"); ax[0].set_ylabel("False alerts / day\n(RF, est.)")
    ax[1].plot(dd.day, dd.svc_flows, color="#2a78d6", lw=1); ax[1].set_ylabel("Benign flows / day\nto port 9092 (est.)")
    for a_ in ax:
        for t in dd.day[dd.outage]:
            a_.axvspan(t - pd.Timedelta(hours=12), t + pd.Timedelta(hours=12), color="#bbbbbb", alpha=0.35, lw=0)
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_episode.png"), dpi=300); plt.close(fig)
    P("\nE5 Spearman daily false alerts vs port-9092 benign flows:", spearmanr(dd.false_alerts, dd.svc_flows))
    ep = dd[(dd.day >= "2020-11-18") & (dd.day <= "2021-01-07")]; pre = dd[(dd.day >= "2020-08-01") & (dd.day < "2020-11-18")]
    post = dd[(dd.day > "2021-01-07")]
    P("E5 false alerts/day median pre/episode/post:", pre.false_alerts.median(), ep.false_alerts.median(), post.false_alerts.median())
    P("E5 port-9092 benign flows/day in episode: median", ep.svc_flows.median(), "min", ep.svc_flows.min(), "max", ep.svc_flows.max())
    P("E5 share of episode false alerts explained by port 9092 (est.):", (ep.svc_flows.sum()) / ep.false_alerts.sum())
out.close()
