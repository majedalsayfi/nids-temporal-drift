"""E15 analysis: monitor generality across seeds, events, calibration windows, uncertainty bands and burst thresholds.
Definitions
  estimated false alerts on day d = fp_d / n_benign_d * B_d (B_d = benign flows of day d in the full release)
  burst day (threshold K) = estimated false alerts > K x median over the August-2020 calibration days
  event = maximal run of burst days in Sep-2020..Feb-2021, runs separated by at most 2 non-burst days are merged
  alarm = monitor value > its maximum over the calibration window (last 7, 14 or 31 days of August 2020)
  an event is detected if an alarm occurs between 3 days before its first day and its last day;
  delay = first such alarm day minus first event day (negative = early warning)
  false-alarm rate = share of non-burst days with an alarm
Matched budget: threshold set to the 95th percentile of the monitor over non-burst days (false-alarm rate 5%),
an evaluation-only comparison that removes calibration differences between monitors."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
m = pd.read_csv("results/e15_monitors.csv")
cnt = pd.read_csv("data/luflow_daily_counts.csv").set_index("day")
m["est_fa"] = m.fp / m.n_benign.clip(lower=1) * m.day.map(cnt.benign)
MON = ["alert_rate", "unc_10_90", "unc_20_80", "unc_30_70", "port_surge", "mean_score", "ks_score"]
EPI = ("2020-11-18", "2021-01-07")


def events(days, burst):
    ev, cur, gap = [], None, 0
    for d, b in zip(days, burst):
        if b:
            if cur is None: cur = [d, d]
            else: cur[1] = d
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > 2: ev.append(tuple(cur)); cur = None; gap = 0
    if cur is not None: ev.append(tuple(cur))
    return ev


def evaluate(g, mon, thr, ev, burst):
    days = g.day.values; al = g[mon].values > thr
    det, delays = 0, []
    for a, b in ev:
        start = pd.Timestamp(a)
        win = [(pd.Timestamp(d) - start).days for d, x in zip(days, al)
               if x and pd.Timestamp(a) - pd.Timedelta(days=3) <= pd.Timestamp(d) <= pd.Timestamp(b)]
        if win: det += 1; delays.append(min(win))
    nb = ~burst
    return det, (np.median(delays) if delays else np.nan), al[nb].mean() if nb.any() else np.nan, al.sum()


rows = []
for (model, seed), g in m.groupby(["model", "seed"]):
    g = g.sort_values("day"); cal = g[g.day < "2020-09-01"]; ev_ = g[g.day >= "2020-09-01"]
    for K in (3, 5, 10):
        burst = (ev_.est_fa > K * cal.est_fa.median()).values
        if burst.sum() < 3 or burst.all(): continue
        E = events(ev_.day.values, burst)
        epi = [e for e in E if e[0] <= EPI[1] and e[1] >= EPI[0]]
        for mon in MON:
            auc = roc_auc_score(burst, ev_[mon])
            early = ev_.day < "2020-11-01"; bE = burst[early.values]
            auc_early = roc_auc_score(bE, ev_[mon][early]) if 0 < bE.sum() < len(bE) else np.nan
            late = ~early.values; bL = burst[late]
            auc_late = roc_auc_score(bL, ev_[mon][late]) if 0 < bL.sum() < len(bL) else np.nan
            for win in (7, 14, 31):
                thr = cal[mon].tail(win).max()
                det, dly, far, nal = evaluate(ev_, mon, thr, E, burst)
                det_e, dly_e, _, _ = evaluate(ev_, mon, thr, epi, burst)
                rows.append(dict(model=model, seed=seed, K=K, monitor=mon, calib=win, setting="calibrated",
                                 auc=auc, auc_sep_oct=auc_early, auc_nov_feb=auc_late, n_events=len(E),
                                 detected=det, median_delay=dly, episode_detected=det_e, episode_delay=dly_e,
                                 false_alarm_rate=far, alarm_days=nal, burst_days=int(burst.sum())))
            thr = np.quantile(ev_[mon].values[~burst], 0.95)
            det, dly, far, nal = evaluate(ev_, mon, thr, E, burst)
            det_e, dly_e, _, _ = evaluate(ev_, mon, thr, epi, burst)
            rows.append(dict(model=model, seed=seed, K=K, monitor=mon, calib=np.nan, setting="matched-5%",
                             auc=auc, auc_sep_oct=auc_early, auc_nov_feb=auc_late, n_events=len(E), detected=det,
                             median_delay=dly, episode_detected=det_e, episode_delay=dly_e,
                             false_alarm_rate=far, alarm_days=nal, burst_days=int(burst.sum())))
R = pd.DataFrame(rows); R.to_csv("results/e15_eval.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
main = R[(R.K == 5) & (R.setting == "calibrated") & (R.calib == 31)]
print("== K=5, August calibration: mean over seeds (min-max AUC)")
print(main.groupby(["model", "monitor"]).agg(auc=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"),
      auc_sep_oct=("auc_sep_oct", "mean"), auc_nov_feb=("auc_nov_feb", "mean"), events=("n_events", "mean"),
      detected=("detected", "mean"), ep_det=("episode_detected", "mean"), ep_delay=("episode_delay", "median"),
      far=("false_alarm_rate", "mean")).round(3).to_string())
mt = R[(R.K == 5) & (R.setting == "matched-5%")]
print("== K=5, matched false-alarm rate 5%")
print(mt.groupby(["model", "monitor"]).agg(detected=("detected", "mean"), events=("n_events", "mean"),
      ep_det=("episode_detected", "mean"), ep_delay=("episode_delay", "median")).round(2).to_string())
print("== sensitivity (RF, uncertainty bands and port surge): AUC by K and episode detection by calibration window")
s = R[(R.setting == "calibrated") & R.monitor.isin(["unc_10_90", "unc_20_80", "unc_30_70", "port_surge"])]
print(s.groupby(["model", "monitor", "K", "calib"]).agg(auc=("auc", "mean"), ep_det=("episode_detected", "mean"),
      ep_delay=("episode_delay", "median"), far=("false_alarm_rate", "mean")).round(3).to_string())
# selection on earlier data: pick the monitor with the best Sep-Oct AUC per model and seed, evaluate on Nov-Feb
sel = main.dropna(subset=["auc_sep_oct"]).sort_values("auc_sep_oct").groupby(["model", "seed"]).tail(1)
print("== monitor selected on Sep-Oct (K=5), evaluated on Nov-Feb")
print(sel[["model", "seed", "monitor", "auc_sep_oct", "auc_nov_feb", "episode_detected", "episode_delay"]].to_string())
