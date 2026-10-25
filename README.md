# Random Splits Hide False-Alarm Bursts: reproduction package

Code and results for the article

> M. S. Alsayfi, "Random Splits Hide False-Alarm Bursts: Temporal Evaluation, Label-Free Early Warning and Label-Efficient Updating of Machine-Learning Network Intrusion Detection on Real Traffic," submitted to *IEEE Access*, 2026.

The study compares random and temporal train–test splits for six ML classifiers on two real-traffic NIDS datasets, LUFlow and Kyoto 2006+. It traces the temporal loss to changes in benign traffic, evaluates label-free drift monitors, and measures how many labels a drift-triggered update needs.

Repository: <https://github.com/majedalsayfi/nids-temporal-drift>

Every table and figure in the article can be regenerated from the scripts in `src/`. All result files used in the article are included, so the tables can be checked without rerunning the experiments.

## Repository layout

| Path | Content |
| --- | --- |
| `src/` | Experiment and analysis scripts (Python 3) |
| `data/` | Per-day label counts of LUFlow (used to scale daily estimates). Raw and sampled data are not redistributed |
| `results_prop/` | Results on the 1% volume-proportional LUFlow sample (E1–E5, E9, E10) |
| `results/` | Results on the 10,000-flows-per-day LUFlow sample (E5 daily series, E7, E8) |
| `results/superseded_equal_per_day/` | First analysis with equal-per-day sampling, kept for transparency (see the article, contribution 5). Not used in any table |
| `results_kyoto/` | Kyoto 2006+ results |
| `figures_prop/` | The six figures of the article (300 dpi PNG) |
| `run_all.sh` | Full reproduction in the order used for the article |

## Environment

Python 3.11 or later. Install the exact versions used in the article:

    pip install -r requirements.txt

All runs used a 2-core machine with 8 GB of RAM. Set `LGBM_JOBS=1` on small machines to avoid OpenMP oversubscription with LightGBM.

## Data

**LUFlow** (Lancaster University). Download the daily files from <https://github.com/ruzzzzz/LUFlow> into `data/luflow_raw/`. We use every day from 19 June 2020 to 17 February 2021 and the three days of June 2022. `src/prep.py` builds two samples:

- `data/luflow_prop.parquet`: a uniform 1% sample of every day (volume-proportional; used for all pooled experiments);
- `data/luflow_sample.parquet`: 10,000 flows per day (used for daily-resolution experiments, scaled back to true volume with `data/luflow_daily_counts.csv`).

Sampling is seeded by the date, so the samples are reproducible.

**Kyoto 2006+** (preprocessed AnoShift release). Download the yearly `*_subset.parquet` files of AnoShift (<https://github.com/bit-ml/AnoShift>) into `data/kyoto/raw/`, or set `KYOTO_RAW` to their folder. `src/kyoto_common.py` samples up to 25,000 flows per month and caches the sample in `data/kyoto/kyoto_sample.parquet`.

## Reproducing the article

Run from the repository root:

    bash run_all.sh

or run the steps of `run_all.sh` one by one. The mapping from article items to scripts and result files is:

| Article item | Script(s) | Result file(s) |
| --- | --- | --- |
| Table 1 (monthly composition) | `prep.py`, `table1.py` | `results_prop/table1.csv` |
| Tables 2–3, Fig. 1 (random vs temporal) | `e1_protocols.py`, `analyze.py` | `results_prop/e1.csv`, `results_prop/summary.txt` |
| Table 3 confidence intervals | `e10_dup_boot.py`, `e10_analyze.py` | `results_prop/e10_summary.csv` |
| Table 4, Fig. 2 (false positives by port, daily episode) | `e5_diagnose.py`, `analyze.py` | `results_prop/e5_monthly.csv`, `results/e5_daily.csv` |
| Fig. 3 (train-month × test-month matrix) | `e2_matrix.py`, `analyze.py` | `results_prop/e2.csv` |
| Table 5 (drift measures) | `e4_drift.py`, `analyze.py` | `results_prop/e4.csv` |
| Table 6, Fig. 4 (static, sliding, expanding) | `e3_retrain.py`, `analyze.py` | `results_prop/e3.csv` |
| Table 7 (label-free monitors) | `e7_monitor.py`, `analyze_e7.py`, `e7b_baselines.py` | `results/e7_scores.csv`, `results/e7b_scores.csv` |
| Table 8, Fig. 5 (update strategies on the daily stream) | `e8_response.py`, `analyze_e8.py`, `fig_e8.py` | `results/e8_summary.csv` |
| Table 9 (robustness: no ports, outliers, duplicates) | `e1_protocols.py` (variants), `e10_dup_boot.py` | `results_prop/e1_noports.csv`, `results_prop/e1_outlier.csv`, `results_prop/e10_dup.csv` |
| Table 10 (hyper-parameter tuning) | `e9_tuning.py` | `results_prop/e9.csv` |
| Table 11, Fig. 6 (Kyoto random vs temporal) | `kyoto_e1.py`, `kyoto_analyze.py`, `kyoto_diag.py`, `kyoto_fig.py` | `results_kyoto/ke1.csv`, `results_kyoto/ke1_perseed.csv`, `results_kyoto/ke5.csv` |
| Table 12 (Kyoto update strategies) | `kyoto_e8.py` | `results_kyoto/ke8_s0.csv`, `results_kyoto/ke8_summary.csv` |

## Experimental protocol in brief

- **Features (LUFlow):** 11 flow features; time stamps and IP fields are excluded (the source AS is a label proxy in LUFlow).
- **Protocols:** RAND-ALL (stratified random 70/30 split over Jun-2020 to Feb-2021), RAND-T0 (random split inside Jun–Jul 2020), TEMP (train on Jun–Jul 2020, test on every later month). Training sets are capped at 150,000 flows for every protocol.
- **Models:** LR, DT, RF, XGBoost, LightGBM, MLP with fixed hyper-parameters (`src/common.py`); decision threshold 0.5.
- **Seeds:** five for E1 (0–4), three for E3, the variants and the E8 runs; seeds change both the split and the model's randomness.
- **Daily estimates:** if a model flags f of the b benign flows sampled on a day with B benign flows in the full release, the estimated number of false alerts is f·B/b.

## Notes

- The equal-per-day sample in `results/superseded_equal_per_day/` over-weighted eleven low-volume collection-gap days and overstated the December FPR about fivefold. The article reports this correction; all article numbers come from the volume-proportional sample or from volume-scaled daily estimates.
- In `results_prop/e4.csv`, the `psi_proto` column uses quantile bins; `analyze.py` reports the value-level version.

## License and citation

Code is released under the MIT License (see `LICENSE`). The datasets remain under the terms of their original providers. If you use this package, please cite the article (see `CITATION.cff`).

Contact: Majed S. Alsayfi, Department of Cybersecurity, College of Computer Science and Engineering, Taibah University, Medina, Saudi Arabia (msayfi@taibahu.edu.sa).
