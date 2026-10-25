#!/usr/bin/env bash
# Full reproduction, run from the repository root. Runtimes are for a 2-core machine.
# Steps 1-2 need the raw data (see README). Every later step only needs the files in data/.
set -euo pipefail
LUFLOW_RAW=${LUFLOW_RAW:-data/luflow_raw}
P=data/luflow_prop.parquet; S=data/luflow_sample.parquet; R=results_prop

# 1. Samples: 1% volume-proportional (pooled experiments) and 10,000 flows per day (daily experiments)
python3 src/prep.py "$LUFLOW_RAW" data 0 0.01 luflow_prop
python3 src/prep.py "$LUFLOW_RAW" data 10000

# 2. Pooled LUFlow experiments (Tables 1-6, 9, 10; Figs 1-4)
python3 src/table1.py
python3 src/e1_protocols.py $P $R/e1.csv drop
python3 src/e5_diagnose.py  $P $R/e5_monthly.csv
python3 src/e4_drift.py     $P $R/e4.csv
python3 src/e3_retrain.py   $P $R/e3.csv
python3 src/e2_matrix.py    $P $R/e2.csv
NSEEDS=3 python3 src/e1_protocols.py $P $R/e1_noports.csv drop noports
NSEEDS=3 python3 src/e1_protocols.py $P $R/e1_outlier.csv malicious
python3 src/e9_tuning.py    $P $R/e9.csv
LGBM_JOBS=1 python3 src/e10_dup_boot.py && python3 src/e10_analyze.py
# daily false-positive series for Fig 2 comes from the 10,000-per-day sample
python3 src/e5_diagnose.py $S results/e5.csv && cp results/e5_daily.csv $R/e5_daily.csv
RES=$R FIG=figures_prop python3 src/analyze.py

# 3. Daily-resolution LUFlow experiments (Tables 7, 8; Fig 5)
python3 src/e7_monitor.py $S results/e7_rf.csv RF dom
python3 src/e7_monitor.py $S results/e7_cheap.csv RF,XGB,LGBM,DT,MLP
python3 src/analyze_e7.py results/e7_rf.csv results/e7_cheap.csv
python3 src/e7b_baselines.py
python3 src/e8_response.py $S results/e7_rf.csv results/e8_rf0.csv RF 0
NS=200 MONITORS=uncertainty,port_surge,dom_all python3 src/e8_response.py $S results/e7_rf.csv results/e8_rfseeds.csv RF 1,2
NS=200 MONITORS=uncertainty,port_surge,dom_all python3 src/e8_response.py $S results/e7_rf.csv results/e8_xgb.csv XGB 0
NS=200 MONITORS=uncertainty,port_surge python3 src/e8_response.py $S results/e7_rf.csv results/e8_xgbseeds.csv XGB 1,2
LGBM_JOBS=1 NS=200 MONITORS=uncertainty,port_surge,dom_all python3 src/e8_response.py $S results/e7_rf.csv results/e8_lgbm.csv LGBM 0
LGBM_JOBS=1 NS=200 MONITORS=uncertainty,port_surge python3 src/e8_response.py $S results/e7_rf.csv results/e8_lgbmseeds.csv LGBM 1,2
python3 src/analyze_e8.py results/e8_rf0.csv results/e8_rfseeds.csv results/e8_xgb.csv results/e8_xgbseeds.csv results/e8_lgbm.csv results/e8_lgbmseeds.csv
python3 src/fig_e8.py

# 4. Kyoto 2006+ (Tables 11, 12; Fig 6). Needs the AnoShift *_subset.parquet files in data/kyoto/raw
python3 src/kyoto_e1.py results_kyoto/ke1.csv
python3 src/kyoto_analyze.py
python3 src/kyoto_diag.py
SEEDS=0 python3 src/kyoto_e8.py results_kyoto/ke8_s0.csv
python3 src/kyoto_fig.py
