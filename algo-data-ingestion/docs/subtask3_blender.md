# Subtask 3 – Blender Refresh

## Goal
Train a logistic blender combining the horizon-120 base model and tightened TCN outputs (plus RSS features) with 5 bps transaction costs.

## Workflow
1. Generate base and TCN probabilities on `datasets/training_matrix_months_2025-08-09.parquet` using the refreshed models (`models/base_xgb_h120_turn200`, `models/tcn_cost_h120_turn200`).
2. Feed the merged dataset into `training.blender.train_blender` to learn a logistic regression with standardized features.

## Command
Executed manually (due to long runtime) via:
```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from training.data import load_parquet_dataset, ensure_labels
from training.feature_eng import augment_market_features
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn
from training.blender import train_blender, save_blender

matrix_path = Path('datasets/training_matrix_months_2025-08-09.parquet')
base_dir = Path('models/base_xgb_h120_turn200')
tcn_dir = Path('models/tcn_cost_h120_turn200')
out_dir = Path('models/blender_h120')

# load + enrich
df = load_parquet_dataset(matrix_path)
df = ensure_labels(df)
df = augment_market_features(df)
df = df.sort_values('timestamp').reset_index(drop=True)

calib_base, feat_cols = load_base_predictor(base_dir)
df['base_prob'] = predict_base(df, calib_base, feat_cols).values

model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(tcn_dir)
tcn_df = predict_tcn(df, model_tcn, calib_tcn, series_cols, scaler, window)
merged = df.merge(tcn_df, on='timestamp', how='left').dropna(subset=['base_prob','tcn_prob','y_dir','ret_next']).reset_index(drop=True)

pipe, thr, rep, cols = train_blender(merged, cost_bps=5.0)
save_blender(out_dir, pipe, cols, thr, rep)
print({'threshold': thr, 'report': rep, 'features': cols})
PY
```

## Result (`models/blender_h120/report.json`)
- `final_equity`: **1.0119**
- `sharpe`: ~1.74
- `total_turnover`: 22 (long/short symmetric)
- `selected_threshold`: 0.675
- Features used: `['base_prob', 'tcn_prob', 'rss_count', 'rss_sent_mean', 'rvol_5', 'rvol_20']`

## Notes
- Running the packaged script timed out; the manual invocation reproduces equivalent logic but avoids re-running TCN predictions on every call.
- Future improvement: persist `tcn_prob` alongside the training matrix to keep the official CLI responsive.
