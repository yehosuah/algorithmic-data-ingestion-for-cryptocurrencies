# Subtask 3 – Blender Refresh

## Procedure
The CLI runner stalled, so I executed the underlying pipeline directly:
```
.venv/bin/python - <<'PY'
from pathlib import Path
from training.data import load_parquet_dataset, ensure_labels
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn
from training.blender import train_blender, save_blender

matrix = ensure_labels(load_parquet_dataset('datasets/training_matrix_months_2025-08-09.parquet')).sort_values('timestamp').reset_index(drop=True)
base_calib, base_cols = load_base_predictor(Path('models/base_xgb_h120_turn200_v5'))
matrix['base_prob'] = predict_base(matrix, base_calib, base_cols)
model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path('models/tcn_cost_h120_turn200_ls'))
tcn_df = predict_tcn(matrix, model_tcn, calib_tcn, series_cols, scaler, window, stride=30)
bset = matrix.merge(tcn_df, on='timestamp', how='left').dropna(subset=['base_prob','tcn_prob','y_dir','ret_next']).reset_index(drop=True)
pipe, thr, rep, cols = train_blender(bset, cost_bps=5.0)
save_blender(Path('models/blender_h120_v2'), pipe, cols, thr, rep)
PY
```

## Result (`models/blender_h120_v2/report.json`)
- `final_equity` **0.9985**, `total_turnover` **2** at threshold `0.825`.
- Feature set reduced to `[base_prob, tcn_prob, rss_count, rss_sent_mean, rvol_5, rvol_20]` because Reddit aggregates were absent in the matrix slice.

## Findings
- Both `base_prob` and `tcn_prob` collapse to ≈0.50 on this matrix (max `base_prob` ≈ 0.4967, max `tcn_prob` ≈ 0.524), so the blender has no material signal to combine. This stems from the validation parquet lacking the enriched market feature set expected by `feature_list.json`.
- Added `--tcn-stride` to `scripts/train_blender.py` to keep future runs consistent with the TCN training stride; the manual path used the same stride (30).

## Remediation Plan
1. Regenerate `training_matrix_months_2025-08-09.parquet` with the full market feature bundle (or rebuild a fresh horizon-aware validation slice) so base/TCN probabilities are non-degenerate.
2. Re-run the blender once the matrix is rebuilt; target equity remains ≥1.2 with non-trivial turnover.

## Update – Full Matrix Attempt (`models/blender_h120_v3/report.json`)
- Rebuilt the validation slice with augmented features and persisted probabilities at `datasets/training_matrix_months_2025-08-09_full.parquet`.
- Retrained the blender using the gated XGB (`models/base_xgb_h120_turn200_v7`) and the symmetric TCN; manual run mirrors the script with `--tcn-stride 30`.
- Outcome unchanged: `final_equity` 0.9985, `total_turnover` 2. The base gate only fires ~70 times across the full year and almost never during Aug–Sep 2025, so the validation window still offers no overlap for stacking.
- Action: expand the validation matrix (longer horizon or alternative months) to capture enough co-occurrent TCN/XGB signals before revisiting blender training.
