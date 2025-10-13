# Subtask 4 – Meta-Label Training Refresh

## Procedure
```
.venv/bin/python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from training.data import load_parquet_dataset, ensure_labels
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn
from training.meta import triple_barrier_events, rolling_vol
from training.metrics import equity_curve, summary_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

matrix = ensure_labels(load_parquet_dataset('datasets/training_matrix_months_2025-08-09.parquet')).sort_values('timestamp').reset_index(drop=True)
matrix['timestamp'] = pd.to_datetime(matrix['timestamp'], utc=True)
base_calib, base_cols = load_base_predictor(Path('models/base_xgb_h120_turn200_v5'))
matrix['base_prob'] = predict_base(matrix, base_calib, base_cols)
model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path('models/tcn_cost_h120_turn200_ls'))
tcn_df = predict_tcn(matrix, model_tcn, calib_tcn, series_cols, scaler, window, stride=30)
matrix = matrix.merge(tcn_df, on='timestamp', how='left')
close = matrix.set_index('timestamp')['close'].astype(float)
vol = rolling_vol(np.log(close).diff())
events = triple_barrier_events(close, pt_mult=1.5, sl_mult=2.0, max_hold=180, vol=vol)
labels = events['label'].reindex(close.index).ffill().fillna(0).astype(int)
feat_cols = [c for c in ['base_prob','tcn_prob','rvol_5','rvol_20','rss_count','rss_sent_mean','reddit_count','reddit_sent_mean'] if c in matrix.columns]
X = matrix[feat_cols].astype(float)
mask = X.notna().all(axis=1)
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
pipe.fit(X[mask].values, labels.values[mask])
raw_prob = pipe.predict_proba(X[mask].values)[:,1]
dn = matrix.loc[mask].reset_index(drop=True)
best = {'meta_threshold': 0.5, 'final_equity': -np.inf}
for thr in np.linspace(0.5, 0.9, 9):
    mask_keep = raw_prob >= thr
    prob_for_trading = dn['base_prob'].values.copy()
    prob_for_trading[~mask_keep] = 0.5
    rep = summary_stats(equity_curve(dn['ret_next'], pd.Series(prob_for_trading, index=dn.index), threshold=0.6, cost_bps=5.0))
    if rep['final_equity'] > best['final_equity']:
        best = {**rep, 'meta_threshold': float(thr)}
out = Path('models/meta_h120_v2')
out.mkdir(exist_ok=True, parents=True)
joblib.dump(pipe, out / 'meta_model.joblib')
(out / 'features.txt').write_text('\n'.join(feat_cols))
(out / 'meta_threshold.txt').write_text(str(best['meta_threshold']))
import json
(out / 'report.json').write_text(json.dumps({**best, 'pt_mult': 1.5, 'sl_mult': 2.0, 'max_hold': 180, 'label_pos_frac': float(labels.values[mask].mean())}, indent=2))
PY
```

## Result (`models/meta_h120_v2/report.json`)
- `final_equity` **1.00**, `total_turnover` **0**; meta threshold defaults to `0.5` because the base probability stream sits at ~0.497 on this matrix.
- Positive label share after the revised mapping: ~0.54 (balanced), so the barrier definition is now usable even though the downstream trading layer needs a non-degenerate primary signal.

## Updates
- `scripts/train_meta_label.py` gains `--tcn-stride` and now constructs the triple-barrier events on a timestamp-indexed close series to avoid the previous all-zero label issue.
- Residual warnings stem from deprecated pandas `fillna(method=...)` inside `training/meta.py`; a follow-up refactor is noted but not part of this subtask.

## Outstanding Work
1. Restore informative `base_prob`/`tcn_prob` on the validation matrix (see Subtask 3) so the meta gate can actually filter trades.
2. Once the matrix is rebuilt, re-run the script to pick a threshold that beats the 1.2 equity hurdle against 5 bps costs.

## Update – TCN-Primary Gating Attempt (`models/meta_h120_v3/report.json`)
- After rebuilding the validation matrix (`…_full.parquet`) and persisting `tcn_prob`, retried meta training with `tcn_prob` as the primary decision stream.
- Triple-barrier labels now cover ~54 % positives, but the 2054-row overlap still collapses to a single class when filtered for valid RSS/Reddit features, causing logistic fitting to fail (`ValueError: only one class present`).
- Action: extend the validation window or relax the feature completeness requirement before the next meta refresh; until then, the production gate should rely on the deterministic XGB/TCN filters captured in Subtask 2 and Subtask 1.
