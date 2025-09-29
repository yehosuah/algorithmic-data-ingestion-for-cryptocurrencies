# Model Training Status (XGB & TCN)

_Last updated: 2025-09-29 02:05 UTC_

## Data Baseline
- Dataset: `datasets/market_btcusdt_1m_2024_2025.parquet`
- Feature set: original market features + augmented signals (`training/feature_eng.py`)
- Cost assumptions referenced below: 5 bps transaction cost, 0 slippage, zero spread scaling unless otherwise stated.

## XGBoost Classifier (Base Model)
- Latest artifacts: `models/base_xgb_tuned_features_cost` (with costs) and `models/base_xgb_tuned_features_nocost` (zero cost sanity check).
- **Performance snapshots**
  - With costs (`models/base_xgb_tuned_features_cost/report.json`):
    - `final_equity` = **0.9546**, `sharpe` = -1.85, `threshold` = 0.915, `oof_auc` ≈ 0.533, total trades = 80.
  - Zero cost (`models/base_xgb_tuned_features_nocost/report.json`):
    - `final_equity` = **1.1548**, `sharpe` = 0.68, `threshold` = 0.55, turnover ≈ 11.5%.
- **Observation**: Feature augmentation improves raw signal (positive equity without costs), but 5 bps costs still drag the model below break-even. Prior `hl_spread_z` cost scaling at 0.5x wiped out returns (equity ≈ 0), so future gating needs smaller scaling or hard filters.

### Next Steps (XGB)
1. **Regularisation Sweep**: Try shallower depth (4–5), adjust `reg_lambda`/`reg_alpha`, and monitor how the calibrated threshold shifts; aim for a workable threshold <0.9 under costs.
2. **Cost-aware Filtering**: Re-run spread-aware costs with smaller scaling (0.05–0.10) or explicit trade gating on high `hl_spread_z` bars before thresholding.
3. **Metric Diagnostics**: Capture full threshold-grid PnL/turnover traces per fold to spot calibration drift or regimes with flat equity curves.
4. **Regime/Turnover Controls**: Evaluate monthly/volatility splits; consider per-fold thresholds or volatility regime filters before moving to blender.

## Temporal Convolutional Network (TCN)
- Latest artifacts: `models/tcn_tuned` (5 bps costs, stride=5) and `models/tcn_tuned_nocost` (zero cost).
- **Performance snapshots**
  - With costs (`models/tcn_tuned/report.json`):
    - `final_equity` = **0.9753**, `sharpe` = -9.16, `threshold` = 0.80, total trades = 48 (low turnover due to stride).
  - Zero cost (`models/tcn_tuned_nocost/report.json`):
    - `final_equity` = **1.0248**, `sharpe` = 1.30, `threshold` ≈ 0.575, turnover ≈ 2.6%.
- **Observation**: Stride-based windows and expanded channel counts reduced runtime and improved raw edge, but per-trade costs still keep equity <1.00; calibration is sensitive to fold coverage with the lower sample count.

### Next Steps (TCN)
1. **Capacity/Training Adjustments**: Increase channels (e.g., `64,64`) and epochs; add early stopping or LR scheduling to stabilise calibration while keeping stride at 5.
2. **Class Weighting Check**: Compare runs with/without `--class-weight` to ensure positive-class weighting isn’t suppressing high-confidence predictions.
3. **Per-window Normalisation Audit**: Inspect tensors; test `standardize_per_window=False` or hybrid scaling to preserve slower trends (`ret_mean_20`, `macd_hist`).
4. **Cost-aware Thresholding**: Add spread gating or adaptive cost scaling before rerunning threshold selection to protect against high-cost bars.

## General Follow-up
- Keep blender/meta-label training on hold until post-cost equity exceeds 1.0 for at least one base model; command templates live in `scripts/train_blender.py` and `scripts/train_meta_label.py` once unblocked.
- Continue logging outcomes (report JSONs) alongside parameter settings so we can track which adjustments move the needle.
- Capture diagnostics (threshold grids, turnover, per-fold AUC) in notebooks or `structure_exports/` for quicker iteration reviews.
