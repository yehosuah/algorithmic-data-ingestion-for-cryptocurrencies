# Training Walkthrough (Base · TCN · Blender)

_Last updated: 2025-10-23 01:00 UTC_

This guide walks through the refreshed modeling stack: relaxed-gate retrains for the horizon-120 XGBoost baseline, the Calmon TCN suite, and the elastic-net blender that now clears 5 bps costs with RSS enrichment.

## Prerequisites
- Python 3.12 virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Ensure the year-wide market parquet and the latest RSS matrix are present:
  - `datasets/market_btcusdt_1m_2024_2025.parquet`
  - `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`
  - Optional forward replay matrix for gate audits: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`

## 1. Generate/Refresh Feature Matrices
The relaxed gate relies on augmented features and RSS spikes engineered by the new builder.
```bash
python scripts/build_blender_matrix.py \
  --source datasets/market_btcusdt_1m_2024_2025.parquet \
  --out datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --tcn-stride 30 --include-reddit --timeframe 1min
```
- Emits a JSON summary (`..._stats.json`) with RSS coverage diagnostics.
- Persists minute/day RSS features, probability momentum (`prob_diff`, `*_mom_1`), and gating fields aligned to the relaxed training mask.
- For forward audits we snapshot `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (Oct 1–Oct 21 2025, probabilities included); regenerate by rerunning the builder with the desired window and output path.

## 2. Train Horizon-120 XGBoost (Relaxed Gate)
```bash
python scripts/train_base_gbdt.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/base_xgb_h120_calmon_spread0 \
  --fold-scheme calendar_month --n-folds 6 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --threshold-criterion final_equity --diagnostic-thresholds 0.5,0.55,0.6,0.65 \
  --calmon-gate
```
- Artifacts: booster (`model.json`), calibrator, feature list, threshold, manifest (`gates.training` vs `gates.inference`), `report.json`.
- `report.json` captures monthly diagnostics, RSS audits, and spread stress-test metadata.

## 3. Train Calmon TCN Variants
```bash
python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_h120_calmon_relaxed \
  --window 192 --stride 30 --channels 64,64 \
  --epochs 10 --batch-size 256 --lr 5e-4 --dropout 0.1 \
  --class-weight 2.0 --n-folds 4 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --min-total-turnover 4 --max-total-turnover 200 \
  --horizon 120 --diagnostic-thresholds 0.55,0.6,0.65,0.7
```
- Produces `tcn.pt`, scaler/preprocess bundle, calibrator, manifest, threshold, `fold_logits.parquet`, and `report.json` with the probability variance guardrail.
- Repeat with `--horizon 60` / `--horizon 180` to populate the alternate manifests (`models/tcn_h60_calmon_relaxed`, `models/tcn_h180_calmon_relaxed`).

## 4. Train Elastic-Net Blender
```bash
python scripts/train_blender.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/blender_h120_v6 \
  --cost-bps 5 --tcn-stride 30 \
  --max-total-turnover 10000 --min-toggle-count 2 \
  --l1-ratio-grid 0.15 0.35 0.55 0.75 0.9
```
- Builds a StandardScaler + LogisticRegressionCV pipeline, sweeps thresholds, and records RSS audits, feature usage, and elastic-net weights.
- Result: `final_equity ≈ 1.84`, 711 toggles at threshold 0.95 with RSS spike gating automatically noted in `report.json`.

## 5. Compile Deployment Shortlist
```bash
python scripts/report_shortlist.py \
  --models-root models \
  --min-equity 1.05 --min-turnover 10 --min-sharpe 0 \
  --out models/report_shortlist.json
```
- Reads every `report.json`, enforces KPI schema, filters by RSS audit pass/fail, and ranks candidates (base, TCN, blender).
- The shortlist feeds deployment review or CI checks.

## 6. Forward Replay & Gate Audit
- Review `models/oos_replay_oct_nov_2025.json` (Oct 1–Oct 21 window) to compare training-vs-inference gate metrics across base, TCN, and blender models.
- The relaxed gates retain equity >1 while the deployable mask currently produces zero toggles; adjust thresholds or introduce a fallback path before promoting to production.

## 7. Run Regression Suites
- Ensure manifests stay aligned with their reports and the shortlist keeps surfacing deployable candidates:
  ```bash
  pytest tests/regression -q
  ```
- Exercise async ingestion routes end-to-end with fakeredis before packaging artifacts:
  ```bash
  pytest tests/ingestion_service -q
  ```

## 8. Optional – Meta Label Refresh
- Meta gating still lacks a stable decision surface; rerun `scripts/train_meta_label.py` only after extending the blender matrix to newer months.
- Leverage the existing manifests for deployable gates in live inference until a calibrated meta filter clears the ≥1.2 equity hurdle.
