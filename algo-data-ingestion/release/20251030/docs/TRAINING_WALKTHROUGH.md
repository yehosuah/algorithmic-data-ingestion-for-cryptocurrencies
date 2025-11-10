# Training Walkthrough (Base · TCN · Blender)

_Last updated: 2025-11-10 04:13 UTC (archival snapshot refreshed)_

> Update 2025-11-10: Refreshed the walkthrough notes to mention the scheduler-driven inference jobs and trading dry-run dashboards that accompany the release/20251030 drop.

This guide walks through the refreshed modeling stack: relaxed-gate retrains for the horizon-120 XGBoost baseline, the Calmon TCN suite, and the elastic-net blender that now clears 5 bps costs with RSS enrichment.

_Update 2025-11-05_: The active branch adds scheduler-managed inference (`INFER_JOBS`) and a trading dry run (`app/trading/service.py`). Consult `TRAINING_WALKTHROUGH.md` at repo root for the augmented steps covering Redis decision queues, Prometheus trading metrics, and dry-run validation.

## Prerequisites
- Python 3.12 virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Ensure the year-wide market parquet and the latest RSS matrix are present:
  - `datasets/market_btcusdt_1m_2024_2025.parquet` (2024-01-01 ➜ 2025-10-27, 959 039 bars).
  - `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (alias `..._2025-10_rss_latest.parquet`, 606 121 rows covering 2024-09-01 ➜ 2025-10-26).
  - Optional forward replay matrix for gate audits: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (28 681 rows, 2025-10-01 ➜ 2025-10-20 22:00).

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
- Emits a JSON summary (`..._stats.json`) with window start/end and probability means for quick sanity checks.
- Persists minute/day RSS features, probability momentum (`prob_diff`, `*_mom_1`), and gating fields aligned to the relaxed training mask.
- The matrix cadence (set by `--tcn-stride` or inferred from timestamps) now controls the blender gate smoothing window, so choose the stride you expect to honour at inference.
- For forward audits we snapshot `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (Oct 1–Oct 20 2025, probabilities included); regenerate by rerunning the builder with the desired window and output path.

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
- Artifacts: booster (`model.json`), calibrator, feature list, threshold, manifest (`gates.training` vs `gates.inference`), `report.json`; the deployable mask now defaults to `hl_spread ≤ 7e-4`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`.
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
- Gate masks are smoothed over the stride detected above (defaults to the TCN stride); the chosen window is emitted as `gate_smoothing_stride` in the report, and sandboxed stride‑1 runs (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) illustrate the turnover impact of reducing smoothing.
- CLI adds `--class-weight {balanced,none}` (default balanced) and treats `--calibration-cv <= 1` as “no calibration”; the manifest gates inference at `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`, restoring ≈16 % deployable coverage on Oct 2025 data.

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
- Review `models/oos_replay_summary_latest.json` (Oct 1–Oct 27 window) alongside the forward matrix to compare training-vs-inference gate metrics across base, TCN, and blender models.
- The retuned base manifest now logs 12 deployable gate hits (8 trades, `final_equity 1.23`) and the blender fires 5 870 toggles (`gate_coverage ≈ 16 %`); all TCN manifests remain idle, so widen their thresholds or stage a fallback before promoting to production.
- Wire the replay into monitoring by feeding batches through `training/infer.py::score_base_with_manifest`; the helper updates Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma`) so coverage drift and probability variance show up on dashboards.
- `training/infer.predict_tcn` now processes inference in stride-aware batches, preventing memory spikes when experimenting with smaller strides (e.g., the stride‑1 blender tests).

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
