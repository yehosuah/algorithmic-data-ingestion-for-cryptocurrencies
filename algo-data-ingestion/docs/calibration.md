# Probability Calibration Refresh (Nov 2025)

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Folded in the sanitizer-driven multi-symbol feed plus the parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so calibration refreshes cite the same symbol-aware gates and drift checks now enforced by scheduler/trading.

## Overview

- **Issue:** live probabilities from the base XGB, TCN, and blender manifests were saturating near 0/1, leaving the dual-threshold gate almost binary.
- **Dataset:** `datasets/blender_matrix_2025-09_to_2025-11_oos.parquet` rebuilt via `scripts/build_blender_matrix.py` from the latest feature pipeline outputs (market + RSS features, base + TCN scores).
- **Gate payload:** `release/symbol_gates/market_multi_3symbol_1m.json` (generated via `scripts/compute_symbol_gate_config.py`) anchors the per-symbol `hl_spread`, `rvol`, and liquidity caps that scheduler/trading enforce alongside the calibrated probabilities.
- **Workflow:** `scripts/refresh_calibration.py` now fits post-hoc calibrators, emits reliability/histogram plots, and persists the calibrator parameters under `models/<manifest>/calibration/`.
- **Artifacts:** metrics + plots land in `release/calibration/2025-11-oos_stride2/` and are referenced by `release/calibration/2025-11-oos_stride2/calibration_summary.json`.

## Running the refresh

```bash
# 1) Rebuild the OOS matrix with up-to-date features/preds
python3 scripts/build_blender_matrix.py \
  --source datasets/market_btcusdt_1m_2024_2025.parquet \
  --out datasets/blender_matrix_2025-09_to_2025-11_oos.parquet \
  --base-dir models/base_xgb_h120_calmon_spread005 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --start-date 2025-09-01 --end-date 2025-11-03 \
  --tcn-stride 2

# 2) Fit calibrators + export diagnostics
python3 scripts/refresh_calibration.py \
  --data datasets/blender_matrix_2025-09_to_2025-11_oos.parquet \
  --base-model models/base_xgb_h120_calmon_spread005 \
  --tcn-model models/tcn_h120_calmon_relaxed \
  --blender-model models/blender_h120_v6 \
  --split-ratio 0.65 \
  --out-dir release/calibration/2025-11-oos_stride2
```

Outputs:

| Manifest | Method | Brier (→ lower) | ECE (→ lower) |
|----------|--------|-----------------|---------------|
| `base_xgb_h120_calmon_spread005` | Platt | 6.998e-4 → **6.996e-4** | 1.036e-4 → **1.021e-4** |
| `tcn_h120_calmon_relaxed` | Isotonic blend (w=0.6) | 5.750e-2 → **5.737e-2** | 1.646e-2 → **1.247e-2** |
| `blender_h120_v6` | Identity (best) | 1.800e-2 (unchanged) | 2.694e-2 (unchanged) |

`release/calibration/2025-11-oos_stride2/` contains the refreshed metrics (`*_metrics.json`), calibration plots, and the aggregate `calibration_summary.json`. The saved TCN calibrator is now an isotonic/identity blend (`isotonic_blend` with weight 0.6) that smooths the extreme probabilities uncovered by the denser stride‑2 replay.
Reliability curves (`*_reliability.png`) & histograms (`*_hist.png`) in the release folder illustrate the spread before/after calibration.

## Inference & gating integration

- `training/infer.load_base_predictor` / `predict_base` and `load_tcn_predictor` / `predict_tcn` automatically detect `calibration/<prob_col>.json` under each manifest and apply the saved calibrator at inference time (scheduler jobs, scoring API, dataset builders, etc.).
- `app.ingestion_service.scoring.BlenderRunner` now loads the blender calibrator and emits calibrated probabilities to downstream gates.
- The blender manifest (`models/blender_h120_v6/manifest.json`) now enforces `prob_gate_min = 0.55` for both training and inference sections, and `threshold.txt` has been updated to `0.55`. This matches the `select_prob_threshold` sweep on the calibrated probabilities (gate coverage ≈19.4%, final-equity criterion 4.48x).
- After any calibration refresh, export a scheduler slice and diff it via `scripts/compare_feature_stats.py` so the release folder (e.g., `release/calibration/latest/feature_parity.json`) documents live vs training drift across `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` before manifests are widened.

To disable a calibrator, remove the corresponding JSON/joblib pair from `models/<manifest>/calibration/` and re-run inference.
