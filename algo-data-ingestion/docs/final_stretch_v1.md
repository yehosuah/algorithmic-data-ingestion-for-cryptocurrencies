# Final Stretch – Production Checklist (Calmon Stack)

Last updated: 2025-10-21 02:50 UTC

Scope: Align the relaxed-gate Horizon-120 XGB, Calmon TCN suite, and elastic-net blender for a deployable release, with manifest-driven governance and monitoring.

## 1. Pre-Flight
- Freeze `datasets/market_btcusdt_1m_2024_2025.parquet` and `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (record commit hash + SHA in release notes).
- Persist monthly gate coverage snapshots for base and TCN (`live_gate_coverage.csv`, OOS gate summaries) and attach to the release package.
- Ensure manifests include deployable gates, thresholds, feature lists, and RSS audits; treat them as the contract between training and inference.

## 2. Base XGBoost (H120 Calmon)
- **Status**: `final_equity 4.48`, Sharpe 108, relaxed gate coverage 9.4 %. Deployable inference mask: `hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold = 10`.
- **Actions**
  1. Replay Oct–Nov 2025 with the inference mask to confirm turnover stays <0.02 % and equity >1.2.
  2. Add regression tests that run `training/infer.py` with manifest gates and compare against `report.json` KPIs (equity, turnover, threshold).
  3. Export coverage alert configs (±2× baseline) for production monitoring.

## 3. TCN Suite (H60/H120/H180 Calmon)
- **Status**: All horizons clear 5 bps costs (`final_equity` 1.05–1.33) with ≤200 toggles and matching inference gates.
- **Actions**
  1. Validate forward months with the deployable mask to catch variance collapse.
  2. Bundle `fold_logits.parquet`, calibrator, scaler, and manifests for each horizon; document when to switch horizons.
  3. Integrate probability σ guardrail (alert <0.03) into monitoring.

## 4. Elastic-Net Blender (H120)
- **Status**: `final_equity 1.84`, Sharpe 28.7, 711 toggles at threshold 0.95 with RSS spike gate share ≈2 %.
- **Actions**
  1. Verify RSS coverage alerts (minute spike share ≥5e-4, daily coverage ≥0.8). Configure automatic fallback to no-RSS feature set.
  2. Document feature weights, probability momentum, and gating logic so live scoring can match the training pipeline.
  3. Replay Oct–Nov 2025; confirm equity ≥1.2 and turnover within ±25 % of the current run before sign-off.

## 5. Meta Label (Deferred)
- Training still collapses due to limited dynamic range. Revisit after extending the blender matrix or widening event definitions. Until then, rely on deterministic manifest gates + blender.

## 6. Release Packaging
- Assemble `/release/<YYYYMMDD>` containing:
  - Manifests + artifacts (`models/base_xgb_h120_calmon_spread0`, `tcn_h{60,120,180}_calmon_relaxed`, `blender_h120_v6`).
  - Coverage CSVs, RSS audits, shortlist (`models/report_shortlist.json`).
  - Documentation snapshot: updated `TRAINING_STATUS.md`, `TRAINING_WALKTHROUGH.md`, this checklist.
- Tag git commit, push artifacts to remote storage (or repo with Git LFS) per ops policy.

## 7. Monitoring Rollout
- Metrics to expose: equity curve, turnover, gate activation rate, RSS coverage, probability σ.
- Alert thresholds:
  - Gate coverage outside ±2× baseline for two consecutive days.
  - RSS minute spike share <5e-4 or audit `passed=false`.
  - Probability σ <0.03 for any validation window.
- Document remediation runbooks (e.g., rerun relaxed retrain, switch to backup model, disable blender).

## 8. Post-Launch Tasks
- Schedule monthly retrain cadence with automated shortlist generation.
- Expand dataset coverage (additional symbols/timeframes) using the same relaxed gate framework.
- Continue meta-label R&D with the longer matrix and event definitions tuned for Calmon coverage.
