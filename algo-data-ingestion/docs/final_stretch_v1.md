# Final Stretch – Production Checklist (Calmon Stack)

Last updated: 2025-10-23 01:00 UTC

Scope: Align the relaxed-gate Horizon-120 XGB, Calmon TCN suite, and elastic-net blender for a deployable release, with manifest-driven governance and monitoring.

## 1. Pre-Flight
- Freeze `datasets/market_btcusdt_1m_2024_2025.parquet`, `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`, and the forward replay matrix `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (record commit hash + SHA in release notes).
- Persist monthly gate coverage snapshots for base and TCN (`live_gate_coverage.csv`), forward replay diagnostics (`models/oos_replay_oct_nov_2025.json`), and attach to the release package.
- Ensure manifests include deployable gates, thresholds, feature lists, and RSS audits; treat them as the contract between training and inference.
- Keep `.github/workflows/ci.yml` green so manifest gating and shortlist regressions (`tests/regression`) stay enforced ahead of release tagging.

## 2. Base XGBoost (H120 Calmon)
- **Status**: `final_equity 4.48`, Sharpe 108, relaxed gate coverage 9.4 %. Oct 1–Oct 21 2025 replay retained equity under the relaxed gate but the deployable mask (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold = 10`) fired zero trades.
- **Actions**
  1. Retune or stage a fallback for the inference mask so Oct 2025 delivers minimal but non-zero coverage before launch.
  2. Add regression tests that run `training/infer.py` with manifest gates and compare against `report.json` KPIs (equity, turnover, threshold).
  3. Export coverage alert configs (±2× baseline) for production monitoring.

## 3. TCN Suite (H60/H120/H180 Calmon)
- **Status**: All horizons clear 5 bps costs (`final_equity` 1.05–1.33) with ≤200 toggles and matching inference gates, yet Oct 2025 replay likewise recorded zero deployable trades.
- **Actions**
  1. Retune inference thresholds or staged fallbacks alongside the base model so forward windows maintain coverage without breaching 200-toggle guardrails.
  2. Bundle `fold_logits.parquet`, calibrator, scaler, and manifests for each horizon; document when to switch horizons.
  3. Integrate probability σ guardrail (alert <0.03) into monitoring.

## 4. Elastic-Net Blender (H120)
- **Status**: `final_equity 1.84`, Sharpe 28.7, 711 toggles at threshold 0.95 with RSS spike gate share ≈2 %. Oct 2025 replay (via `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`) output zero deployable trades despite healthy training-gate equity.
- **Actions**
  1. Verify RSS coverage alerts (minute spike share ≥5e-4, daily coverage ≥0.8). Configure automatic fallback to no-RSS feature set.
  2. Document feature weights, probability momentum, and gating logic so live scoring can match the training pipeline.
  3. Align blender gating with the post-retune base/TCN mask and ensure Oct 2025 replay delivers equity ≥1.2 with turnover within ±25 % of the current run before sign-off.

## 5. Meta Label (Deferred)
- Training still collapses due to limited dynamic range. Revisit after extending the blender matrix or widening event definitions. Until then, rely on deterministic manifest gates + blender.

## 6. Release Packaging
- Assemble `/release/<YYYYMMDD>` containing:
  - Manifests + artifacts (`models/base_xgb_h120_calmon_spread0`, `tcn_h{60,120,180}_calmon_relaxed`, `blender_h120_v6`).
  - Coverage CSVs, RSS audits, shortlist (`models/report_shortlist.json`), OOS replay snapshots (`models/oos_replay_oct_nov_2025.json`), and the forward matrix `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`.
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
