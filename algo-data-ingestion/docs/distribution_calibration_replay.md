# Probability Distribution, Calibration, and Replay Forensics

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-16: Documented the hourly/stress outputs from `scripts/probability_distribution_audit.py`, clarified that `refresh_calibration.py` re-scores the raw base booster before fitting calibrators to avoid double-scaling, and tied the replay checklist to the new sampler-fed baselines.

## 4.1 Multi-horizon, multi-model distribution audit
- **Capture pre-gate snapshots**: keep `PROB_SAMPLE_ENABLED=1` so every scheduler/API batch writes `logs/probability_samples/<model>_<prob>.jsonl` and (optionally) pushes to `probability:samples` on Redis. The sampler already tags `timestamp`, `symbol`, `timeframe`, `model_kind`, `stride`, `exchange`, and `job_id`.
- **Persist hourly parquet** (schema-consistent, versioned): run the new audit script against the live JSONL/Redis stream and write enriched hourly partitions with model/prob tags:
  ```bash
  python3 scripts/probability_distribution_audit.py \
    --samples logs/probability_samples \
    --fold-logits models/tcn_h120_calmon_relaxed/fold_logits.parquet \
    --fold-column prob_calibrated \
    --features /tmp/features_debug.parquet \
    --out-parquet release/calibration/latest/live_prob_samples.parquet \
    --hourly-dir release/calibration/latest/live_prob_hourly \
    --summary-out release/calibration/latest/distribution_audit.json
  ```
  - Tags added automatically: `model`, `prob_column`, `timeframe`, `session (Asia/EU/US)`, `symbol_cluster`, `vol_bucket` (`rvol_20` or `rvol20`), `spread_bucket` (`hl_spread`), `volume_decile`, and `regime_label` (`vol_bucket/spread_bucket`). Include `feature_version` by merging `--features` (e.g., an export from `scripts/export_feature_slice.py`).
  - `--baseline-days` (default 3) compares each stratified distribution to the last N days of live samples; `--fold-logits` anchors KS/PSI/Wasserstein against the training-time logits for each prob column.
  - Hourly parquet partitions (under `live_prob_hourly/model=<...>/prob=<...>/*.parquet`) form the rolling baseline store for follow-on jobs or Grafana/ETL.
- **Outputs**:
  - `distribution_audit.json` for each `(model, prob_column, timeframe, session/regime/symbol_cluster/vol_bucket/spread_bucket)` tuple: summary stats, histogram + CDF, KS/PSI/Wasserstein versus `fold_logits`, and versus the rolling live baseline. Collapse/saturation flags are set when ≥60 % of mass sits near 0.5 or the extremes.
  - Optional `live_prob_samples.parquet` with all tags; use it for drift dashboards or downstream Parquet dumps.
  - Run the same command against perturbed slices to emit `distribution_audit_stress.json` so stress-induced collapse/saturation is explicit in the incident packet.

## 4.2 Full recalibration under covariate & label shift
- **Re-fit on the freshest labeled live batch** (e.g., `y_dir` or realized PnL sign merged via `scripts/export_feature_slice.py`):
  ```bash
  python3 scripts/refresh_calibration.py \
    --data /tmp/features_debug.parquet \
    --base-model models/base_xgb_h120_calmon_spread0 \
    --tcn-model models/tcn_h120_calmon_relaxed \
    --blender-model models/blender_h120_v6 \
    --split-ratio 0.65 \
    --out-dir release/calibration/live_recalibration_latest
  ```
  - Run once globally, then per-slice by filtering the parquet (e.g., `session`/`regime_label`/`symbol_cluster`) before invoking the script so calibrators match the live covariates/labels.
  - The script re-scores `base_prob` from the raw booster before fitting so calibrators do not double-scale already clipped outputs; stash both the refreshed artifacts and the plotting summaries it produces.
- **Compute calibration metrics per slice** using the new audit script with `--label-column y_dir` (or your meta-label) to emit ECE/Brier/log-loss/ROC-AUC/PR-AUC plus reliability/histograms:
  ```bash
  python3 scripts/probability_distribution_audit.py \
    --samples release/calibration/latest/live_prob_samples.parquet \
    --fold-logits models/base_xgb_h120_calmon_spread0/fold_logits.parquet \
    --fold-column prob_calibrated \
    --label-column y_dir \
    --baseline-days 7 \
    --summary-out release/calibration/latest/calibration_slices.json
  ```
- **Collapse vs saturation triage**:
  - `collapsed=true` → mass near 0.5 (feature mismatch or regime shift), gate tuning will not help; re-align features/manifests or retrain.
  - `saturated=true` → mass at extremes (label drift or overconfident calibrator); re-run `refresh_calibration.py` on the live slice and recheck ECE/Brier before touching thresholds.
  - When ECE/Brier remain poor but ROC-PR are monotone, treat probabilities as mis-scaled (gate retune OK). When ROC also degrades, treat as signal/feature failure.

## 4.3 Stress-test calibration under adversarial regimes
- **Generate perturbed slices** from replay/backtest data (OHLCV, book, news) and re-run the calibration checks:
  ```bash
  # Example: widen spreads and spike volatility to mimic stress
  python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet")
df["hl_spread"] *= 3.0
df["rvol_20"] *= 2.0
df.to_parquet("/tmp/blender_matrix_stress.parquet", index=False)
PY
  python3 scripts/probability_distribution_audit.py \
    --samples /tmp/blender_matrix_stress.parquet \
    --fold-logits models/blender_h120_v6/fold_logits.parquet \
    --fold-column prob_calibrated \
    --summary-out release/calibration/latest/stress_distribution.json
  ```
- **Define guardrails/SLAs** (per model + prob column):
  - `ECE`/`Brier`/`PSI` ceilings; `KS`/`Wasserstein` bounds by regime/symbol bucket.
  - Minimum coverage at target prob thresholds (use gate coverage from replay + live counters).
  - If breached in live: automatically fall back to the previous manifest, raise gates (higher threshold/lower leverage), and alert risk/engineering.

## 5. Replay vs training vs live alignment (multi-layer causality)
- **Deterministic replay on the exact dry-run window**:
  ```bash
  # Base/TCN/Blender depending on manifest; align gates to inference
  python3 scripts/run_oos_eval.py \
    --family tcn \
    --data datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet \
    --model-dir models/tcn_h120_calmon_relaxed \
    --align-gates \
    --stride 30 \
    --save-oos /tmp/tcn_oos.parquet \
    --save-fold-logits models/tcn_h120_calmon_relaxed/fold_logits.parquet
  ```
  - Pin to the deployed manifest (feature set, gate thresholds, cost model) and confirm the replay outputs match training reports bit-for-bit at decision timestamps.
- **Structured comparison: training → replay → live**:
  1. For each symbol/venue/horizon, compute gate coverage, trade count, turnover, and PnL in replay (`compare_oos_frames` output or the saved `*_oos.parquet`).
  2. Extract live audit stream (Redis/Postgres) for the same window: entries/exits, side, leverage, realized PnL, slippage vs decision price.
  3. Align by `timestamp`, `symbol`, `venue`, `manifest hash`, and infra build version; locate the first divergence (scores vs gates vs orders vs execution).
- **Execution-layer forensic** (when training/replay match but live underperforms):
  - Pull per-order logs (placements/amends/cancels/fills/rejects) and annotate with spread at submission, depth buckets, queue estimates, and post-fill mid-move.
  - Separate issues: rejects (venue limits/precision), stale routing (latency, missing cancel/replace), mis-sized orders (clipping/leverage), or latency asymmetry.
  - Propose fixes (order type, smart routing, size caps, TIF) and quantify expected PnL uplift vs added complexity.
- **If replay fails to match training**:
  - Recheck label integrity (no future leakage, correct session/liquidity/anomaly filters, fees/funding/slippage alignment).
  - Verify bar construction matches training (time/volume/tick), and that symbol migrations/corporate actions aren’t misaligned.
- **Decision framework for remediation**:
  - Broken probabilities/calibration → retrain/feature-engineer per regime; re-fit calibrators; re-evaluate SLAs.
  - Probabilities fine but gates off → threshold optimisation under costs/turnover guardrails.
  - Gates fine but execution lossy → execution fixes or capacity/symbol pruning.
  - Each path must state expected PnL uplift, regime sensitivity, monitoring signals, and kill-switches.
