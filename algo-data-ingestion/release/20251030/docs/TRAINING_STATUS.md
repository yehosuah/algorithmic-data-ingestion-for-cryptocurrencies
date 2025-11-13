# Model Training Status (XGB · TCN · Blender)

_Last updated: 2025-11-13 04:43 UTC (archival snapshot refreshed)_

> Update 2025-11-13: Recorded that the sanitized multi-symbol feed + symbol-gate payload and the parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) are documented in the active branch; this archive stays frozen at release/20251030 for historical comparison.

## Quick Status
- _Update 2025-11-05_: The active branch now includes scheduler-driven inference and a trading dry run (`app/trading/service.py`). Refer to the root `TRAINING_STATUS.md` for the latest metrics covering Redis decision queues and Prometheus trading counters.
- **Base XGB (Calmon relaxed gate)** – `models/base_xgb_h120_calmon_spread0` retains `final_equity 4.48`, Sharpe 108, 3.6 k toggles under the relaxed mask, and the manifest now ships a widened deployable profile (`hl_spread ≤ 7e-4`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`). The refreshed Oct 1 – Oct 27 2025 replay (`models/oos_replay_summary_latest.json`) records 12 gate hits, 8 toggles, and `final_equity 1.23`, restoring minimal live coverage.
- **TCN suite (Calmon relaxed)** – horizons 60/120/180 still clear 5 bps costs (`final_equity` 1.05/1.33/1.19 with ≤200 toggles); the deployable manifest mirrors the base thresholds and remains idle in the latest replay (zero toggles, gate coverage <0.001 %), but `training/infer.predict_tcn` now batches by stride so stride-1 experiments no longer exhaust memory.
- **Logistic blender (elastic-net)** – `models/blender_h120_v6` continues to post `final_equity 1.84`, Sharpe 28.7, 711 toggles at threshold 0.95; the run now records `gate_smoothing_stride` (defaults to the TCN stride) and fresh stride‑1 prototypes (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) highlight how collapsing the smoothing window drives gate share into the 52–70 % band while keeping relaxed equity at 4.48.
- **CI guardrails & monitoring** – `.github/workflows/ci.yml` runs ingestion, regression, and training suites with manifest tests split out; `app/monitoring/model_metrics.py` surfaces gate coverage, RSS share, and probability σ gauges so Prometheus alerts trigger when deployable behaviour drifts.

## Data Landscape
- **Year-wide minute feed** – `datasets/market_btcusdt_1m_2024_2025.parquet`, 959 039 bars from 2024-01-01 ➜ 2025-10-27 23:58 UTC. Spread stats still align with the relaxed training gate (avg `hl_spread` ≈6.8 bps, q99 ≈34 bps).
- **Blender matrix (RSS enriched)** – `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (mirrored as `..._2025-10_rss_latest.parquet`), 606 121 rows covering 2024-09-01 ➜ 2025-10-26; the refreshed stats file now records window start/end and probability means (`base_prob_mean 0.515`, `tcn_prob_mean 0.509`).
- **Forward replay matrix** – `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`, 28 681 rows (2025-10-01 ➜ 2025-10-20 22:00) with base/TCN/blender probabilities embedded for OOS diagnostics.
- **Gate coverage replay** – `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv` keeps historical deployable coverage in the 0.0004 %–0.0179 % band; `models/oos_replay_summary_latest.json` shows the widened manifest producing 12 gate hits (8 trades) in Oct 2025, while the TCN manifests remain idle.

## Horizon-120 XGBoost
- CLI defaults in `scripts/train_base_gbdt.py` now align with the calmon relaxed profile (`--max-spread-z 0.25`, `--max-rvol20 2e-4`, no probability filter) while persisting the deployable gate inside the manifest. Diagnostic output captures calendar-month metrics, RSS audits, and threshold grids.
- Spread stress tests (`spread_scale` ∈ {0, 0.05, 0.1, 0.2}) retain 4.48 equity with identical turnover, showing robustness to 20 % cost inflation.
- `training/reporting.ensure_kpi_schema` normalises KPI payloads, and `scripts/report_shortlist.py` curates `models/report_shortlist.json` so reviewers can spot deployable variants without manual diffing.
- `tests/regression/test_manifest_gating.py` now runs in CI to guarantee each manifest’s gate config and threshold stay in sync with `report.json`; `tests/regression/test_report_shortlist.py` ensures the shortlist CLI continues to surface the Calmon baseline.
- Oct 1 – Oct 27 2025 replay (`models/oos_replay_summary_latest.json`) retains 4.48 equity under the relaxed gate and now reports 12 deployable gate hits (8 toggles, `final_equity 1.23`) after widening the manifest—monitor coverage closely while iterating on additional guardrails.

## TCN Suite
- `scripts/train_tcn.py` refresh adds fold logit persistence, stride control, and the relaxed gate defaults. Each manifest references `fold_logits.parquet`, the monthly probability σ table, and the shared inference gate.
- **Key metrics** (`cost_bps=5`, relaxed training gate):
  - `models/tcn_h60_calmon_relaxed`: final_equity **1.054**, Sharpe 16.6, total_turnover 67, selected_threshold 0.55.
  - `models/tcn_h120_calmon_relaxed`: final_equity **1.331**, Sharpe 24.9, total_turnover 180, selected_threshold 0.65.
  - `models/tcn_h180_calmon_relaxed`: final_equity **1.190**, Sharpe 29.6, total_turnover 48, selected_threshold 0.575.
- `models/oos_replay_summary_latest.json` and `models/tcn_gate_replay_summary.json` document training-vs-inference gate behaviour for audit trails. The Oct 2025 forward replay mirrors the base finding in reverse: relaxed gates stay profitable, but the deployable masks still deliver zero toggles for all TCN horizons, so thresholds or fallback logic must be revisited.

## Blender & Meta Label
- `scripts/build_blender_matrix.py` now builds the RSS-enriched matrix with intraday spike features, exponential decays, probability momentum, and ensures label continuity. Outputs include a JSON summary and leverage `settings.FSSPEC_STORAGE_OPTIONS` for remote stores; the forward replay variant writes `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`.
- `scripts/train_blender.py` consumes the matrix, standardises features, and runs an elastic-net sweep (`l1_ratio` grid). `models/blender_h120_v6` demonstrates that once RSS spikes are engineered into continuous signals the logistic stack can operate with 711 toggles while keeping the 5 bps cost budget, and it now smooths gate masks over the detected stride (from `--tcn-stride` or inferred timestamps) while persisting that window as `gate_smoothing_stride` in `report.json`.
- New sandboxed runs (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) explore stride‑1 gating to quantify the turnover trade-off when smoothing is removed; gate share surges above 50 %, giving us bounds for production manifests.
- Forward diagnostics combine the Oct 2025 matrix with `models/oos_replay_summary_latest.json`; the relaxed gate keeps equity >1 while the retuned deployable manifest now fires 5 870 toggles (`gate_coverage ≈ 16 %`), validating the eased thresholds ahead of broader monitoring rollout.
- `scripts/train_meta_label.py` benefits from the shared relaxed gate yet still lacks a stable decision surface—the current meta artifacts remain placeholders until the blender/base probabilities regain dynamic range on forward months.

## Operational Follow-Ups
1. Extend the deployable retune to the TCN suite (or document a fallback) while tracking `model_gate_coverage_ratio` for both the base manifest and the smoothed blender gate so coverage stays above the new floor.
2. Thread `training/infer.py`’s `load_manifest_artifacts`/`score_base_with_manifest` helpers through the production API, ensuring Prometheus gauges expose gate coverage, RSS share, and probability σ for every inference batch and that stride-aware batching is exercised end-to-end.
3. Monitor RSS coverage via the new gauges + alerts, keep the no-RSS fallback path ready when `model_rss_minute_spike_share` drops below the manifest threshold, and document when lowering the smoothing stride is acceptable to boost coverage.
4. Revisit meta-label training only after deployable coverage stabilises across base/TCN/blender (including the stride experiments); until then prioritise hardening the ensemble and regression checks.
