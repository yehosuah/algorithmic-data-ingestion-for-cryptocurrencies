# Model Training Status (XGB · TCN · Blender)

_Last updated: 2025-10-23 01:00 UTC_

## Quick Status
- **Base XGB (Calmon relaxed gate)** – `models/base_xgb_h120_calmon_spread0` holds `final_equity 4.48`, Sharpe 108, and 3.6 k toggles under the relaxed training mask (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`). Oct 1 – Oct 21 2025 replay (`models/oos_replay_oct_nov_2025.json`) confirms the training gate’s equity but shows the deployable mask firing zero trades, so gate tuning or calibration fallback is required before production.
- **TCN suite (Calmon relaxed)** – refreshed horizons clear 5 bps costs:  
  `tcn_h60` → `final_equity 1.05`, 67 toggles; `tcn_h120` → `final_equity 1.33`, 180 toggles; `tcn_h180` → `final_equity 1.19`, 48 toggles. Forward replay mirrors the base outcome: profitable under the relaxed gate, idle under the deployable mask.
- **Logistic blender (elastic-net)** – `models/blender_h120_v6` trains on the RSS-enriched matrices and delivers `final_equity 1.84`, Sharpe 28.7, and 711 toggles at threshold 0.95. The forward matrix `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` plus `oos_replay` highlight RSS audit health but likewise record zero inference coverage.
- **CI guardrails** – `.github/workflows/ci.yml` provisions Python 3.11 + CPU torch, running `tests/ingestion_service`, `tests/regression` (manifest gating + shortlist), and `tests/training` on every push/PR.

## Data Landscape
- **Year-wide minute feed** – `datasets/market_btcusdt_1m_2024_2025.parquet`, 924 562 bars from 2024-01-01 ➜ 2025-10-22 01:20 UTC. Spread stats still align with the relaxed training gate (avg `hl_spread` ≈6.8 bps, q99 ≈34 bps).
- **Blender matrix (RSS enriched)** – `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`, 542 760 rows (2024-09-13 ➜ 2025-09-11). `base_prob_mean` 0.515, `tcn_prob_mean` 0.585 over the gated subset, and RSS spikes cover 0.025 % of minutes with 82.5 % daily coverage (see `...rss_latest_stats.json`).
- **Forward replay matrix** – `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`, 30 201 rows (2025-10-01 ➜ 2025-10-21) with base/TCN/blender probabilities embedded for OOS diagnostics.
- **Gate coverage replay** – `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv` confirms the deployable mask historically fires in the 0.0004 %–0.0179 % band, but Oct 2025 replay shows zero fills under the current mask (see `models/oos_replay_oct_nov_2025.json`).

## Horizon-120 XGBoost
- CLI defaults in `scripts/train_base_gbdt.py` now align with the calmon relaxed profile (`--max-spread-z 0.25`, `--max-rvol20 2e-4`, no probability filter) while persisting the deployable gate inside the manifest. Diagnostic output captures calendar-month metrics, RSS audits, and threshold grids.
- Spread stress tests (`spread_scale` ∈ {0, 0.05, 0.1, 0.2}) retain 4.48 equity with identical turnover, showing robustness to 20 % cost inflation.
- `training/reporting.ensure_kpi_schema` normalises KPI payloads, and `scripts/report_shortlist.py` curates `models/report_shortlist.json` so reviewers can spot deployable variants without manual diffing.
- `tests/regression/test_manifest_gating.py` now runs in CI to guarantee each manifest’s gate config and threshold stay in sync with `report.json`; `tests/regression/test_report_shortlist.py` ensures the shortlist CLI continues to surface the Calmon baseline.
- Oct 1 – Oct 21 2025 replay retains 4.48 equity under the relaxed gate but returns `final_equity 1.0` under the deployable mask (zero toggles). Gate thresholds need widening or dynamic fallback before live rollout.

## TCN Suite
- `scripts/train_tcn.py` refresh adds fold logit persistence, stride control, and the relaxed gate defaults. Each manifest references `fold_logits.parquet`, the monthly probability σ table, and the shared inference gate.
- **Key metrics** (`cost_bps=5`, relaxed training gate):
  - `models/tcn_h60_calmon_relaxed`: final_equity **1.054**, Sharpe 16.6, total_turnover 67, selected_threshold 0.55.
  - `models/tcn_h120_calmon_relaxed`: final_equity **1.331**, Sharpe 24.9, total_turnover 180, selected_threshold 0.65.
  - `models/tcn_h180_calmon_relaxed`: final_equity **1.190**, Sharpe 29.6, total_turnover 48, selected_threshold 0.575.
- `models/oos_replay_summary.json` and `models/tcn_gate_replay_summary.json` document training-vs-inference gate behaviour for audit trails. The Oct 2025 forward replay mirrors the base finding: relaxed gates stay profitable, while the deployable mask delivered zero toggles, so inference thresholds must be revisited.

## Blender & Meta Label
- `scripts/build_blender_matrix.py` now builds the RSS-enriched matrix with intraday spike features, exponential decays, probability momentum, and ensures label continuity. Outputs include a JSON summary and leverage `settings.FSSPEC_STORAGE_OPTIONS` for remote stores; the forward replay variant writes `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`.
- `scripts/train_blender.py` consumes the matrix, standardises features, and runs an elastic-net sweep (`l1_ratio` grid). `models/blender_h120_v6` demonstrates that once RSS spikes are engineered into continuous signals the logistic stack can operate with 711 toggles while keeping the 5 bps cost budget.
- Forward diagnostics combine the Oct 2025 matrix with `models/oos_replay_oct_nov_2025.json`; the relaxed gate keeps equity >1 while the deployable inference mask produced zero coverage, matching the base/TCN outcome.
- `scripts/train_meta_label.py` benefits from the shared relaxed gate yet still lacks a stable decision surface—the current meta artifacts remain placeholders until the blender/base probabilities regain dynamic range on forward months.

## Operational Follow-Ups
1. Retune inference gate thresholds or add a fallback so Oct–Nov 2025 maintains non-zero coverage while respecting turnover budgets (current deployable mask idles).
2. Integrate the manifest gates (spread/rvol/prob/min-hold) into the live scoring path and add regression tests that replay `training/infer.py` outputs against `report.json` KPIs.
3. Monitor RSS coverage using the audit block; auto-fallback to a no-RSS feature set if `minute_spike_share < 5e-4` to prevent the blender from overfitting sparse spikes.
4. Revisit meta-label training only after a longer blended matrix is available; otherwise focus on hardening the base + TCN ensemble now that both exceed post-cost profitability.
