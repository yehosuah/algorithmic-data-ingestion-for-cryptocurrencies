# Model Training Status (XGB · TCN · Blender)

_Last updated: 2025-10-21 02:50 UTC_

## Quick Status
- **Base XGB (Calmon relaxed gate)** – `models/base_xgb_h120_calmon_spread0` holds `final_equity 4.48`, Sharpe 108, and 3.6 k toggles under the relaxed training mask (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`). The manifest persists the deployable inference gate (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold = 10`) and monthly coverage, keeping live activation inside the <0.02 % envelope.
- **TCN suite (Calmon relaxed)** – refreshed horizons clear 5 bps costs:  
  `tcn_h60` → `final_equity 1.05`, 67 toggles; `tcn_h120` → `final_equity 1.33`, 180 toggles; `tcn_h180` → `final_equity 1.19`, 48 toggles. All share the relaxed training mask and the inference gate mirrored from the base model.
- **Logistic blender (elastic-net)** – `models/blender_h120_v6` trains on the new RSS-enriched matrix and delivers `final_equity 1.84`, Sharpe 28.7, and 711 toggles at threshold 0.95. Reports now include RSS coverage audits and feature inventories so ops can verify signal health.

## Data Landscape
- **Year-wide minute feed** – `datasets/market_btcusdt_1m_2024_2025.parquet`, 894 240 bars from 2024-01-01 ➜ 2025-09-12. Spread stats align with the relaxed training gate (avg `hl_spread` 6.8 bps, q99 34 bps).
- **Blender matrix (RSS enriched)** – `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`, 542 760 rows (2024-09-13 ➜ 2025-09-11). `base_prob_mean` 0.515, `tcn_prob_mean` 0.585 over the gated subset, and RSS spikes cover 0.025 % of minutes with 82.5 % daily coverage (see `...rss_latest_stats.json`).
- **Gate coverage replay** – `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv` confirms the deployable mask fires in the 0.0004 %–0.0179 % band per month (≤1.63× the historical baseline), satisfying turnover constraints for live deployment.

## Horizon-120 XGBoost
- CLI defaults in `scripts/train_base_gbdt.py` now align with the calmon relaxed profile (`--max-spread-z 0.25`, `--max-rvol20 2e-4`, no probability filter) while persisting the deployable gate inside the manifest. Diagnostic output captures calendar-month metrics, RSS audits, and threshold grids.
- Spread stress tests (`spread_scale` ∈ {0, 0.05, 0.1, 0.2}) retain 4.48 equity with identical turnover, showing robustness to 20 % cost inflation.
- `training/reporting.ensure_kpi_schema` normalises KPI payloads, and `scripts/report_shortlist.py` curates `models/report_shortlist.json` so reviewers can spot deployable variants without manual diffing.

## TCN Suite
- `scripts/train_tcn.py` refresh adds fold logit persistence, stride control, and the relaxed gate defaults. Each manifest references `fold_logits.parquet`, the monthly probability σ table, and the shared inference gate.
- **Key metrics** (`cost_bps=5`, relaxed training gate):
  - `models/tcn_h60_calmon_relaxed`: final_equity **1.054**, Sharpe 16.6, total_turnover 67, selected_threshold 0.55.
  - `models/tcn_h120_calmon_relaxed`: final_equity **1.331**, Sharpe 24.9, total_turnover 180, selected_threshold 0.65.
  - `models/tcn_h180_calmon_relaxed`: final_equity **1.190**, Sharpe 29.6, total_turnover 48, selected_threshold 0.575.
- `models/oos_replay_summary.json` and `models/tcn_gate_replay_summary.json` document training-vs-inference gate behaviour for audit trails; live gates remain intentionally sparse (coverage ≤0.0008 %).

## Blender & Meta Label
- `scripts/build_blender_matrix.py` now builds the RSS-enriched matrix with intraday spike features, exponential decays, probability momentum, and ensures label continuity. Outputs include a JSON summary and leverage `settings.FSSPEC_STORAGE_OPTIONS` for remote stores.
- `scripts/train_blender.py` consumes the matrix, standardises features, and runs an elastic-net sweep (`l1_ratio` grid). `models/blender_h120_v6` demonstrates that once RSS spikes are engineered into continuous signals the logistic stack can operate with 711 toggles while keeping the 5 bps cost budget.
- `scripts/train_meta_label.py` benefits from the shared relaxed gate yet still lacks a stable decision surface—the current meta artifacts remain placeholders until the blender/base probabilities regain dynamic range on forward months.

## Operational Follow-Ups
1. Extend validation to Oct–Nov 2025 to confirm the relaxed training gate and the inference mask maintain equity >1.2 across regimes.
2. Integrate the manifest gates (spread/rvol/prob/min-hold) into live scoring paths and add regression tests that replay `report.json` outputs through the inference adapters.
3. Monitor RSS coverage using the audit block; auto-fallback to a no-RSS feature set if `minute_spike_share < 5e-4` to prevent the blender from overfitting sparse spikes.
4. Revisit meta-label training only after a longer blended matrix is available; otherwise focus on hardening the base + TCN ensemble now that both exceed post-cost profitability.
