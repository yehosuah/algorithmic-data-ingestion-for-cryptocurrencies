# Subtask 4 – Deployability Check

_Last updated: 2025-10-29 15:53 UTC_

## Models In Scope
- **Base XGB (H120, Calmon relaxed)** – `models/base_xgb_h120_calmon_spread0`
  - `final_equity 4.48`, Sharpe 108.3, relaxed gate coverage 9.4 %.
  - Deployable gate (`manifest.gates.inference`): `hl_spread ≤ 0.0007`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`.
  - Oct 2025 replay (`models/oos_replay_summary_latest.json`) now records 12 deployable gate hits (8 trades, `final_equity 1.23`); continue tracking coverage while thresholds settle.
- **TCN suite (Calmon relaxed)** – `models/tcn_h{60,120,180}_calmon_relaxed`
  - Horizons 60/120/180 deliver `final_equity` 1.05 / 1.33 / 1.19 with ≤200 toggles. Shared inference gate mirrors the base mask.
  - Oct 2025 replay (`models/oos_replay_summary_latest.json`) still shows zero deployable toggles—decide between widening the manifest further or documenting a TCN fallback while base/blender carry coverage.
- **Blender (H120 elastic-net)** – `models/blender_h120_v6`
  - `final_equity 1.84`, Sharpe 28.7, 711 toggles at threshold 0.95, RSS audit passed (daily coverage 82.5 %, minute spike share 0.254).
  - Forward replay using `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` now produces ≈16 % deployable coverage (5 870 toggles) under the eased manifest (`prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`); keep it aligned with the base thresholds.
- **Meta label** – `models/meta_h120_v2` (research only)
  - Equity remains ≈1.0; defer deployment until extended matrix and new barrier configuration restore separation.

## Data Artifacts
- `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (plus stats JSON) – Year-wide matrix with intraday RSS spikes, probability momentum, and relaxed gate masks.
- `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` – Oct 1–Oct 21 2025 window with embedded probabilities for forward gate audits.
- `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv` – Monthly coverage baseline for deployable mask.
- `models/oos_replay_summary_latest.json`, `models/tcn_gate_replay_summary.json` – Training vs inference gate diagnostics for base/TCN/blender (latest thresholds).
- `models/oos_replay_oct_nov_2025.json` – Archived zero-coverage replay kept for regression comparisons.
- `models/report_shortlist.json` – KPI-normalized shortlist compiled via `scripts/report_shortlist.py`.
- `.github/workflows/ci.yml` – GitHub Actions pipeline running ingestion E2E tests and manifest/shortlist regressions.

## Deployment Readiness Checklist
1. **Forward Validation** – Replay Oct–Nov 2025 (and beyond) for base, TCN, blender using the deployable gate; keep base/blender at equity ≥1.2 with turnover deviations <±25 % of baseline and either widen the TCN mask or document a fallback so it no longer idles.
2. **Inference Parity** – Integrate manifest gates + feature lists into the live adapter, add regression tests that compare `training/infer.py` outputs against `report.json` KPIs.
3. **Monitoring Hooks** – Instrument equity, turnover, gate activation rate, RSS coverage, and probability σ. Alert when:
   - Gate coverage leaves ±2× band for two consecutive days
   - RSS minute spike share <5e-4 or audit `passed=false`
   - Probability σ <0.03 on any validation month
4. **Fallback Plan** – Document no-RSS blender configuration and base-only mode in case RSS feeds fail the audit; include the deployable-gate fallback that restores coverage when Oct 2025-like droughts recur.

## Outstanding Work
- Automate packaging of manifests/artifacts into a release bundle (see `docs/final_stretch_v1.md`).
- Retune deployable gates (base/TCN/blender) so Oct 2025 forward replay shows non-zero coverage; update manifests once thresholds stabilise.
- Extend the blender matrix to new months before reattempting meta-label deployment.
- Evaluate storage strategy (Git LFS or artifact bucket) for large parquet/model files prior to production handoff.
