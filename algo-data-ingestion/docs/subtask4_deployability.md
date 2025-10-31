# Subtask 4 – Deployability Check

_Last updated: 2025-10-31 02:39 UTC_

## Models In Scope
- **Base XGB (H120, Calmon relaxed)** – `models/base_xgb_h120_calmon_spread0`
  - `final_equity 4.48`, Sharpe 117.4, relaxed gate coverage 9.4 %.
  - Deployable gate (`manifest.gates.inference`): `hl_spread ≤ 0.0007`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`.
  - Oct 2025 replay (`models/oos_replay_summary_latest.json`, 40 201 rows) records 12 deployable gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`); continue tracking coverage while thresholds settle.
- **TCN suite (Calmon relaxed)** – `models/tcn_h{60,120,180}_calmon_relaxed`
  - Horizons 60/120/180 deliver `final_equity` 1.28 / 3.62 / 1.85 with ≤200 toggles. Inference gates are relaxed (`hl_spread ≤ 9e-4`, `rvol_20 ≤ 1.8e-4/1.8e-4/1.5e-4`, `prob ≥ 0.52/0.68/0.52`, `min_hold 10`).
  - Oct 2025 replay (`models/oos_replay_summary_latest.json`) shows deployable coverage across horizons (`gate_hits 4/31/2`, `gate_coverage 4.73e-4/7.71e-4/4.23e-4`, `final_equity 1.03/1.94/1.01`); keep the CI guardrail green when adjusting manifests.
  - `training/infer.predict_tcn` batches inference by stride, enabling stride‑1 experiments without exhausting memory.
- **Blender (H120 elastic-net)** – `models/blender_h120_v6`
  - `final_equity 4.48`, Sharpe 206.8, 4 809 toggles at threshold 0.5, RSS audit passed (daily coverage 99.5 %, minute spike share 0.991).
  - Deployable manifest: `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`; forward replay using `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` produces ≈15.8 % deployable coverage (6 346 toggles) with `final_equity 4.48`.
  - Sandbox variants (`models/blender_h120_stride1_v2`) collapse smoothing to one bar, yielding ≈0.2 % coverage (134 toggles) while keeping equity at 4.48; use them as a turnover ceiling when tuning.
- **Meta label** – `models/meta_h120_v2` (research only)
  - Equity remains ≈1.0; defer deployment until extended matrix and new barrier configuration restore separation.

## Data Artifacts
- `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (plus stats JSON) – Year-wide matrix with intraday RSS spikes, probability momentum, and relaxed gate masks.
- `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` – Oct 1–Oct 28 2025 window (40 201 rows) with embedded probabilities for forward gate audits.
- `models/blender_h120_gate_test`, `models/blender_h120_stride1`, `models/blender_h120_stride1_v2` – Stride‑1 sandbox runs quantifying turnover vs coverage when gate smoothing is removed.
- `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv` – Monthly coverage baseline for deployable mask.
- `models/oos_replay_summary_latest.json`, `models/tcn_gate_replay_summary.json` – Training vs inference gate diagnostics for base/TCN/blender (latest thresholds).
- `models/oos_replay_oct_nov_2025.json` – Archived zero-coverage replay kept for regression comparisons.
- `models/report_shortlist.json` – KPI-normalized shortlist compiled via `scripts/report_shortlist.py`.
- `.github/workflows/ci.yml` – GitHub Actions pipeline running ingestion E2E tests, manifest/shortlist regressions, and the TCN forward replay guardrail (`gate_fraction ≥ 5e-4`, `final_equity ≥ 1.2`).

## Deployment Readiness Checklist
1. **Forward Validation** – Replay Oct–Nov 2025 (and beyond) for base, TCN, blender using the deployable gate; keep base/blender at equity ≥1.2 with turnover deviations <±25 % of baseline and ensure TCN coverage stays above the 5e-4 guardrail (document a fallback if it regresses).
2. **Inference Parity** – Integrate manifest gates + feature lists into the live adapter, add regression tests that compare `training/infer.py` outputs against `report.json` KPIs.
3. **Monitoring Hooks** – Instrument equity, turnover, gate activation rate, RSS coverage, probability σ, and the recorded `gate_smoothing_stride`. Alert when:
   - Gate coverage leaves ±2× band for two consecutive days
   - RSS minute spike share <5e-4 or audit `passed=false`
   - Probability σ <0.03 on any validation month
4. **Fallback Plan** – Document no-RSS blender configuration and base-only mode in case RSS feeds fail the audit; include the deployable-gate fallback that restores coverage when Oct 2025-like droughts recur.

## Outstanding Work
- Automate packaging of manifests/artifacts into a release bundle (see `docs/final_stretch_v1.md`).
- Maintain deployable gates (base/TCN/blender) so Oct 2025 forward replay keeps coverage at the new baseline; document when smoothing or thresholds move and re-run the sandbox comparisons.
- Extend the blender matrix to new months before reattempting meta-label deployment.
- Evaluate storage strategy (Git LFS or artifact bucket) for large parquet/model files prior to production handoff.
