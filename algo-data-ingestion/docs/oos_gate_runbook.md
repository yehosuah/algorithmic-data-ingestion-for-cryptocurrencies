# OOS Matrix + Gate Coverage Runbook

_Last updated: 2025-11-10 04:13 UTC_

> Update 2025-11-10: Runbook links the release/20251030 calibration outputs with the scheduler/trading monitoring hooks (Redis queue alerts, Prometheus gauges) introduced in this branch.

## 1. Rebuild the stride-2 OOS blender matrix

```bash
python3 scripts/build_blender_matrix.py \
  --source datasets/market_btcusdt_1m_2024_2025.parquet \
  --out datasets/blender_matrix_2025-09_to_2025-11_oos.parquet \
  --base-dir models/base_xgb_h120_calmon_spread005 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --start-date 2025-09-01 \
  --end-date 2025-10-29T23:58:00Z \
  --tcn-stride 2
```

- This replay produces ~84 k rows with **42 k** populated TCN probabilities (stride‑2, window 192) so the validation slice has ≥14 k rows.
- Inspect the JSON footer the script prints for quick stats (`rows`, `tcn_prob_mean`, etc).

## 2. Refresh calibration

```bash
python3 scripts/refresh_calibration.py \
  --data datasets/blender_matrix_2025-09_to_2025-11_oos.parquet \
  --base-model models/base_xgb_h120_calmon_spread005 \
  --tcn-model models/tcn_h120_calmon_relaxed \
  --blender-model models/blender_h120_v6 \
  --split-ratio 0.65 \
  --out-dir release/calibration/2025-11-oos_stride2
```

- Outputs land in `release/calibration/2025-11-oos_stride2/` (per-model metrics JSON, reliability/histogram PNGs, and `calibration_summary.json`).
- The TCN refresh now selects `isotonic_blend` at weight 0.6 (Brier 5.75e‑2 → 5.74e‑2, ECE 1.65e‑2 → 1.25e‑2) thanks to the stride‑2 coverage.
- `save_calibrator` writes `models/<manifest>/calibration/<prob>.json` + joblib; `load_tcn_predictor` automatically attaches the new post-hoc mapping.

## 3. Update manifest thresholds

1. Read the recommended gate threshold from `release/calibration/2025-11-oos_stride2/calibration_summary.json` (blender section).
2. Update `models/blender_h120_v6/threshold.txt` and `manifest.json` (`gates.training.prob_gate_min`, `gates.inference.prob_gate_min`) to the selected value (currently 0.55).
3. Copy the refreshed manifest bundle to `MODELS_ROOT` so the scheduler/trading services reload it.

## 4. Monitoring blender gate coverage

- **Scheduler logs:** `_record_gate_coverage` now prints `coverage`, `passed`, `total`, and the active threshold per job.
- **Prometheus gauges:** `model_gate_coverage_ratio` exposes three key modes:
  - `mode="inference"` – per batch coverage.
  - `mode="inference_rolling24h"` – rolling 24 h coverage computed from raw gate counts.
  - `mode="oos_reference"`, `oos_reference_lower`, `oos_reference_upper` – the 19 % ±3 pp band set from `models/blender_h120_v6/manifest.json` metadata.
- **Alert:** `monitoring/alert.rules.yml` defines `BlenderGateCoverageOutOfBand` which fires when the rolling 24 h coverage leaves `[0.16, 0.22]` for 30 minutes.
- **Fatigue checks:** correlate `model_gate_coverage_ratio{model="blender_h120_v6",mode="inference_rolling24h"}` with trading counters (`trading_trade_attempts_total`, `trading_gate_toggles_total`) and the scheduler decision logs to catch over/under-trading.

## 5. Responding to coverage drift / fatigued trades

1. **Detect:** Alert fires or manual inspection shows either coverage outside `[0.16, 0.22]` or qualitative fatigue (noise trading, PnL deterioration).
2. **Rebuild:** Re-run the stride‑2 OOS matrix and `scripts/refresh_calibration.py` (sections 1–2).
3. **Review:** Inspect the blender section in `release/calibration/<new_stamp>/calibration_summary.json` for the recommended gate threshold and coverage deltas.
4. **Deploy:** Update `threshold.txt`/manifest gates, sync to `MODELS_ROOT`, and restart scheduler + trading.
5. **Verify:** Watch `model_gate_coverage_ratio{model="blender_h120_v6",mode="inference_rolling24h"}` converge back to the new `oos_reference`, confirm alerts clear, and sanity-check trade logs for quality improvements.

Keep this loop documented in Git (calibration artifact folder + manifest diff) so future refreshes can be audited.
