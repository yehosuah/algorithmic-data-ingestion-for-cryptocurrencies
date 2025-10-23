# Subtask 3 – Elastic-Net Blender Refresh

_Last updated: 2025-10-23 01:00 UTC_

## Goal
Train an elastic-net logistic blender that combines Calmon relaxed base and TCN probabilities with RSS spike features, clearing 5 bps transaction costs while maintaining actionable turnover.

## Workflow
1. Build the RSS-enriched blender matrix using `scripts/build_blender_matrix.py` (intraday RSS spikes, probability momentum, relaxed gate masks) and capture forward windows (e.g., Oct 2025) into `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`.
2. Run `scripts/train_blender.py` with elastic-net sweep and turnover guards.
3. Review RSS audit (`passed=true`) and feature inventory before promoting the artifact.

## Command
```bash
python scripts/train_blender.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/blender_h120_v6 \
  --cost-bps 5 --tcn-stride 30 \
  --max-total-turnover 10000 --min-toggle-count 2 \
  --l1-ratio-grid 0.15 0.35 0.55 0.75 0.9
```

## Result (`models/blender_h120_v6/report.json`)
- `final_equity`: **1.837**
- `sharpe`: 28.7
- `total_turnover`: 711
- `selected_threshold`: 0.95
- RSS audit: daily coverage 0.825, minute spike share 0.254 (pass, indicator `rss_spike_presence`)
- RSS gate applied: `rss_spike_decay_fast ≥ 0.08` (share ≈0.021)

## Notes
- Feature set includes probability momentum (`prob_diff`, `*_mom_1`), RSS spike windows, and volatility deltas. The manifest lists candidate features for transparency.
- Training pipeline now leverages KPI schema normalization (`training/reporting.ensure_kpi_schema`) so reports align with base/TCN outputs.
- The matrix stats JSON (`..._rss_latest_stats.json`) provides sanity checks on probability distributions and RSS coverage before fitting.
- Oct 2025 replay (`models/oos_replay_oct_nov_2025.json`) showed relaxed gate equity >1 but zero deployable trades; align blender gating with the retuned base/TCN thresholds.

## Next Steps
1. Retune blender gating (or stage a fallback) so Oct–Nov 2025 delivers equity ≥1.2 with turnover within ±25 % of the 711-toggle baseline.
2. Integrate RSS audit thresholds into monitoring; automatically switch to a no-RSS fallback when coverage dips below requirements.
3. Document feature preprocessing in inference code to mirror StandardScaler + selected columns from the manifest.
