# Subtask 4 – Meta-Label Training Refresh

_Last updated: 2025-10-21 02:50 UTC_

## Run
```
python scripts/train_meta_label.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/meta_h120_v2 \
  --pt-mult 1.5 --sl-mult 2.0 --max-hold 180 \
  --tcn-stride 30 --cost-bps 5 --threshold-grid-min 0.5 --threshold-grid-max 0.9
```

## Result (`models/meta_h120_v2/report.json`)
- `final_equity` **1.00** (no improvement)
- `total_turnover` **0** (meta threshold defaults to 0.5)
- `label_pos_frac` ≈0.54 after barrier generation

## Findings
- Even with the expanded matrix, filtering to rows that satisfy deployable gates and have complete RSS coverage collapses to one class, preventing the logistic meta model from learning a separation.
- The relaxed gate already suppresses most trades; additional meta gating adds no value without a longer validation window or alternative event definition.

## Next Steps
1. Extend the blender matrix to include Oct–Nov 2025 so more co-occurrent signals survive gating.
2. Explore asymmetric barrier settings or longer holding periods to diversify labels.
3. Defer deployment of meta models; rely on manifest gates + blender until a refreshed matrix demonstrates ≥1.2 equity and meaningful turnover after meta filtering.
