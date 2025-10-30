# Subtask 4 – Meta-Label Training Refresh

_Last updated: 2025-10-30 16:05 UTC_

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
- Even with the forward replay matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`), filtering to rows that satisfy deployable gates still collapses to a single dominant class—base now contributes only 12 gate hits (8 trades) and TCN/blender probabilities offer limited overlap despite the blender manifest firing 5 870 trades.
- Blender stride experiments (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) indicate that shrinking the smoothing window can flood coverage (>50 %) but pushes turnover beyond deployable limits, so the meta layer still sees imbalanced labels.
- The relaxed gate already suppresses most trades; additional meta gating adds no value until deployable thresholds are retuned to yield meaningful overlap.

## Next Steps
1. Wait for deployable gate retuning (especially on the TCN suite) to restore meaningful overlap on forward windows before attempting further meta training; the current stride‑1 sandbox runs are for analysis only.
2. Once coverage stabilises, explore asymmetric barrier settings or longer holding periods to diversify labels.
3. Defer deployment of meta models; rely on manifest gates + blender until a refreshed matrix demonstrates ≥1.2 equity and meaningful turnover after meta filtering.
