# Subtask 4 – Meta-Label Training Refresh

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Folded in the sanitized multi-symbol feed + symbol-gate generator and the parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so meta updates align with the gates/metrics enforced by scheduler + trading.

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
- Even with the forward replay matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`, 40 201 rows), filtering to rows that satisfy deployable gates still collapses to a single dominant class—base contributes only 12 gate hits (8 trades), TCN horizons add sparse coverage (`gate_hits 4/31/2`), and the blender manifest dominates with 6 346 trades.
- Blender stride experiments (`models/blender_h120_stride1_v2`) indicate that shrinking the smoothing window can reduce turnover to 134 toggles while maintaining equity, yet the meta layer still sees imbalanced labels.
- The relaxed gate already suppresses most trades; additional meta gating adds no value until deployable thresholds are retuned to yield meaningful overlap.
- The scheduler/trading dry run continues to rely solely on manifest gates; keep meta gating disabled in `TRADING_MODELS` until label balance improves and faux P&L/coverage justify another layer.
- Any future attempt must reuse the sanitized multi-symbol dataset (`training.data.sanitize_market_dataset` → `datasets/market_multi_3symbol_1m.parquet`) plus the matching gate payload (`release/symbol_gates/market_multi_3symbol_1m.json`) and capture feature parity diffs via `scripts/export_feature_slice.py` + `scripts/compare_feature_stats.py` so we know meta labels observe the same `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` drift as trading.

## Next Steps
1. Wait for deployable gate expansion beyond the current 5e-4 floor (especially on the TCN suite) to restore meaningful overlap on forward windows before attempting further meta training; the current stride‑1 sandbox runs are for analysis only.
2. Once coverage stabilises, explore asymmetric barrier settings or longer holding periods to diversify labels.
3. Defer deployment of meta models; rely on manifest gates + blender until a refreshed matrix demonstrates ≥1.2 equity and meaningful turnover after meta filtering.
4. Document in the trading runbook that meta signals remain disabled (`order_notional` not configured) to avoid confusion during dry-run retros.
