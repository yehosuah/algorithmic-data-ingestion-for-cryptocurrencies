# Edge diagnostics for xgb

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
Generated via `analysis/xgb_edge_diagnostics.py` on `configs/canonical_training_contract_market_multi_3symbol_1m.yaml` (post-sweep XGB primary artifacts).

## Global signal metrics
- AUC: 0.5826957650800236
- Brier: 0.26093942527585806
- Log loss: 0.7203716342580947
- IC: 0.004487039818580772

## Global PnL metrics (loose gate)
- PnL net: None
- Sharpe: 2.1014659167816006
- Max drawdown: 3161.9925997745204
- Toggle count: 159625

## Per-regime
