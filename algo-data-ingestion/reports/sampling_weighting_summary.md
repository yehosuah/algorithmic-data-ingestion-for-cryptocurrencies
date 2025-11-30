# Sampling & Weighting Summary

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
Generated via `python training/run_sampling_weighting_experiments.py --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml --best-configs configs/best_model_configs.yaml --experiment-config configs/sampling_weighting_experiments.yaml --cv-config configs/cv_config.yaml --output-file experiments/sampling_weighting_comparison.csv`. Full CSV lives at `experiments/sampling_weighting_comparison.csv`.

## Top combinations
- tcn | sampling=uniform | weight=none | sharpe=214.5489 | pnl=2.808387 | n=1
- tcn | sampling=vol_weighted | weight=cost_capacity_combo | sharpe=214.0007 | pnl=2.806475 | n=1
- xgb | sampling=regime_balanced | weight=none | sharpe=211.6658 | pnl=27.333700 | n=1
- xgb | sampling=uniform | weight=cost_capacity_combo | sharpe=211.6658 | pnl=27.333700 | n=1
- xgb | sampling=uniform | weight=none | sharpe=211.6658 | pnl=27.333700 | n=1

## Recommendation for xgb
- Recommend sampling=regime_balanced weight=none (sharpe=211.6658, pnl=27.333700)

## Recommendation for tcn
- Recommend sampling=uniform weight=none (sharpe=214.5489, pnl=2.808387)
