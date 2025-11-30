# Preflight Coverage

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.


Generated at: 2025-11-30T01:32:57.220623+00:00
Contract: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/deployment_portfolio_contract.yaml
Features: data/features_labels_regimes_market_multi_3symbol_1m.parquet

## ETH/USDT
- Model: xgb_primary
- Samples: 151
- Prob quantiles: p50=0.5657 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.52 | fraction>=gate: 0.5099
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.5099
- Implied trade proxy (signal changes): 35

## BTC/USDT
- Model: xgb_primary
- Samples: 180
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.52 | fraction>=gate: 0.3611
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.3611
- Implied trade proxy (signal changes): 27

## SOL/USDT
- Model: xgb_primary
- Samples: 181
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.7031
- Prob gate min: 0.52 | fraction>=gate: 0.4917
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.4917
- Implied trade proxy (signal changes): 23
