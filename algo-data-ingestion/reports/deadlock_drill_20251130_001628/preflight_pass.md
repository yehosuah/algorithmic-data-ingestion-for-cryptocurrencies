# Preflight Coverage

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.


Generated at: 2025-11-30T00:13:11.942408+00:00
Contract: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/deployment_portfolio_contract.yaml
Features: data/features_labels_regimes_market_multi_3symbol_1m.parquet

## ETH/USDT
- Model: xgb_primary
- Samples: 147
- Prob quantiles: p50=0.5657 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.52 | fraction>=gate: 0.5102
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.5102
- Implied trade proxy (signal changes): 32

## BTC/USDT
- Model: xgb_primary
- Samples: 176
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.52 | fraction>=gate: 0.3636
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.3636
- Implied trade proxy (signal changes): 24

## SOL/USDT
- Model: xgb_primary
- Samples: 177
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.7068
- Prob gate min: 0.52 | fraction>=gate: 0.4972
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.4972
- Implied trade proxy (signal changes): 20
