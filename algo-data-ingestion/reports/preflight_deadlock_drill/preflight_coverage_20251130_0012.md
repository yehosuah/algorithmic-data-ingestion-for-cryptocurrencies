# Preflight Coverage

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.


Generated at: 2025-11-30T00:12:44.654833+00:00
Contract: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/deployment_portfolio_contract.yaml
Features: experiments/dummy_live_features.parquet

## NO-GO reasons
- ETH/USDT: fraction_above_prob_gate_min<=epsilon (0.000000)
- BTC/USDT: no samples available for coverage estimation
- BTC/USDT: fraction_above_prob_gate_min<=epsilon (0.000000)
- SOL/USDT: no samples available for coverage estimation
- SOL/USDT: fraction_above_prob_gate_min<=epsilon (0.000000)
- implied_trade_proxy==0 across all symbols

## ETH/USDT
- Model: xgb_primary
- Samples: 5
- Prob quantiles: p50=0.5002 p90=0.5002 p95=0.5002 p99=0.5002
- Prob gate min: 0.52 | fraction>=gate: 0.0000
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.0000
- Implied trade proxy (signal changes): 0

## BTC/USDT
- Model: xgb_primary
- Samples: 0
- Prob quantiles: p50=0.0000 p90=0.0000 p95=0.0000 p99=0.0000
- Prob gate min: None | fraction>=gate: 0.0000
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.0000
- Implied trade proxy (signal changes): 0

## SOL/USDT
- Model: xgb_primary
- Samples: 0
- Prob quantiles: p50=0.0000 p90=0.0000 p95=0.0000 p99=0.0000
- Prob gate min: None | fraction>=gate: 0.0000
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.0000
- Implied trade proxy (signal changes): 0
