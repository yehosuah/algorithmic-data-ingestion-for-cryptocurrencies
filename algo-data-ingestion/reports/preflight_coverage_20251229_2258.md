# Preflight Coverage

Generated at: 2025-12-29T22:58:04.188963+00:00
Contract: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/deployment_portfolio_contract.yaml
Features: data/features_labels_regimes_market_multi_3symbol_1m.parquet

## Warnings
- BTC/USDT: risk_limits prob_gate_min=0.48 differs from manifest=0.56 (preflight uses manifest)

## ETH/USDT
- Model: xgb_primary
- Samples: 151
- Prob quantiles: p50=0.5657 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.48 | fraction>=gate: 0.7351
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.2185
- Implied trade proxy (signal changes): 42

## BTC/USDT
- Model: xgb_primary
- Samples: 180
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.6844
- Prob gate min: 0.56 | fraction>=gate: 0.3611
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.1333
- Implied trade proxy (signal changes): 31

## SOL/USDT
- Model: xgb_primary
- Samples: 181
- Prob quantiles: p50=0.5002 p90=0.6844 p95=0.6844 p99=0.7031
- Prob gate min: 0.48 | fraction>=gate: 0.6133
- Entry/exit: None / None | fraction between: 0.0000
- Gate coverage ratio: 0.1547
- Implied trade proxy (signal changes): 32
