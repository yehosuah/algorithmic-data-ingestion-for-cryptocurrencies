# Performance Sweep Summary

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
Generated via `python -m portfolio.run_perf_sweeps --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml --best-model-configs configs/best_model_configs.yaml --risk-limits configs/portfolio_risk_limits.yaml --sweep-config configs/perf_sweep_scenarios.yaml --base-output-dir experiments/perf_sweeps`. Full CSV: `experiments/perf_sweeps/summary.csv`.

- Promoted bundle: `medium_xgb_low_cost` (Sharpe **278.1**, pnl_net **1021.6**, `trade_count 100`, `fraction_time_in_position 0.515`, turnover `7.3e-4`, cost 0.25 bps, long-only) → artifacts under `experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final`, policies under `experiments/perf_sweeps/medium_xgb_low_cost/final_policies.yaml`, mirrored to `configs/final_portfolio_policies.yaml` and `configs/deployment_portfolio_contract.yaml`.
- Trigger optimizer builds on this bundle to sweep entry/exit/hold/SL/TP/spread guardrails (`analysis.trigger_optimizer` with `configs/trigger_search_space*.yaml`) and promotes the winner to `configs/final_trigger_policy.yaml`; use `scripts/trigger_preflight.py` to catch zero-coverage trigger configs before rehearsal.

Coverage thresholds: trade_count >= 100, fraction_time_in_position >= 0.010.
## Top scenarios
scenario_id,models,max_rows,oos_fraction,primary_pnl_net,primary_sharpe,primary_max_drawdown,primary_turnover,trade_count,fraction_time_in_position,avg_gross_exposure,transaction_cost_bps,long_only,primary_metrics_path,final_policies_path,final_dir
medium_xgb_low_cost,xgb,150000,0.2,1021.5995,278.0551821839252,1.999999994950485e-05,0.0007310475911981,100,0.5149171639054635,0.1129249214123839,0.25,True,experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/portfolio_final_metrics.json,experiments/perf_sweeps/medium_xgb_low_cost/final_policies.yaml,experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final
medium_xgb_realistic_cost,xgb,150000,0.2,1021.5988,278.05495975491243,4.8000000106185325e-05,0.0007310475911981,100,0.5149171639054635,0.1129249214123839,0.6,True,experiments/perf_sweeps/medium_xgb_realistic_cost/portfolio_final/portfolio_final_metrics.json,experiments/perf_sweeps/medium_xgb_realistic_cost/final_policies.yaml,experiments/perf_sweeps/medium_xgb_realistic_cost/portfolio_final
medium_xgb_high_cost,xgb,150000,0.2,1021.598,278.0547055409233,8.000000002539309e-05,0.0007310475911981,100,0.5149171639054635,0.1129249214123839,1.0,True,experiments/perf_sweeps/medium_xgb_high_cost/portfolio_final/portfolio_final_metrics.json,experiments/perf_sweeps/medium_xgb_high_cost/final_policies.yaml,experiments/perf_sweeps/medium_xgb_high_cost/portfolio_final
