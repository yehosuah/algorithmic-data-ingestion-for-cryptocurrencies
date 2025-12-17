# Project Stance Snapshot

Generated at: 2025-12-15T02:27:03.979828Z

## Models
- xgb_primary: label=xgb_primary path=experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary (manifest_mtime=2025-12-15T00:21:10.583590Z)

## Symbols
- ETH/USDT: policy=primary shadow=False notional=80.0 max_spread_bps=15 entry_thr=0.48 exit_thr=0.47 min_hold_bars=3 take_profit_pct=0.0003 stop_loss_pct=0.0
- BTC/USDT: policy=conservative shadow=False notional=50.0 max_spread_bps=20 entry_thr=0.56 exit_thr=0.47 min_hold_bars=3 take_profit_pct=0.0003 stop_loss_pct=0.0
- SOL/USDT: policy=conservative shadow=False notional=40.0 max_spread_bps=22 entry_thr=0.48 exit_thr=0.47 min_hold_bars=3 take_profit_pct=0.0003 stop_loss_pct=0.0

## Risk limits
Source: configs/runtime_overrides/risk_limits_stage_0.yaml
{
  "capital": 1000000,
  "max_gross_leverage": 3.0,
  "max_net_exposure": 1.5,
  "max_turnover_per_day": 1.0,
  "max_orders_per_hour": 120,
  "max_concurrent_positions": 5,
  "max_symbol_weight": 0.2,
  "max_symbol_notional": 100000,
  "max_notional_per_symbol": 120000,
  "max_total_notional": 120000,
  "max_daily_turnover": 0.5,
  "daily_loss_limit_pct": 0.03,
  "max_drawdown_pct": 0.15,
  "max_drawdown": 0.15,
  "cooldown_minutes_after_exit": 0,
  "cooldown_minutes_after_loss": 0,
  "halt_on_safe_mode": true,
  "allow_exits_during_halt": true,
  "halt_if_spread_bps_gt": 35.0,
  "halt_if_vol_zscore_gt": 5.0,
  "halt_if_missing_price_bars": false,
  "halt_if_data_stale_seconds": 180,
  "min_trade_notional": 0.0001,
  "transaction_cost_bps": 1.0,
  "slippage_bps": 0.0,
  "spread_scale": 0.0,
  "spread_column": "hl_spread",
  "gate_mode": "inference",
  "long_only": false,
  "model_weights": {},
  "ensemble_mode": "weighted_sum",
  "gate_config": {
    "spread_column": "hl_spread",
    "prob_column": "base_prob",
    "training": {
      "hl_spread_max": null,
      "hl_spread_z_max": null,
      "rvol20_max": null,
      "prob_gate_min": 0.48,
      "min_hold_bars": 3,
      "long_only": false
    },
    "inference": {
      "hl_spread_max": null,
      "hl_spread_z_max": null,
      "rvol20_max": 0.002,
      "prob_gate_min": 0.48,
      "min_hold_bars": 3,
      "long_only": true
    }
  },
  "symbols": {
    "BTC/USDT": {
      "max_symbol_notional": 15000,
      "max_symbol_weight": 0.2,
      "max_spread_bps": 25,
      "min_trade_notional": 0.0001,
      "qty_step": 0.0001,
      "price_tick": 0.01,
      "max_entry_slippage_bps": 25,
      "max_position_age_minutes": 360
    },
    "ETH/USDT": {
      "max_symbol_notional": 20000,
      "max_symbol_weight": 0.2,
      "max_spread_bps": 25,
      "min_trade_notional": 0.0001,
      "qty_step": 0.001,
      "price_tick": 0.01,
      "max_entry_slippage_bps": 25,
      "max_position_age_minutes": 360
    },
    "SOL/USDT": {
      "max_symbol_notional": 8000,
      "max_symbol_weight": 0.15,
      "max_spread_bps": 28,
      "min_trade_notional": 0.0001,
      "qty_step": 0.01,
      "price_tick": 0.001,
      "max_entry_slippage_bps": 30,
      "max_position_age_minutes": 360
    }
  },
  "loss_guard": {
    "enabled": true,
    "max_consecutive_losses": 3,
    "cooldown_minutes": 90,
    "prob_buffer": 0.0,
    "notional_scale": 0.5,
    "reset_after_profit": true
  }
}

## Deadlock policy
{
  "enabled": true,
  "window_minutes": 30,
  "min_trades_window": 1,
  "min_coverage_ratio_window": 0.02,
  "cooldown_minutes": 45,
  "max_actions_per_day": 2,
  "audit_every_action": true,
  "adjust_prob_gate_min": {
    "step": 0.02,
    "floor": 0.5
  },
  "actions": [
    {
      "adjust_prob_gate_min": {
        "step": 0.02,
        "floor": 0.5
      }
    },
    {
      "enter_safe_mode": true
    }
  ]
}

## Env (trading)
- DECISION_QUEUE_KEY=trading:decisions
- DECISION_QUEUE_URL=redis://redis:6379/0
- TRADING_AUDIT_BACKEND=file
- TRADING_AUDIT_HMAC_KEY=${TRADING_AUDIT_HMAC_KEY}
- TRADING_AUDIT_LOG_PATH=/app/data_lake/trading/audit.log
- TRADING_AUDIT_REDIS_URL=redis://redis:6379/0
- TRADING_DEADLOCK_POLICY_PATH=configs/runtime_overrides/deadlock_policy_stage_0.yaml
- TRADING_DECISION_MAX_AGE_SECONDS=240
- TRADING_DRY_RUN=true
- TRADING_INTENT_LEDGER_BACKEND=redis
- TRADING_INTENT_LEDGER_REDIS_URL=redis://redis:6379/0
- TRADING_LAST_TS_GRACE_BARS=10
- TRADING_LOG_LEVEL=DEBUG
- TRADING_METRICS_PORT=9010
- TRADING_MODELS=[{"model":"xgb_primary","symbol":"ETH/USDT","exchange":"binance","timeframe":"1m","order_notional":80.0,"max_spread_bps":15,"stop_loss_pct":0.0,"take_profit_pct":0.0003,"max_hold_minutes":60,"shadow_mode":false,"policy_id":"primary","min_hold_bars_override":3},{"model":"xgb_primary","symbol":"BTC/USDT","exchange":"binance","timeframe":"1m","order_notional":50.0,"max_spread_bps":20,"stop_loss_pct":0.0,"take_profit_pct":0.0003,"max_hold_minutes":60,"shadow_mode":false,"policy_id":"conservative","min_hold_bars_override":3},{"model":"xgb_primary","symbol":"SOL/USDT","exchange":"binance","timeframe":"1m","order_notional":40.0,"max_spread_bps":22,"stop_loss_pct":0.0,"take_profit_pct":0.0003,"max_hold_minutes":60,"shadow_mode":false,"policy_id":"conservative","min_hold_bars_override":3}]
- TRADING_RISK_LIMITS_PATH=configs/runtime_overrides/risk_limits_stage_0.yaml
- TRADING_SHADOW_SYMBOLS=[]
- TRADING_STATE_BACKEND=file
- TRADING_STATE_PATH=/app/trading_state/state.json
- TRADING_STATE_REDIS_URL=redis://redis:6379/0

## Git
Recent commits:
- a3b8b4b8 Refresh docs for new trading posture
- d14d8bc8 Prune tracked dataset artifacts
- aa3c29c0 Tighten stage-0 trading cadence
- a388e3e0 docs: refresh launch runbooks and readiness reports
- 7a4be51a feat: add launch-stage tooling and runtime risk safeguards

Diffstat:
algo-data-ingestion/Dockerfile                     |   1 +
 algo-data-ingestion/RUNBOOK_LIVE_LAUNCH.md         |  24 +-
 algo-data-ingestion/analysis/preflight_coverage.py |  33 +-
 .../app/ingestion_service/routes.py                | 176 ++---
 algo-data-ingestion/app/ingestion_service/utils.py |  25 +-
 algo-data-ingestion/app/scheduler/main.py          | 235 ++++++-
 algo-data-ingestion/app/trading/audit.py           |   3 +
 algo-data-ingestion/app/trading/config.py          |   5 +
 algo-data-ingestion/app/trading/deadlock.py        |   7 +-
 algo-data-ingestion/app/trading/decision.py        | 129 +++-
 algo-data-ingestion/app/trading/executor.py        |  40 ++
 algo-data-ingestion/app/trading/service.py         | 709 ++++++++++++++++++++-
 algo-data-ingestion/app/trading/state.py           |   7 +
 .../configs/deployment_portfolio_contract.yaml     |   4 +-
 .../dry_run/infer_jobs_portfolio_policy.yaml       |  14 +-
 .../configs/final_trigger_policy.yaml              |  32 +-
 .../runtime_overrides/risk_limits_stage_0.yaml     |  29 +-
 .../configs/runtime_overrides/stage_0.yaml         |   9 +-
 algo-data-ingestion/docker-compose.yml             |  31 +-
 .../models/final_xgb_primary/manifest.json         |  12 +-
 algo-data-ingestion/labels/label_generator.py      |  87 +++
 .../tests/ingestion_service/test_utils.py          |   2 +-
 algo-data-ingestion/tests/test_decision_logic.py   |  35 +-
 23 files changed, 1463 insertions(+), 186 deletions(-)