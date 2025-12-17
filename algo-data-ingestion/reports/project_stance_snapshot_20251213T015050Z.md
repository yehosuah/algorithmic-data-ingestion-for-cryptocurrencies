# Project Stance Snapshot

Generated at: 2025-12-13T01:50:50.694335Z

## Models
- xgb_primary: experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary (manifest_mtime=2025-12-01T21:33:26.224658Z)

## Symbols
- ETH/USDT: policy=primary shadow=False notional=80.0 thresholds={'entry_threshold': 0.495, 'exit_threshold': 0.485, 'exit_prob_drop': 0.1, 'min_hold_bars': 1, 'stop_loss_pct': None, 'take_profit_pct': 0.0002}
- BTC/USDT: policy=conservative shadow=False notional=50.0 thresholds={'entry_threshold': 0.495, 'exit_threshold': 0.485, 'exit_prob_drop': 0.1, 'min_hold_bars': 1, 'stop_loss_pct': None, 'take_profit_pct': 0.0002}
- SOL/USDT: policy=conservative shadow=False notional=40.0 thresholds={'entry_threshold': 0.495, 'exit_threshold': 0.485, 'exit_prob_drop': 0.1, 'min_hold_bars': 1, 'stop_loss_pct': None, 'take_profit_pct': 0.0002}

## Risk limits
Source: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/runtime_overrides/risk_limits_stage_0.yaml
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
      "prob_gate_min": 0.49,
      "min_hold_bars": 1,
      "long_only": false
    },
    "inference": {
      "hl_spread_max": null,
      "hl_spread_z_max": null,
      "rvol20_max": null,
      "prob_gate_min": 0.49,
      "min_hold_bars": 1,
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
    "prob_buffer": 0.01,
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
 algo-data-ingestion/app/scheduler/main.py          | 179 +++++-
 algo-data-ingestion/app/trading/audit.py           |   3 +
 algo-data-ingestion/app/trading/config.py          |   5 +
 algo-data-ingestion/app/trading/decision.py        | 111 +++-
 algo-data-ingestion/app/trading/service.py         | 649 ++++++++++++++++++++-
 algo-data-ingestion/app/trading/state.py           |   7 +
 .../configs/deployment_portfolio_contract.yaml     |   4 +-
 .../dry_run/infer_jobs_portfolio_policy.yaml       |  14 +-
 .../configs/final_trigger_policy.yaml              |  26 +-
 .../runtime_overrides/risk_limits_stage_0.yaml     |  23 +-
 .../configs/runtime_overrides/stage_0.yaml         |   9 +-
 algo-data-ingestion/docker-compose.yml             |  25 +-
 .../models/final_xgb_primary/manifest.json         |   9 +-
 algo-data-ingestion/tests/test_decision_logic.py   |  35 +-
 16 files changed, 1047 insertions(+), 77 deletions(-)