# Boosted Live Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251226T020251Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251226T020251Z/trading_audit/audit.log`
- Window: 2025-12-26T01:30:00+00:00 → 2025-12-26T02:05:00+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |    avg_win |     avg_loss |   payoff_ratio |   profit_factor |      cvar_95 |     max_loss |        pnl |
|:---------|---------:|-----------:|-----------:|-------------:|---------------:|----------------:|-------------:|-------------:|-----------:|
| BTC/USDT |        2 |        0.5 | 0.0074346  | -0.000803381 |        9.25413 |         9.25413 | -0.000803381 | -0.000803381 | 0.00663121 |
| ETH/USDT |        2 |        0.5 | 0.00517736 | -0.0002339   |       22.1349  |        22.1349  | -0.0002339   | -0.0002339   | 0.00494346 |
| SOL/USDT |        2 |        0.5 | 0.0148694  | -0.00513488  |        2.89577 |         2.89577 | -0.00513488  | -0.00513488  | 0.00973456 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       1 |     0.00513488 |     0.831942 |
| prob_floor            |       2 |     0.00103728 |     0.168058 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.833333 |        0.00813454 |                           0 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |        pnl |   win_rate |   avg_win |     avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |      cvar_95 |     max_loss |   max_win |
|---------:|-----------:|-----------:|----------:|-------------:|---------------:|----------------:|-------------------:|-------------:|-------------:|-------------:|-------------:|----------:|
|        2 | 0.00663121 |        0.5 | 0.0074346 | -0.000803381 |        9.25413 |         9.25413 |        0.000803381 | -0.000803381 | -0.000803381 | -0.000803381 | -0.000803381 | 0.0074346 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |    0.000803381 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 1 |        0.00454136 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate |    avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |    cvar_95 |   max_loss |    max_win |
|---------:|-----------:|-----------:|-----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|-----------:|-----------:|-----------:|
|        2 | 0.00494346 |        0.5 | 0.00517736 | -0.0002339 |        22.1349 |         22.1349 |          0.0002339 | -0.0002339 | -0.0002339 | -0.0002339 | -0.0002339 | 0.00517736 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |      0.0002339 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 1 |         0.0178728 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |        pnl |   win_rate |   avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |     cvar_95 |    max_loss |   max_win |
|---------:|-----------:|-----------:|----------:|------------:|---------------:|----------------:|-------------------:|------------:|------------:|------------:|------------:|----------:|
|        2 | 0.00973456 |        0.5 | 0.0148694 | -0.00513488 |        2.89577 |         2.89577 |         0.00513488 | -0.00513488 | -0.00513488 | -0.00513488 | -0.00513488 | 0.0148694 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       1 |     0.00513488 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.5 |        0.00198948 |                           0 |                    0 |                               |                       |                               |
