# Post Change Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251225T215109Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251225T215109Z/trading_audit/audit.log`
- Window: 2025-12-25T21:23:00+00:00 → 2025-12-25T21:47:00+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |      avg_win |     avg_loss |   payoff_ratio |   profit_factor |      cvar_95 |     max_loss |          pnl |
|:---------|---------:|-----------:|-------------:|-------------:|---------------:|----------------:|-------------:|-------------:|-------------:|
| BTC/USDT |        1 |        0   | nan          | -0.000891305 |     nan        |        0        | -0.000891305 | -0.000891305 | -0.000891305 |
| ETH/USDT |        1 |        0   | nan          | -0.0197418   |     nan        |        0        | -0.0197418   | -0.0197418   | -0.0197418   |
| SOL/USDT |        2 |        0.5 |   0.00522656 | -0.00597392  |       0.874896 |        0.874896 | -0.00597392  | -0.00597392  | -0.00074736  |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       2 |     0.0206331  |     0.775476 |
| gate_close            |       1 |     0.00597392 |     0.224524 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 0 |       0.000579408 |                           0 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |          pnl |   win_rate | avg_win   |     avg_loss | payoff_ratio   |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |      cvar_95 |     max_loss | max_win   |
|---------:|-------------:|-----------:|:----------|-------------:|:---------------|----------------:|-------------------:|-------------:|-------------:|-------------:|-------------:|:----------|
|        1 | -0.000891305 |          0 |           | -0.000891305 |                |               0 |        0.000891305 | -0.000891305 | -0.000891305 | -0.000891305 | -0.000891305 |           |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |    0.000891305 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 0 |       0.000180001 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate | avg_win   |   avg_loss | payoff_ratio   |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |    cvar_95 |   max_loss | max_win   |
|---------:|-----------:|-----------:|:----------|-----------:|:---------------|----------------:|-------------------:|-----------:|-----------:|-----------:|-----------:|:----------|
|        1 | -0.0197418 |          0 |           | -0.0197418 |                |               0 |          0.0197418 | -0.0197418 | -0.0197418 | -0.0197418 | -0.0197418 |           |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |      0.0197418 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 0 |       0.000921718 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |         pnl |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |     cvar_95 |    max_loss |    max_win |
|---------:|------------:|-----------:|-----------:|------------:|---------------:|----------------:|-------------------:|------------:|------------:|------------:|------------:|-----------:|
|        2 | -0.00074736 |        0.5 | 0.00522656 | -0.00597392 |       0.874896 |        0.874896 |         0.00597392 | -0.00597392 | -0.00597392 | -0.00597392 | -0.00597392 | 0.00522656 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       1 |     0.00597392 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 0 |       0.000607957 |                           0 |                    0 |                               |                       |                               |
