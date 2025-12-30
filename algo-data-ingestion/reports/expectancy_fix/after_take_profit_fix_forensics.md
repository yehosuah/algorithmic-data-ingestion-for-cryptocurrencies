# After Take Profit Fix Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251229T030135Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251229T030135Z/trading_audit/audit.log`
- Window: 2025-12-29T02:00:00+00:00 → 2025-12-29T03:01:35+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |   avg_win |    avg_loss |   payoff_ratio |   profit_factor |    cvar_95 |   max_loss |        pnl |
|:---------|---------:|-----------:|----------:|------------:|---------------:|----------------:|-----------:|-----------:|-----------:|
| BTC/USDT |        3 |   0.333333 | 0.0336731 | -0.00974985 |        3.4537  |        1.72685  | -0.0191419 | -0.0191419 |  0.0141734 |
| ETH/USDT |        4 |   0.75     | 0.0187052 | -0.0026105  |        7.16537 |       21.4961   | -0.0026105 | -0.0026105 |  0.0535052 |
| SOL/USDT |        3 |   0.333333 | 0.0226974 | -0.0213881  |        1.06121 |        0.530607 | -0.0284053 | -0.0284053 | -0.0200788 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       3 |     0.0619181  |    0.954254  |
| prob_floor            |       2 |     0.00296833 |    0.0457465 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.6 |        0.00301056 |                         0.2 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |     p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|------------:|---------------:|----------------:|-------------------:|------------:|-------------:|-----------:|-----------:|----------:|
|        3 | 0.0141734 |   0.333333 | 0.0336731 | -0.00974985 |         3.4537 |         1.72685 |          0.0194997 | -0.00129702 | -0.000545662 | -0.0191419 | -0.0191419 | 0.0336731 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       1 |    0.0191419   |    0.98165   |
| prob_floor            |       1 |    0.000357822 |    0.0183501 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 1 |        0.00378654 |                    0.333333 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|-----------:|-----------:|----------:|
|        4 | 0.0535052 |       0.75 | 0.0187052 | -0.0026105 |        7.16537 |         21.4961 |          0.0026105 | -0.0026105 | -0.0026105 | -0.0026105 | -0.0026105 | 0.0448095 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |      0.0026105 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.5 |        0.00346461 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |        pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|-----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|-----------:|-----------:|----------:|
|        3 | -0.0200788 |   0.333333 | 0.0226974 | -0.0213881 |        1.06121 |        0.530607 |          0.0284053 | -0.0150726 | -0.0145112 | -0.0284053 | -0.0284053 | 0.0226974 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| gate_close            |       2 |      0.0427762 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.333333 |        0.00162916 |                    0.333333 |                    0 |                               |                       |                               |
