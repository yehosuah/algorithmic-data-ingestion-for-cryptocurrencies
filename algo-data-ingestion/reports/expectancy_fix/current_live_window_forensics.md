# Current Live Window Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251229T192242Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251229T192242Z/trading_audit/audit.log`
- Window: 2025-12-29T02:49:30+00:00 → <none>
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |    cvar_95 |   max_loss |       pnl |
|:---------|---------:|-----------:|----------:|-----------:|---------------:|----------------:|-----------:|-----------:|----------:|
| BTC/USDT |       37 |   0.135135 | 0.0137673 | -0.0154856 |       0.889038 |        0.138912 | -0.0163327 | -0.0518731 | -0.426703 |
| ETH/USDT |       36 |   0.305556 | 0.0789414 | -0.042736  |       1.84719  |        0.812763 | -0.046077  | -0.113309  | -0.200044 |
| SOL/USDT |       41 |   0.365854 | 0.018284  | -0.0240222 |       0.761127 |        0.439112 | -0.0259575 | -0.0638619 | -0.350319 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      56 |     1.81285    |  0.828347    |
| gate_close            |      26 |     0.373955   |  0.170871    |
| prob_trailing         |       1 |     0.00171009 |  0.000781394 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.684211 |        0.00935124 |                    0.307018 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|------------:|-----------:|-----------:|----------:|
|       37 | -0.426703 |   0.135135 | 0.0137673 | -0.0154856 |       0.889038 |        0.138912 |            0.43831 | -0.00460596 | -0.00237331 | -0.0163327 | -0.0518731 | 0.0340232 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      19 |     0.302108   |   0.609655   |
| gate_close            |      12 |     0.191721   |   0.386894   |
| prob_trailing         |       1 |     0.00171009 |   0.00345098 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.702703 |        0.00506331 |                    0.378378 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |   cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|------------:|----------:|-----------:|----------:|
|       36 | -0.200044 |   0.305556 | 0.0789414 |  -0.042736 |        1.84719 |        0.812763 |           0.671345 | -0.00734735 | -0.00330494 | -0.046077 |  -0.113309 |  0.209732 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      22 |      1.02763   |    0.961842  |
| gate_close            |       3 |      0.0407677 |    0.0381577 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.777778 |         0.0214217 |                    0.111111 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |    p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|------------:|-----------:|-----------:|----------:|
|       41 | -0.350319 |   0.365854 |  0.018284 | -0.0240222 |       0.761127 |        0.439112 |           0.368078 | -0.0012144 | -0.00077832 | -0.0259575 | -0.0638619 | 0.0536538 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      15 |       0.483112 |     0.773501 |
| gate_close            |      11 |       0.141466 |     0.226499 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.585366 |        0.00262235 |                    0.414634 |                    0 |                               |                       |                               |
