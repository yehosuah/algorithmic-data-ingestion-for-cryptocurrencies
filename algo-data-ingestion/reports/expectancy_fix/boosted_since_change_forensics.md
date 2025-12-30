# Boosted Since Change Forensics (Executed Exits)

- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/data_lake/trading/audit.log`
- Window: 2025-12-26T02:01:00+00:00 → 2025-12-29T01:30:00+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |     cvar_95 |   max_loss |        pnl |
|:---------|---------:|-----------:|-----------:|------------:|---------------:|----------------:|------------:|-----------:|-----------:|
| BTC/USDT |      139 |   0.309353 | 0.00805155 | -0.00647993 |        1.24254 |        0.556553 | -0.00682562 | -0.0356204 | -0.275857  |
| ETH/USDT |      119 |   0.344538 | 0.00504593 | -0.0029213  |        1.72729 |        0.907934 | -0.00307142 | -0.0130143 | -0.0209782 |
| SOL/USDT |      190 |   0.384211 | 0.0129115  | -0.0116272  |        1.11046 |        0.692851 | -0.0122368  | -0.0579824 | -0.417839  |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |     246 |       1.80191  |     0.81523  |
| gate_close            |      41 |       0.24186  |     0.109423 |
| stop_loss             |       4 |       0.166541 |     0.075347 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.473214 |        0.00719091 |                           0 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |       pnl |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |     cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|-----------:|------------:|---------------:|----------------:|-------------------:|-------------:|-------------:|------------:|-----------:|----------:|
|      139 | -0.275857 |   0.309353 | 0.00805155 | -0.00647993 |        1.24254 |        0.556553 |           0.445576 | -0.000320591 | -0.000104564 | -0.00682562 | -0.0356204 | 0.0986606 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      84 |      0.580021  |    0.932399  |
| gate_close            |      12 |      0.0420528 |    0.0676011 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.417266 |        0.00806774 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate |    avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |     cvar_95 |   max_loss |   max_win |
|---------:|-----------:|-----------:|-----------:|-----------:|---------------:|----------------:|-------------------:|-------------:|-------------:|------------:|-----------:|----------:|
|      119 | -0.0209782 |   0.344538 | 0.00504593 | -0.0029213 |        1.72729 |        0.907934 |           0.110845 | -0.000253226 | -7.83636e-05 | -0.00307142 | -0.0130143 | 0.0488159 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      76 |    0.224849    |   0.986779   |
| stop_loss             |       1 |    0.0026253   |   0.0115215  |
| gate_close            |       1 |    0.000387184 |   0.00169921 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.394958 |         0.0103733 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-------------:|-------------:|-----------:|-----------:|----------:|
|      190 | -0.417839 |   0.384211 | 0.0129115 | -0.0116272 |        1.11046 |        0.692851 |           0.587775 | -0.000552704 | -0.000367002 | -0.0122368 | -0.0579824 |  0.101833 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      86 |       0.997044 |     0.732916 |
| gate_close            |      28 |       0.19942  |     0.146592 |
| stop_loss             |       3 |       0.163915 |     0.120492 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.563158 |        0.00455623 |                           0 |                    0 |                               |                       |                               |
