# Current Since Change Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251229T014037Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251229T014037Z/trading_audit/audit.log`
- Window: 2025-12-26T02:00:00+00:00 → 2025-12-29T01:40:37+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |     cvar_95 |   max_loss |        pnl |
|:---------|---------:|-----------:|-----------:|------------:|---------------:|----------------:|------------:|-----------:|-----------:|
| BTC/USDT |      139 |   0.309353 | 0.00805155 | -0.00647993 |        1.24254 |        0.556553 | -0.00682562 | -0.0356204 | -0.275857  |
| ETH/USDT |      120 |   0.35     | 0.00492669 | -0.0029213  |        1.68647 |        0.9081   | -0.00307142 | -0.0130143 | -0.0209405 |
| SOL/USDT |      191 |   0.382199 | 0.0129115  | -0.0118047  |        1.09376 |        0.676648 | -0.0124184  | -0.0579824 | -0.450415  |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |     247 |       1.83449  |    0.817913  |
| gate_close            |      41 |       0.24186  |    0.107834  |
| stop_loss             |       4 |       0.166541 |    0.0742527 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.475556 |        0.00728039 |                           0 |                    0 |                               |                       |                               |

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
|          0.417266 |        0.00806798 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate |    avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |     cvar_95 |   max_loss |   max_win |
|---------:|-----------:|-----------:|-----------:|-----------:|---------------:|----------------:|-------------------:|-------------:|-------------:|------------:|-----------:|----------:|
|      120 | -0.0209405 |       0.35 | 0.00492669 | -0.0029213 |        1.68647 |          0.9081 |           0.110845 | -0.000253226 | -7.83636e-05 | -0.00307142 | -0.0130143 | 0.0488159 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      76 |    0.224849    |   0.986779   |
| stop_loss             |       1 |    0.0026253   |   0.0115215  |
| gate_close            |       1 |    0.000387184 |   0.00169921 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.4 |         0.0103628 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-------------:|-------------:|-----------:|-----------:|----------:|
|      191 | -0.450415 |   0.382199 | 0.0129115 | -0.0118047 |        1.09376 |        0.676648 |           0.587775 | -0.000562568 | -0.000367059 | -0.0124184 | -0.0579824 |  0.101833 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      87 |       1.02962  |     0.739162 |
| gate_close            |      28 |       0.19942  |     0.143163 |
| stop_loss             |       3 |       0.163915 |     0.117674 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.565445 |        0.00477064 |                           0 |                    0 |                               |                       |                               |
