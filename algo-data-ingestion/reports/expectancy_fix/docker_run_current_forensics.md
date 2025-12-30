# Docker Run Current Forensics (Executed Exits)

- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/data_lake/trading/audit.log`
- Window: 2025-12-29T02:49:30+00:00 → <none>
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |    cvar_95 |   max_loss |       pnl |
|:---------|---------:|-----------:|----------:|-----------:|---------------:|----------------:|-----------:|-----------:|----------:|
| BTC/USDT |       39 |   0.128205 | 0.0137673 | -0.0156066 |       0.882145 |        0.129727 | -0.0164083 | -0.0518731 | -0.461788 |
| ETH/USDT |       38 |   0.315789 | 0.0775243 | -0.0434991 |       1.78221  |        0.822557 | -0.0467645 | -0.113309  | -0.200684 |
| SOL/USDT |       43 |   0.348837 | 0.018284  | -0.0230186 |       0.794312 |        0.425525 | -0.0247278 | -0.0638619 | -0.370262 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      61 |     1.93046    |  0.837101    |
| gate_close            |      26 |     0.373955   |  0.162157    |
| prob_trailing         |       1 |     0.00171009 |  0.000741545 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|          0.691667 |        0.00901179 |                       0.325 |                   11 |                    0.00340471 |             0.0239559 |                      0.727273 |

## BTC/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |    p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|------------:|-----------:|-----------:|----------:|
|       39 | -0.461788 |   0.128205 | 0.0137673 | -0.0156066 |       0.882145 |        0.129727 |           0.473396 | -0.0047435 | -0.00241609 | -0.0164083 | -0.0518731 | 0.0340232 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      21 |     0.337194   |   0.635465   |
| gate_close            |      12 |     0.191721   |   0.361312   |
| prob_trailing         |       1 |     0.00171009 |   0.00322279 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.717949 |        0.00490908 |                    0.384615 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|------------:|-----------:|-----------:|----------:|
|       38 | -0.200684 |   0.315789 | 0.0775243 | -0.0434991 |        1.78221 |        0.822557 |           0.671985 | -0.00762019 | -0.00334377 | -0.0467645 |  -0.113309 |  0.209732 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      23 |      1.09021   |    0.963954  |
| gate_close            |       3 |      0.0407677 |    0.0360465 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|          0.763158 |         0.0204007 |                    0.131579 |                    8 |                    0.00391209 |              0.030301 |                          0.75 |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |     p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|-------------:|-----------:|-----------:|----------:|
|       43 | -0.370262 |   0.348837 |  0.018284 | -0.0230186 |       0.794312 |        0.425525 |           0.388021 | -0.00136416 | -0.000779965 | -0.0247278 | -0.0638619 | 0.0536538 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      17 |       0.503055 |      0.78051 |
| gate_close            |      11 |       0.141466 |      0.21949 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|          0.604651 |        0.00266822 |                     0.44186 |                    3 |                    0.00205169 |            0.00703572 |                      0.666667 |
