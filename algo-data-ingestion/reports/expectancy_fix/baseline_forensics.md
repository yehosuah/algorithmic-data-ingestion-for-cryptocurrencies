# Baseline Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251225T181022Z`
- Audit log: `reports/log_forensics/evidence/20251225T181022Z/trading_audit/audit.log`
- Window: 2025-12-24T20:33:41+00:00 → 2025-12-25T18:08:00+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |     cvar_95 |    max_loss |        pnl |
|:---------|---------:|-----------:|-----------:|------------:|---------------:|----------------:|------------:|------------:|-----------:|
| BTC/USDT |        7 |   0.285714 | 0.00510981 | -0.00597682 |       0.854938 |        0.341975 | -0.00659701 | -0.00982058 | -0.0196645 |
| ETH/USDT |       42 |   0.595238 | 0.0211473  | -0.0359917  |       0.587561 |        0.86406  | -0.0378796  | -0.0960756  | -0.0831764 |
| SOL/USDT |       87 |   0.390805 | 0.00801957 | -0.00931447 |       0.860979 |        0.552326 | -0.00985161 | -0.053989   | -0.221002  |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      30 |       0.481451 |    0.424033  |
| prob_trailing         |       9 |       0.213395 |    0.187945  |
| gate_close            |      32 |       0.186873 |    0.164586  |
| stop_loss             |       2 |       0.150065 |    0.132168  |
| time_limit            |       2 |       0.103627 |    0.0912681 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|          0.448529 |        0.00291853 |                           0 |                   38 |                    0.00147521 |            0.00477028 |                      0.473684 |

## BTC/USDT

**Metrics**
|   trades |        pnl |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |   p99_loss |     cvar_95 |    max_loss |    max_win |
|---------:|-----------:|-----------:|-----------:|------------:|---------------:|----------------:|-------------------:|------------:|-----------:|------------:|------------:|-----------:|
|        7 | -0.0196645 |   0.285714 | 0.00510981 | -0.00597682 |       0.854938 |        0.341975 |          0.0189076 | -0.00370721 | -0.0035383 | -0.00659701 | -0.00982058 | 0.00576559 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       3 |      0.0197414 |     0.660598 |
| gate_close            |       2 |      0.0101427 |     0.339402 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.428571 |         0.0021271 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|-----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|------------:|-----------:|-----------:|----------:|
|       42 | -0.0831764 |   0.595238 | 0.0211473 | -0.0359917 |       0.587561 |         0.86406 |           0.277057 | -0.00799976 | -0.00622904 | -0.0378796 | -0.0960756 | 0.0724356 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_trailing         |       9 |      0.213395  |    0.348765  |
| prob_floor            |       2 |      0.160888  |    0.262949  |
| time_limit            |       2 |      0.103627  |    0.169364  |
| stop_loss             |       1 |      0.0960756 |    0.157022  |
| gate_close            |       3 |      0.0378738 |    0.0618995 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|           0.47619 |        0.00282126 |                           0 |                   22 |                    0.00167081 |             0.0041711 |                      0.363636 |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate |    avg_win |    avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |     p95_loss |     p99_loss |     cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|-----------:|------------:|---------------:|----------------:|-------------------:|-------------:|-------------:|------------:|-----------:|----------:|
|       87 | -0.221002 |   0.390805 | 0.00801957 | -0.00931447 |       0.860979 |        0.552326 |           0.227096 | -0.000380448 | -0.000358477 | -0.00985161 |  -0.053989 |  0.034039 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      25 |       0.300822 |     0.609361 |
| gate_close            |      27 |       0.138856 |     0.281275 |
| stop_loss             |       1 |       0.053989 |     0.109363 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades |   take_profit_avg_exit_return |   take_profit_avg_mfe |   take_profit_regret_fraction |
|------------------:|------------------:|----------------------------:|---------------------:|------------------------------:|----------------------:|------------------------------:|
|          0.436782 |        0.00302918 |                           0 |                   16 |                    0.00120626 |            0.00559414 |                         0.625 |
