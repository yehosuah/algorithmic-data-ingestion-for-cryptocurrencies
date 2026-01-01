# Docker Live 48H Forensics (Executed Exits)

- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/data_lake/trading/audit.log`
- Window: 2025-12-30T00:46:53.678749+00:00 → 2026-01-01T00:46:53.678749+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |    cvar_95 |   max_loss |        pnl |
|:---------|---------:|-----------:|----------:|-----------:|---------------:|----------------:|-----------:|-----------:|-----------:|
| BTC/USDT |       20 |   0.55     | 0.0307003 | -0.0729242 |       0.420989 |        0.514542 | -0.0808118 |  -0.103321 | -0.318615  |
| ETH/USDT |       13 |   0.538462 | 0.0969656 | -0.106847  |       0.907518 |        1.05877  | -0.125504  |  -0.164368 |  0.0376767 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| stop_loss             |       9 |       1.06011  |     0.817105 |
| time_limit            |       6 |       0.237288 |     0.182895 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.545455 |        0.00389438 |                           0 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|-----------:|-----------:|----------:|
|       20 | -0.318615 |       0.55 | 0.0307003 | -0.0729242 |       0.420989 |        0.514542 |           0.479464 | -0.0185174 | -0.0115619 | -0.0808118 |  -0.103321 | 0.0606711 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| stop_loss             |       5 |       0.479836 |     0.731103 |
| time_limit            |       4 |       0.176482 |     0.268897 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.5 |        0.00308999 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |   cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|----------:|-----------:|----------:|
|       13 | 0.0376767 |   0.538462 | 0.0969656 |  -0.106847 |       0.907518 |         1.05877 |            0.18783 | -0.0219822 | -0.0152454 | -0.125504 |  -0.164368 |  0.234962 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| stop_loss             |       4 |      0.580276  |    0.90515   |
| time_limit            |       2 |      0.0608066 |    0.0948498 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.615385 |         0.0051319 |                           0 |                    0 |                               |                       |                               |
