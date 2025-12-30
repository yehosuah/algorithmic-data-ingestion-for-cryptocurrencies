# Docker Since Changes Forensics (Executed Exits)

- Evidence bundle: `reports/log_forensics/evidence/20251230T005217Z`
- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/log_forensics/evidence/20251230T005217Z/trading_audit/audit.log`
- Window: 2025-12-29T18:00:00+00:00 → 2025-12-30T00:52:17+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |     avg_win |   avg_loss |   payoff_ratio |   profit_factor |    cvar_95 |   max_loss |        pnl |
|:---------|---------:|-----------:|------------:|-----------:|---------------:|----------------:|-----------:|-----------:|-----------:|
| BTC/USDT |       10 |   0.2      |   0.0138546 | -0.0154685 |       0.895666 |        0.223916 | -0.0176021 | -0.0518731 | -0.0960385 |
| ETH/USDT |        9 |   0.333333 |   0.037201  | -0.0475291 |       0.782699 |        0.39135  | -0.052879  | -0.0667829 | -0.173572  |
| SOL/USDT |       10 |   0        | nan         | -0.0218436 |     nan        |        0        | -0.0240513 | -0.0535536 | -0.218436  |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |      23 |      0.604209  |    0.963101  |
| gate_close            |       1 |      0.0231491 |    0.0368994 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.586207 |        0.00217905 |                    0.172414 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |        pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |    p95_loss |     p99_loss |    cvar_95 |   max_loss |   max_win |
|---------:|-----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|------------:|-------------:|-----------:|-----------:|----------:|
|       10 | -0.0960385 |        0.2 | 0.0138546 | -0.0154685 |       0.895666 |        0.223916 |           0.108056 | -0.00147914 | -0.000721895 | -0.0176021 | -0.0518731 | 0.0156914 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       8 |       0.123748 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.4 |        0.00177671 |                         0.2 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |       pnl |   win_rate |   avg_win |   avg_loss |   payoff_ratio |   profit_factor |   max_drawdown_pnl |   p95_loss |   p99_loss |   cvar_95 |   max_loss |   max_win |
|---------:|----------:|-----------:|----------:|-----------:|---------------:|----------------:|-------------------:|-----------:|-----------:|----------:|-----------:|----------:|
|        9 | -0.173572 |   0.333333 |  0.037201 | -0.0475291 |       0.782699 |         0.39135 |           0.173572 | -0.0227775 | -0.0211789 | -0.052879 | -0.0667829 | 0.0619365 |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       6 |       0.285174 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.777778 |        0.00243775 |                    0.111111 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |       pnl |   win_rate | avg_win   |   avg_loss | payoff_ratio   |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |    cvar_95 |   max_loss | max_win   |
|---------:|----------:|-----------:|:----------|-----------:|:---------------|----------------:|-------------------:|------------:|------------:|-----------:|-----------:|:----------|
|       10 | -0.218436 |          0 |           | -0.0218436 |                |               0 |           0.218436 | -0.00233335 | -0.00204645 | -0.0240513 | -0.0535536 |           |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       9 |      0.195287  |     0.894023 |
| gate_close            |       1 |      0.0231491 |     0.105977 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|               0.6 |        0.00234858 |                         0.2 |                    0 |                               |                       |                               |
