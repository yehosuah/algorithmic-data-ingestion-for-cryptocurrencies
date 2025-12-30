# Run 20251229 21H Forensics (Executed Exits)

- Audit log: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/data_lake/trading/audit.log`
- Window: 2025-12-29T21:00:00+00:00 → 2025-12-29T21:36:00+00:00
- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)

## Headline
| symbol   |   trades |   win_rate |      avg_win |     avg_loss | payoff_ratio   |   profit_factor |      cvar_95 |     max_loss |         pnl |
|:---------|---------:|-----------:|-------------:|-------------:|:---------------|----------------:|-------------:|-------------:|------------:|
| BTC/USDT |        1 |          0 | nan          |  -0.00524664 |                |               0 |  -0.00524664 |  -0.00524664 | -0.00524664 |
| ETH/USDT |        1 |          1 |   0.00493658 | nan          |                |             nan | nan          | nan          |  0.00493658 |
| SOL/USDT |        1 |          0 | nan          |  -0.00277168 |                |               0 |  -0.00277168 |  -0.00277168 | -0.00277168 |

## Portfolio loss drivers (by exit_reason_primary)
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       2 |     0.00801832 |            1 |

## Upside starvation
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|          0.666667 |        0.00211342 |                           0 |                    0 |                               |                       |                               |

## BTC/USDT

**Metrics**
|   trades |         pnl |   win_rate | avg_win   |    avg_loss | payoff_ratio   |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |     cvar_95 |    max_loss | max_win   |
|---------:|------------:|-----------:|:----------|------------:|:---------------|----------------:|-------------------:|------------:|------------:|------------:|------------:|:----------|
|        1 | -0.00524664 |          0 |           | -0.00524664 |                |               0 |         0.00524664 | -0.00524664 | -0.00524664 | -0.00524664 | -0.00524664 |           |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |     0.00524664 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 0 |        0.00119148 |                           0 |                    0 |                               |                       |                               |

## ETH/USDT

**Metrics**
|   trades |        pnl |   win_rate |    avg_win | avg_loss   | payoff_ratio   | profit_factor   |   max_drawdown_pnl | p95_loss   | p99_loss   | cvar_95   | max_loss   |    max_win |
|---------:|-----------:|-----------:|-----------:|:-----------|:---------------|:----------------|-------------------:|:-----------|:-----------|:----------|:-----------|-----------:|
|        1 | 0.00493658 |          1 | 0.00493658 |            |                |                 |                  0 |            |            |           |            | 0.00493658 |

**Loss drivers**
_(none)_

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 1 |        0.00222741 |                           0 |                    0 |                               |                       |                               |

## SOL/USDT

**Metrics**
|   trades |         pnl |   win_rate | avg_win   |    avg_loss | payoff_ratio   |   profit_factor |   max_drawdown_pnl |    p95_loss |    p99_loss |     cvar_95 |    max_loss | max_win   |
|---------:|------------:|-----------:|:----------|------------:|:---------------|----------------:|-------------------:|------------:|------------:|------------:|------------:|:----------|
|        1 | -0.00277168 |          0 |           | -0.00277168 |                |               0 |         0.00277168 | -0.00277168 | -0.00277168 | -0.00277168 | -0.00277168 |           |

**Loss drivers**
| exit_reason_primary   |   count |   loss_sum_abs |   loss_share |
|:----------------------|--------:|---------------:|-------------:|
| prob_floor            |       1 |     0.00277168 |            1 |

**Upside starvation**
|   regret_fraction |   mean_regret_gap |   short_hold_fraction_lt_5m |   take_profit_trades | take_profit_avg_exit_return   | take_profit_avg_mfe   | take_profit_regret_fraction   |
|------------------:|------------------:|----------------------------:|---------------------:|:------------------------------|:----------------------|:------------------------------|
|                 1 |        0.00292137 |                           0 |                    0 |                               |                       |                               |
