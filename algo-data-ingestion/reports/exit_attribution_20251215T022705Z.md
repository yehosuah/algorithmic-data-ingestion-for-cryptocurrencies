# Exit Attribution

Total exits: 31
Average hold seconds: 284.52
Overall negative fraction: 0.4194
Executed exits: 10
Executed negative fraction: 0.0000
Churn fraction (<=1 bar): 0.2258

## By reason
| reason             |   count |    avg_pnl |   avg_pnl_gross |   avg_pnl_net |   avg_hold_seconds |   executed_fraction |
|:-------------------|--------:|-----------:|----------------:|--------------:|-------------------:|--------------------:|
| gate_close         |      11 | -0.0748487 |      -0.0594568 |    -0.0748487 |            316.364 |           0.0909091 |
| prob_floor         |       1 | -0.0168723 |      -0.006     |    -0.0168723 |             60     |           0         |
| take_profit        |      17 |  0.0399521 |       0.0508941 |     0.0399521 |            236.471 |           0.529412  |
| trailing_prob_drop |       2 | -0.190432  |      -0.139     |    -0.190432  |            630     |           0         |

## Executed exits by reason
| reason      |   count |   avg_pnl |   avg_pnl_gross |   avg_pnl_net |   avg_hold_seconds |   negative_fraction |
|:------------|--------:|----------:|----------------:|--------------:|-------------------:|--------------------:|
| gate_close  |       1 | 0.001083  |       0.009     |     0.001083  |            180     |                   0 |
| take_profit |       9 | 0.0510424 |       0.0619822 |     0.0510424 |            373.333 |                   0 |

## Reason frequency by symbol
| symbol   | reason             |   count |
|:---------|:-------------------|--------:|
| BTC/USDT | gate_close         |       2 |
| BTC/USDT | take_profit        |       9 |
| ETH/USDT | gate_close         |       4 |
| ETH/USDT | take_profit        |       5 |
| ETH/USDT | trailing_prob_drop |       2 |
| SOL/USDT | gate_close         |       5 |
| SOL/USDT | prob_floor         |       1 |
| SOL/USDT | take_profit        |       3 |

## Negative fraction by reason
| reason             |   negative_fraction |
|:-------------------|--------------------:|
| gate_close         |            0.909091 |
| prob_floor         |            1        |
| take_profit        |            0        |
| trailing_prob_drop |            1        |

## Exits by hour
| hour                      | reason             |   count |
|:--------------------------|:-------------------|--------:|
| 2025-12-15T01:00:00+00:00 | gate_close         |       3 |
| 2025-12-15T01:00:00+00:00 | prob_floor         |       1 |
| 2025-12-15T02:00:00+00:00 | gate_close         |       8 |
| 2025-12-15T02:00:00+00:00 | take_profit        |      17 |
| 2025-12-15T02:00:00+00:00 | trailing_prob_drop |       2 |
