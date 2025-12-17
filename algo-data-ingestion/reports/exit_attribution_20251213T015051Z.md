# Exit Attribution

Total exits: 239
Average hold seconds: 1843.49
Overall negative fraction: 0.5816
Churn fraction (<=1 bar): 0.0293

## By reason
| reason             |   count |   avg_pnl |   avg_pnl_gross |   avg_pnl_net |   avg_hold_seconds |   executed_fraction |
|:-------------------|--------:|----------:|----------------:|--------------:|-------------------:|--------------------:|
| gate_close         |     185 | -0.386149 |       -0.385913 |     -0.397647 |            2061.26 |          0.00540541 |
| prob_floor         |      33 | -0.4819   |       -0.4819   |     -0.491945 |             120    |          0          |
| take_profit        |      15 |  0.118162 |        0.118162 |      0.106127 |             364    |          0.4        |
| trailing_prob_drop |       6 | -0.1196   |       -0.1196   |     -0.129588 |            1220    |          0          |

## Reason frequency by symbol
| symbol   | reason             |   count |
|:---------|:-------------------|--------:|
| BTC/USDT | gate_close         |      59 |
| BTC/USDT | prob_floor         |      10 |
| BTC/USDT | take_profit        |       7 |
| ETH/USDT | gate_close         |      56 |
| ETH/USDT | prob_floor         |      11 |
| ETH/USDT | take_profit        |       5 |
| SOL/USDT | gate_close         |      70 |
| SOL/USDT | prob_floor         |      12 |
| SOL/USDT | take_profit        |       3 |
| SOL/USDT | trailing_prob_drop |       6 |

## Negative fraction by reason
| reason             |   negative_fraction |
|:-------------------|--------------------:|
| gate_close         |            0.681081 |
| prob_floor         |            0.030303 |
| take_profit        |            0.4      |
| trailing_prob_drop |            1        |

## Exits by hour
| hour                      | reason             |   count |
|:--------------------------|:-------------------|--------:|
| 2025-12-01T21:00:00+00:00 | gate_close         |       2 |
| 2025-12-01T21:00:00+00:00 | prob_floor         |       4 |
| 2025-12-01T22:00:00+00:00 | gate_close         |      55 |
| 2025-12-01T22:00:00+00:00 | prob_floor         |      27 |
| 2025-12-01T22:00:00+00:00 | take_profit        |       1 |
| 2025-12-01T23:00:00+00:00 | gate_close         |       7 |
| 2025-12-01T23:00:00+00:00 | take_profit        |       6 |
| 2025-12-02T00:00:00+00:00 | gate_close         |      19 |
| 2025-12-02T01:00:00+00:00 | gate_close         |       4 |
| 2025-12-02T01:00:00+00:00 | prob_floor         |       2 |
| 2025-12-02T02:00:00+00:00 | gate_close         |      98 |
| 2025-12-02T02:00:00+00:00 | take_profit        |       8 |
| 2025-12-02T02:00:00+00:00 | trailing_prob_drop |       6 |
