# Expectancy Fix Sweep Summary
Dataset: /private/tmp/features_14d_base_xgb_h120_calmon_spread0.parquet
Rows: 58149

## Best config (ranked by pnl_min_half then pnl_total)
|   btc_thr |   eth_thr |   sol_thr |   btc_hold |   eth_hold |   sol_hold |   pnl_total |   pnl_first_half |   pnl_second_half |   pnl_min_half |   median_daily_pnl |   pos_day_fraction |   trades_total |   stop_losses_total |   time_exits_total |
|----------:|----------:|----------:|-----------:|-----------:|-----------:|------------:|-----------------:|------------------:|---------------:|-------------------:|-------------------:|---------------:|--------------------:|-------------------:|
|      0.64 |       0.7 |      0.78 |         90 |        240 |         90 |       4.899 |            3.197 |             1.702 |          1.702 |          0.0393324 |                0.5 |            433 |                 111 |                322 |

Results: experiments/expectancy_fix/sweeps/results.csv