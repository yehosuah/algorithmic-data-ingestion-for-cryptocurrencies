# Market alignment
- Trades aligned: 204
- Recommendations: 8

## ETH/USDT
- Aligned trades: 72
- Mean MFE: 0.1991 | Mean MAE: -0.0044 | Mean exit return: 0.0007
- Post-exit drift: max 0.0190 | min -0.0039
- Exit reason effects:
  - take_profit: exit_ret=0.0007, post_max=0.0009, post_min=-0.0046
  - dry_run: exit_ret=n/a, post_max=0.1991, post_min=-0.0044
  - turnover_limit: exit_ret=n/a, post_max=0.0167, post_min=-0.0039

## BTC/USDT
- Aligned trades: 56
- Mean MFE: 0.0020 | Mean MAE: -0.0019 | Mean exit return: -0.0007
- Post-exit drift: max 0.0047 | min -0.0072
- Exit reason effects:
  - prob_floor: exit_ret=-0.0010, post_max=0.0029, post_min=-0.0043
  - take_profit: exit_ret=0.0003, post_max=0.0017, post_min=-0.0182
  - dry_run: exit_ret=n/a, post_max=0.0020, post_min=-0.0019
  - turnover_limit: exit_ret=n/a, post_max=0.0050, post_min=-0.0073

## SOL/USDT
- Aligned trades: 76
- Mean MFE: 0.0036 | Mean MAE: -0.0023 | Mean exit return: 0.0001
- Post-exit drift: max 0.0025 | min -0.0033
- Exit reason effects:
  - gate_close: exit_ret=-0.0002, post_max=0.0039, post_min=-0.0020
  - take_profit: exit_ret=0.0004, post_max=0.0032, post_min=-0.0028
  - dry_run: exit_ret=n/a, post_max=0.0036, post_min=-0.0022
  - turnover_limit: exit_ret=n/a, post_max=0.0023, post_min=-0.0034

## Recommendations
- ETH/USDT exit_reason=dry_run insufficient data for regret; review exit logic.
- ETH/USDT exit_reason=turnover_limit insufficient data for regret; review exit logic.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0029); consider relaxing/retiming exits.
- BTC/USDT exit_reason=dry_run insufficient data for regret; review exit logic.
- BTC/USDT exit_reason=turnover_limit insufficient data for regret; review exit logic.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0039); consider relaxing/retiming exits.
- SOL/USDT exit_reason=dry_run insufficient data for regret; review exit logic.
- SOL/USDT exit_reason=turnover_limit insufficient data for regret; review exit logic.