# Market alignment
- Trades aligned: 4513
- Recommendations: 18

## ETH/USDT
- Aligned trades: 1294
- Mean MFE: 0.1277 | Mean MAE: -0.0574 | Mean exit return: -0.0045
- Post-exit drift: max 0.1049 | min -0.0495
- Exit reason effects:
  - time_limit: exit_ret=-0.0090, post_max=0.0890, post_min=-0.0621
  - prob_floor: exit_ret=-0.0065, post_max=0.0875, post_min=-0.0635
  - trailing_prob_drop: exit_ret=-0.0031, post_max=0.1239, post_min=-0.0223
  - gate_close: exit_ret=-0.0027, post_max=0.1109, post_min=-0.0409
  - prob_trailing: exit_ret=0.0005, post_max=0.2309, post_min=-0.0290

## BTC/USDT
- Aligned trades: 1324
- Mean MFE: 0.0115 | Mean MAE: -0.1331 | Mean exit return: -0.0024
- Post-exit drift: max 0.0135 | min -0.1042
- Exit reason effects:
  - time_limit: exit_ret=-0.0038, post_max=0.0071, post_min=-0.0858
  - prob_floor: exit_ret=-0.0035, post_max=0.0141, post_min=-0.1074
  - trailing_prob_drop: exit_ret=-0.0018, post_max=0.0085, post_min=-0.0892
  - gate_close: exit_ret=-0.0013, post_max=0.0163, post_min=-0.2107
  - prob_trailing: exit_ret=0.0000, post_max=0.0009, post_min=-0.0081

## SOL/USDT
- Aligned trades: 1895
- Mean MFE: 0.0028 | Mean MAE: -0.0100 | Mean exit return: -0.0047
- Post-exit drift: max 0.0059 | min -0.0063
- Exit reason effects:
  - time_limit: exit_ret=-0.0086, post_max=0.0054, post_min=-0.0050
  - prob_floor: exit_ret=-0.0071, post_max=0.0060, post_min=-0.0069
  - trailing_prob_drop: exit_ret=-0.0031, post_max=0.0058, post_min=-0.0054
  - gate_close: exit_ret=-0.0029, post_max=0.0060, post_min=-0.0067
  - take_profit: exit_ret=0.0003, post_max=0.0061, post_min=-0.0050

## Recommendations
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0890); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0875); consider relaxing/retiming exits.
- ETH/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.1239); consider relaxing/retiming exits.
- ETH/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.1109); consider relaxing/retiming exits.
- ETH/USDT exit_reason=dry_run insufficient data for regret; review exit logic.
- ETH/USDT exit_reason=spread_too_wide insufficient data for regret; review exit logic.
- BTC/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0071); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0141); consider relaxing/retiming exits.
- BTC/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0085); consider relaxing/retiming exits.
- BTC/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0163); consider relaxing/retiming exits.
- BTC/USDT exit_reason=dry_run insufficient data for regret; review exit logic.
- BTC/USDT exit_reason=spread_threshold insufficient data for regret; review exit logic.
- BTC/USDT exit_reason=spread_too_wide insufficient data for regret; review exit logic.
- SOL/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0054); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0060); consider relaxing/retiming exits.
- SOL/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0058); consider relaxing/retiming exits.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0060); consider relaxing/retiming exits.
- SOL/USDT exit_reason=dry_run insufficient data for regret; review exit logic.