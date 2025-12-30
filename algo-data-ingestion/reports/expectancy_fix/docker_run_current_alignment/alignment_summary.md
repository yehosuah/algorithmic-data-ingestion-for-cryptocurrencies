# Market alignment
- Trades aligned: 120
- Recommendations: 3

## SOL/USDT
- Aligned trades: 43
- Mean MFE: 0.0024 | Mean MAE: -0.0148 | Mean exit return: -0.0003
- Post-exit drift: max 0.0025 | min -0.0199
- Regret share (MFE >> exit): 0.6047 | Short holds (<5m): 0.4419
- Exit reason effects:
  - prob_floor: exit_ret=-0.0016, post_max=0.0023, post_min=-0.0379
  - gate_close: exit_ret=0.0006, post_max=0.0023, post_min=-0.0042
  - take_profit: exit_ret=0.0021, post_max=0.0050, post_min=-0.0037

## BTC/USDT
- Aligned trades: 39
- Mean MFE: 0.0045 | Mean MAE: -0.0056 | Mean exit return: -0.0005
- Post-exit drift: max 0.0047 | min -0.0053
- Regret share (MFE >> exit): 0.7179 | Short holds (<5m): 0.3846
- Exit reason effects:
  - prob_floor: exit_ret=-0.0010, post_max=0.0029, post_min=-0.0046
  - gate_close: exit_ret=0.0001, post_max=0.0071, post_min=-0.0065
  - prob_trailing: exit_ret=0.0002, post_max=0.0027, post_min=-0.0014

## ETH/USDT
- Aligned trades: 38
- Mean MFE: 0.0203 | Mean MAE: -0.0261 | Mean exit return: -0.0001
- Post-exit drift: max 0.0224 | min -0.0283
- Regret share (MFE >> exit): 0.7632 | Short holds (<5m): 0.1316
- Exit reason effects:
  - prob_floor: exit_ret=-0.0016, post_max=0.0228, post_min=-0.0167
  - gate_close: exit_ret=0.0005, post_max=0.0176, post_min=-0.0327
  - take_profit: exit_ret=0.0039, post_max=0.0253, post_min=-0.0577

## Recommendations
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0023); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0029); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0228); consider relaxing/retiming exits.