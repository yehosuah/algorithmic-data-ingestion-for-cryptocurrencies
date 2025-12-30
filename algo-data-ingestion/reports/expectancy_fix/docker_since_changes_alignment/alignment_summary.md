# Market alignment
- Trades aligned: 29
- Recommendations: 3

## ETH/USDT
- Aligned trades: 9
- Mean MFE: 0.0019 | Mean MAE: -0.0033 | Mean exit return: -0.0005
- Post-exit drift: max 0.0022 | min -0.0028
- Regret share (MFE >> exit): 0.7778 | Short holds (<5m): 0.1111
- Exit reason effects:
  - prob_floor: exit_ret=-0.0012, post_max=0.0025, post_min=-0.0028
  - gate_close: exit_ret=0.0020, post_max=0.0014, post_min=-0.0027

## SOL/USDT
- Aligned trades: 10
- Mean MFE: 0.0013 | Mean MAE: -0.0036 | Mean exit return: -0.0010
- Post-exit drift: max 0.0021 | min -0.0026
- Regret share (MFE >> exit): 0.6000 | Short holds (<5m): 0.2000
- Exit reason effects:
  - prob_floor: exit_ret=-0.0012, post_max=0.0020, post_min=-0.0026
  - gate_close: exit_ret=0.0003, post_max=0.0022, post_min=-0.0032

## BTC/USDT
- Aligned trades: 10
- Mean MFE: 0.0012 | Mean MAE: -0.0029 | Mean exit return: -0.0006
- Post-exit drift: max 0.0013 | min -0.0026
- Regret share (MFE >> exit): 0.4000 | Short holds (<5m): 0.2000
- Exit reason effects:
  - prob_floor: exit_ret=-0.0010, post_max=0.0013, post_min=-0.0027
  - gate_close: exit_ret=0.0013, post_max=0.0011, post_min=-0.0024

## Recommendations
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0025); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0020); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0013); consider relaxing/retiming exits.