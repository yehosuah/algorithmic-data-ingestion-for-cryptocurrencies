# Market alignment
- Trades aligned: 3
- Recommendations: 2

## BTC/USDT
- Aligned trades: 1
- Mean MFE: 0.0012 | Mean MAE: -0.0005 | Mean exit return: -0.0000
- Post-exit drift: max 0.0010 | min -0.0005
- Regret share (MFE >> exit): 0.0000 | Short holds (<5m): 0.0000
- Exit reason effects:
  - prob_floor: exit_ret=-0.0000, post_max=0.0010, post_min=-0.0005

## SOL/USDT
- Aligned trades: 1
- Mean MFE: 0.0028 | Mean MAE: -0.0002 | Mean exit return: -0.0001
- Post-exit drift: max 0.0029 | min 0.0000
- Regret share (MFE >> exit): 1.0000 | Short holds (<5m): 0.0000
- Exit reason effects:
  - prob_floor: exit_ret=-0.0001, post_max=0.0029, post_min=0.0000

## ETH/USDT
- Aligned trades: 1
- Mean MFE: 0.0026 | Mean MAE: -0.0005 | Mean exit return: 0.0004
- Post-exit drift: max 0.0022 | min -0.0002
- Regret share (MFE >> exit): 1.0000 | Short holds (<5m): 0.0000
- Exit reason effects:
  - prob_floor: exit_ret=0.0004, post_max=0.0022, post_min=-0.0002

## Recommendations
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0010); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0029); consider relaxing/retiming exits.