# Market alignment
- Trades aligned: 136
- Recommendations: 8

## SOL/USDT
- Aligned trades: 87
- Mean MFE: 0.0030 | Mean MAE: -0.0026 | Mean exit return: -0.0000
- Post-exit drift: max 0.0030 | min -0.0025
- Regret share (MFE >> exit): 0.4368 | Short holds (<5m): 0.0000
- Exit reason effects:
  - stop_loss: exit_ret=-0.0027, post_max=0.0010, post_min=-0.0088
  - prob_floor: exit_ret=-0.0008, post_max=0.0026, post_min=-0.0023
  - gate_close: exit_ret=0.0001, post_max=0.0028, post_min=-0.0028
  - take_profit: exit_ret=0.0012, post_max=0.0046, post_min=-0.0021

## BTC/USDT
- Aligned trades: 7
- Mean MFE: 0.0022 | Mean MAE: -0.0006 | Mean exit return: 0.0001
- Post-exit drift: max 0.0027 | min -0.0008
- Regret share (MFE >> exit): 0.4286 | Short holds (<5m): 0.0000
- Exit reason effects:
  - prob_floor: exit_ret=-0.0006, post_max=0.0031, post_min=-0.0009
  - gate_close: exit_ret=0.0006, post_max=0.0023, post_min=-0.0008

## ETH/USDT
- Aligned trades: 42
- Mean MFE: 0.0030 | Mean MAE: -0.0126 | Mean exit return: 0.0001
- Post-exit drift: max 0.0027 | min -0.0074
- Regret share (MFE >> exit): 0.4762 | Short holds (<5m): 0.0000
- Exit reason effects:
  - stop_loss: exit_ret=-0.0054, post_max=0.0022, post_min=-0.2289
  - prob_floor: exit_ret=-0.0044, post_max=0.0018, post_min=-0.0028
  - time_limit: exit_ret=-0.0027, post_max=0.0007, post_min=-0.0024
  - prob_trailing: exit_ret=-0.0012, post_max=0.0023, post_min=-0.0022
  - gate_close: exit_ret=-0.0000, post_max=0.0035, post_min=-0.0010

## Recommendations
- SOL/USDT exit_reason=stop_loss shows negative exit return but positive post-exit drift (0.0010); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0026); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0031); consider relaxing/retiming exits.
- ETH/USDT exit_reason=stop_loss shows negative exit return but positive post-exit drift (0.0022); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0018); consider relaxing/retiming exits.
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0007); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_trailing shows negative exit return but positive post-exit drift (0.0023); consider relaxing/retiming exits.
- ETH/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0035); consider relaxing/retiming exits.