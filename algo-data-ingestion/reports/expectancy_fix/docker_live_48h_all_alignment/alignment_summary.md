# Market alignment
- Trades aligned: 61
- Recommendations: 3

## SOL/USDT
- Aligned trades: 28
- Mean MFE: 0.0036 | Mean MAE: -0.0030 | Mean exit return: 0.0004
- Post-exit drift: max 0.0061 | min -0.0031
- Regret share (MFE >> exit): 0.6429 | Short holds (<5m): 0.0000
- Exit reason effects:
  - stop_loss: exit_ret=-0.0057, post_max=0.0143, post_min=-0.0038
  - time_limit: exit_ret=0.0024, post_max=0.0033, post_min=-0.0028

## ETH/USDT
- Aligned trades: 13
- Mean MFE: 0.0057 | Mean MAE: -0.0021 | Mean exit return: 0.0005
- Post-exit drift: max 0.0056 | min -0.0024
- Regret share (MFE >> exit): 0.6154 | Short holds (<5m): 0.0000
- Exit reason effects:
  - stop_loss: exit_ret=-0.0054, post_max=0.0097, post_min=-0.0009
  - time_limit: exit_ret=0.0032, post_max=0.0037, post_min=-0.0030

## BTC/USDT
- Aligned trades: 20
- Mean MFE: 0.0023 | Mean MAE: -0.0025 | Mean exit return: -0.0007
- Post-exit drift: max 0.0031 | min -0.0020
- Regret share (MFE >> exit): 0.5000 | Short holds (<5m): 0.0000
- Exit reason effects:
  - stop_loss: exit_ret=-0.0053, post_max=0.0032, post_min=-0.0038
  - time_limit: exit_ret=0.0008, post_max=0.0030, post_min=-0.0014

## Recommendations
- SOL/USDT exit_reason=stop_loss shows negative exit return but positive post-exit drift (0.0143); consider relaxing/retiming exits.
- ETH/USDT exit_reason=stop_loss shows negative exit return but positive post-exit drift (0.0097); consider relaxing/retiming exits.
- BTC/USDT exit_reason=stop_loss shows negative exit return but positive post-exit drift (0.0032); consider relaxing/retiming exits.