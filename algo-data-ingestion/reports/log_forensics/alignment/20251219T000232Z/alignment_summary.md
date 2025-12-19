# Market alignment
- Trades aligned: 1636
- Recommendations: 12

## ETH/USDT
- Aligned trades: 557
- Mean MFE: 0.0521 | Mean MAE: -0.0562 | Mean exit return: -0.0013
- Post-exit drift: max 0.0628 | min -0.0525
- Exit reason effects:
  - time_limit: exit_ret=-0.0081, post_max=0.0683, post_min=-0.0483
  - prob_floor: exit_ret=-0.0055, post_max=0.0738, post_min=-0.0448
  - trailing_prob_drop: exit_ret=-0.0040, post_max=0.0026, post_min=-0.0479
  - gate_close: exit_ret=-0.0013, post_max=0.0052, post_min=-0.0186
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0042, post_min=-0.0269

## BTC/USDT
- Aligned trades: 475
- Mean MFE: 0.0083 | Mean MAE: -0.1399 | Mean exit return: -0.0023
- Post-exit drift: max 0.0091 | min -0.0814
- Exit reason effects:
  - time_limit: exit_ret=-0.0139, post_max=0.0039, post_min=-0.0041
  - prob_floor: exit_ret=-0.0055, post_max=0.0065, post_min=-0.0775
  - trailing_prob_drop: exit_ret=-0.0030, post_max=0.0061, post_min=-0.2018
  - gate_close: exit_ret=-0.0011, post_max=0.0044, post_min=-0.1715
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0019, post_min=-0.0030

## SOL/USDT
- Aligned trades: 604
- Mean MFE: 0.0030 | Mean MAE: -0.0082 | Mean exit return: -0.0037
- Post-exit drift: max 0.0046 | min -0.0082
- Exit reason effects:
  - time_limit: exit_ret=-0.0280, post_max=0.0055, post_min=-0.0034
  - prob_floor: exit_ret=-0.0095, post_max=0.0040, post_min=-0.0115
  - gate_close: exit_ret=-0.0030, post_max=0.0070, post_min=-0.0091
  - trailing_prob_drop: exit_ret=-0.0028, post_max=0.0013, post_min=-0.0109
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0001, post_min=-0.0058

## Recommendations
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0683); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0738); consider relaxing/retiming exits.
- ETH/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0026); consider relaxing/retiming exits.
- ETH/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0052); consider relaxing/retiming exits.
- BTC/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0039); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0065); consider relaxing/retiming exits.
- BTC/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0061); consider relaxing/retiming exits.
- BTC/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0044); consider relaxing/retiming exits.
- SOL/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0055); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0040); consider relaxing/retiming exits.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0070); consider relaxing/retiming exits.
- SOL/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0013); consider relaxing/retiming exits.