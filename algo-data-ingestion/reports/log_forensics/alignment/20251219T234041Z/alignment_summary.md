# Market alignment
- Trades aligned: 1648
- Recommendations: 11

## ETH/USDT
- Aligned trades: 595
- Mean MFE: 0.2178 | Mean MAE: -0.0676 | Mean exit return: -0.0010
- Post-exit drift: max 0.2183 | min -0.0623
- Exit reason effects:
  - time_limit: exit_ret=-0.0054, post_max=0.1639, post_min=-0.0154
  - prob_floor: exit_ret=-0.0049, post_max=0.1664, post_min=-0.0618
  - trailing_prob_drop: exit_ret=-0.0046, post_max=0.0938, post_min=-0.0391
  - gate_close: exit_ret=-0.0033, post_max=0.0881, post_min=-0.0839
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.3727, post_min=-0.2449

## SOL/USDT
- Aligned trades: 532
- Mean MFE: 0.0065 | Mean MAE: -0.0041 | Mean exit return: -0.0005
- Post-exit drift: max 0.0083 | min -0.0037
- Exit reason effects:
  - time_limit: exit_ret=-0.0031, post_max=0.0000, post_min=-0.0142
  - prob_floor: exit_ret=-0.0026, post_max=0.0114, post_min=-0.0024
  - trailing_prob_drop: exit_ret=-0.0024, post_max=0.0144, post_min=-0.0018
  - gate_close: exit_ret=-0.0008, post_max=0.0110, post_min=-0.0042
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0010, post_min=-0.0075

## BTC/USDT
- Aligned trades: 521
- Mean MFE: 0.0062 | Mean MAE: -0.0340 | Mean exit return: -0.0010
- Post-exit drift: max 0.0079 | min -0.0324
- Exit reason effects:
  - prob_floor: exit_ret=-0.0028, post_max=0.0060, post_min=-0.0035
  - trailing_prob_drop: exit_ret=-0.0021, post_max=0.0068, post_min=-0.0040
  - time_limit: exit_ret=-0.0014, post_max=0.0085, post_min=-0.0090
  - prob_trailing: exit_ret=-0.0010, post_max=0.0194, post_min=-0.0042
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0172, post_min=-0.0039

## Recommendations
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.1639); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.1664); consider relaxing/retiming exits.
- ETH/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0938); consider relaxing/retiming exits.
- ETH/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0881); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0114); consider relaxing/retiming exits.
- SOL/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0144); consider relaxing/retiming exits.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0110); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0060); consider relaxing/retiming exits.
- BTC/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0068); consider relaxing/retiming exits.
- BTC/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0085); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_trailing shows negative exit return but positive post-exit drift (0.0194); consider relaxing/retiming exits.