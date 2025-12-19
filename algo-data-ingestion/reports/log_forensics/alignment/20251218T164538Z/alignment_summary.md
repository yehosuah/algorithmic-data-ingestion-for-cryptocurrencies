# Market alignment
- Trades aligned: 1362
- Recommendations: 10

## BTC/USDT
- Aligned trades: 327
- Mean MFE: 0.0110 | Mean MAE: -0.0046 | Mean exit return: -0.0004
- Post-exit drift: max 0.0110 | min -0.0047
- Exit reason effects:
  - time_limit: exit_ret=-0.0049, post_max=0.0017, post_min=-0.0015
  - prob_floor: exit_ret=-0.0025, post_max=0.0017, post_min=-0.0030
  - trailing_prob_drop: exit_ret=-0.0002, post_max=0.0023, post_min=-0.0025
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0026, post_min=-0.0012
  - cooldown_after_loss: exit_ret=0.0000, post_max=0.0030, post_min=-0.0032

## ETH/USDT
- Aligned trades: 557
- Mean MFE: 0.0683 | Mean MAE: -0.0508 | Mean exit return: -0.0003
- Post-exit drift: max 0.0828 | min -0.0522
- Exit reason effects:
  - time_limit: exit_ret=-0.0028, post_max=0.2011, post_min=-0.0677
  - prob_floor: exit_ret=-0.0028, post_max=0.1813, post_min=-0.0278
  - trailing_prob_drop: exit_ret=-0.0014, post_max=0.0025, post_min=-0.0052
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.0012, post_min=-0.0056
  - cooldown_after_loss: exit_ret=0.0000, post_max=0.2016, post_min=-0.0673

## SOL/USDT
- Aligned trades: 478
- Mean MFE: 0.0040 | Mean MAE: -0.0037 | Mean exit return: -0.0009
- Post-exit drift: max 0.0042 | min -0.0044
- Exit reason effects:
  - time_limit: exit_ret=-0.0107, post_max=0.0029, post_min=-0.0027
  - prob_floor: exit_ret=-0.0046, post_max=0.0021, post_min=-0.0056
  - trailing_prob_drop: exit_ret=-0.0016, post_max=0.0006, post_min=-0.0094
  - gate_close: exit_ret=-0.0012, post_max=0.0029, post_min=-0.0047
  - dry_run: exit_ret=0.0000, post_max=0.0031, post_min=-0.0024

## Recommendations
- BTC/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0017); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0017); consider relaxing/retiming exits.
- BTC/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0023); consider relaxing/retiming exits.
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.2011); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.1813); consider relaxing/retiming exits.
- ETH/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0025); consider relaxing/retiming exits.
- SOL/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0029); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0021); consider relaxing/retiming exits.
- SOL/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0006); consider relaxing/retiming exits.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0029); consider relaxing/retiming exits.