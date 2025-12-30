# Market alignment
- Trades aligned: 1911
- Recommendations: 14

## SOL/USDT
- Aligned trades: 803
- Mean MFE: 0.0023 | Mean MAE: -0.0083 | Mean exit return: -0.0044
- Post-exit drift: max 0.0044 | min -0.0045
- Exit reason effects:
  - time_limit: exit_ret=-0.0084, post_max=0.0061, post_min=-0.0026
  - prob_floor: exit_ret=-0.0070, post_max=0.0042, post_min=-0.0043
  - trailing_prob_drop: exit_ret=-0.0035, post_max=0.0047, post_min=-0.0036
  - gate_close: exit_ret=-0.0034, post_max=0.0048, post_min=-0.0045
  - prob_trailing: exit_ret=-0.0008, post_max=0.0075, post_min=-0.0052

## BTC/USDT
- Aligned trades: 640
- Mean MFE: 0.0788 | Mean MAE: -0.0513 | Mean exit return: -0.0018
- Post-exit drift: max 0.0543 | min -0.0711
- Exit reason effects:
  - prob_floor: exit_ret=-0.0033, post_max=0.0525, post_min=-0.0619
  - time_limit: exit_ret=-0.0025, post_max=0.0454, post_min=-0.1571
  - trailing_prob_drop: exit_ret=-0.0014, post_max=0.0359, post_min=-0.0805
  - gate_close: exit_ret=-0.0006, post_max=0.0681, post_min=-0.1211
  - prob_trailing: exit_ret=-0.0004, post_max=0.0850, post_min=-0.0047

## ETH/USDT
- Aligned trades: 468
- Mean MFE: 0.1686 | Mean MAE: -0.1186 | Mean exit return: -0.0040
- Post-exit drift: max 0.1501 | min -0.0684
- Exit reason effects:
  - gate_close: exit_ret=-0.0069, post_max=0.2350, post_min=-0.0741
  - time_limit: exit_ret=-0.0062, post_max=0.1021, post_min=-0.1174
  - prob_floor: exit_ret=-0.0053, post_max=0.1612, post_min=-0.0808
  - trailing_prob_drop: exit_ret=-0.0037, post_max=0.1199, post_min=-0.0374
  - cooldown_after_exit: exit_ret=0.0000, post_max=0.1152, post_min=-0.0465

## Recommendations
- SOL/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0061); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0042); consider relaxing/retiming exits.
- SOL/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0047); consider relaxing/retiming exits.
- SOL/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0048); consider relaxing/retiming exits.
- SOL/USDT exit_reason=prob_trailing shows negative exit return but positive post-exit drift (0.0075); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.0525); consider relaxing/retiming exits.
- BTC/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.0454); consider relaxing/retiming exits.
- BTC/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.0359); consider relaxing/retiming exits.
- BTC/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.0681); consider relaxing/retiming exits.
- BTC/USDT exit_reason=prob_trailing shows negative exit return but positive post-exit drift (0.0850); consider relaxing/retiming exits.
- ETH/USDT exit_reason=gate_close shows negative exit return but positive post-exit drift (0.2350); consider relaxing/retiming exits.
- ETH/USDT exit_reason=time_limit shows negative exit return but positive post-exit drift (0.1021); consider relaxing/retiming exits.
- ETH/USDT exit_reason=prob_floor shows negative exit return but positive post-exit drift (0.1612); consider relaxing/retiming exits.
- ETH/USDT exit_reason=trailing_prob_drop shows negative exit return but positive post-exit drift (0.1199); consider relaxing/retiming exits.