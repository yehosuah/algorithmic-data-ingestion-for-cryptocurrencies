# Trading log forensics

- Symbols: BTC/USDT, ETH/USDT, SOL/USDT
- Trades: 757 (executed: 271)
- PnL: -0.027726 | Win rate: 14.80% | Profit factor: 0.9822978598855018 | Payoff: 1.2015607750385155
- Avg win: 0.01373673174441628 | Avg loss: -0.011432406940860694 | P95 loss: -0.0009014329263701387 | CVaR95: -0.012026557891863836
- Max drawdown (pnl units): 0.5456356990340995 | Max loss: -0.09607562400000044 | Max win: 0.09925593622140406

## BTC/USDT
- Trades: 271 (executed: 13)
- PnL: -0.028585 | Win rate: 0.74% | Profit factor: 0.2633614367240057 | Payoff: 1.0534457468960228
- Avg win: 0.005109810100000932 | Avg loss: -0.004850567876947612 | P95 loss: -0.002342840916764226 | CVaR95: -0.005239781339087465
- Median hold (min): 7.5
- Top exit reasons: risk_clip_to_zero (228), cooldown_after_loss (27), dry_run (6), gate_close (6), prob_floor (4)
- Top skip reasons: risk_clip_to_zero (228), cooldown_after_loss (27), dry_run (13), min_hold (3)

## ETH/USDT
- Trades: 132 (executed: 84)
- PnL: 0.357689 | Win rate: 28.03% | Profit factor: 1.5739154980318797 | Payoff: 0.7656886206641577
- Avg win: 0.02651167406073556 | Avg loss: -0.03462461547063264 | P95 loss: -0.008138096400000436 | CVaR95: -0.03632098320419939
- Median hold (min): 10.0
- Top exit reasons: dry_run (42), take_profit (34), cooldown_after_loss (18), cooldown_after_exit (17), prob_trailing (9)
- Top skip reasons: dry_run (84), cooldown_after_loss (18), cooldown_after_exit (17), min_hold (13)

## SOL/USDT
- Trades: 354 (executed: 174)
- PnL: -0.356830 | Win rate: 20.62% | Profit factor: 0.6053607160725848 | Payoff: 0.9204799929322866
- Avg win: 0.007498114999005562 | Avg loss: -0.008145875039738258 | P95 loss: -0.00048207999999948833 | CVaR95: -0.00859030194719362
- Median hold (min): 4.0
- Top exit reasons: gate_close (97), dry_run (87), prob_floor (53), cooldown_after_loss (51), take_profit (33)
- Top skip reasons: dry_run (174), min_hold (97), cooldown_after_loss (51), cooldown_after_exit (32)
