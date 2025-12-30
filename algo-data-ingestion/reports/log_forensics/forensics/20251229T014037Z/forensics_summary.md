# Trading log forensics

- Symbols: BTC/USDT, ETH/USDT, SOL/USDT
- Trades: 2095 (executed: 900)
- PnL: -3.896627 | Win rate: 20.62% | Profit factor: 0.46873902030766673 | Payoff: 1.0774487202905394
- Avg win: 0.007958446298653225 | Avg loss: -0.007386380575501718 | P95 loss: -0.0003716851388463808 | CVaR95: -0.007764594161338454
- Max drawdown (pnl units): 4.382190330426711 | Max loss: -0.0675959640439533 | Max win: 0.1018330600000003

## BTC/USDT
- Trades: 861 (executed: 278)
- PnL: -1.690006 | Win rate: 18.23% | Profit factor: 0.3756201829914694 | Payoff: 1.169925283330118
- Avg win: 0.0064757291062577885 | Avg loss: -0.005535164679769154 | P95 loss: -0.00034029562307460613 | CVaR95: -0.005821144202702198
- Median hold (min): 5.0
- Top exit reasons: prob_floor (514), dry_run (139), gate_close (132), cooldown_after_loss (60), cooldown_after_exit (14)
- Top skip reasons: min_hold (507), dry_run (278), cooldown_after_loss (60), cooldown_after_exit (14), spread_threshold (1)

## ETH/USDT
- Trades: 337 (executed: 240)
- PnL: -0.170469 | Win rate: 18.40% | Profit factor: 0.5898111327556003 | Payoff: 1.1891353482975813
- Avg win: 0.0039535003426377545 | Avg loss: -0.003324684905126873 | P95 loss: -0.00023844719999999647 | CVaR95: -0.0035139768988531046
- Median hold (min): 10.0
- Top exit reasons: prob_floor (164), dry_run (120), cooldown_after_loss (26), gate_close (21), cooldown_after_exit (3)
- Top skip reasons: dry_run (240), min_hold (66), cooldown_after_loss (26), cooldown_after_exit (3), spread_threshold (1)

## SOL/USDT
- Trades: 897 (executed: 382)
- PnL: -2.036152 | Win rate: 23.75% | Profit factor: 0.5166282907835837 | Payoff: 0.9192587897041232
- Avg win: 0.010217100047381125 | Avg loss: -0.011114498073681348 | P95 loss: -0.00039485754881172174 | CVaR95: -0.011682035052616165
- Median hold (min): 6.0
- Top exit reasons: gate_close (314), prob_floor (274), dry_run (191), cooldown_after_loss (76), cooldown_after_exit (37)
- Top skip reasons: min_hold (401), dry_run (382), cooldown_after_loss (76), cooldown_after_exit (37), spread_threshold (1)
