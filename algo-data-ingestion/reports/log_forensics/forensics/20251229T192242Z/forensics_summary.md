# Trading log forensics

- Symbols: BTC/USDT, ETH/USDT, SOL/USDT
- Trades: 20127 (executed: 3639)
- PnL: -1320.420263 | Win rate: 10.28% | Profit factor: 0.032273770963176435 | Payoff: 0.19102330523711966
- Avg win: 0.02127350428579264 | Avg loss: -0.11136601504924004 | P95 loss: -0.002784842128968277 | CVaR95: -0.11715714799115923
- Max drawdown (pnl units): 1320.420262511701 | Max loss: -15.218688054710361 | Max win: 0.46639918799999114

## BTC/USDT
- Trades: 6205 (executed: 1022)
- PnL: -206.859666 | Win rate: 9.41% | Profit factor: 0.03363487203042896 | Payoff: 0.2277265137128701
- Avg win: 0.0123285358191885 | Avg loss: -0.05413746347837646 | P95 loss: -0.001751379800038177 | CVaR95: -0.05694391817860091
- Median hold (min): 18.0
- Top exit reasons: prob_floor (2782), gate_close (720), trailing_prob_drop (693), turnover_limit (640), dry_run (505)
- Top skip reasons: pnl_block (3132), dry_run (1008), min_hold (902), turnover_limit (640), risk_clip_to_zero (242)

## ETH/USDT
- Trades: 5404 (executed: 1128)
- PnL: -627.739610 | Win rate: 9.68% | Profit factor: 0.031068335578843937 | Payoff: 0.1687075966422883
- Avg win: 0.038485991789221345 | Avg loss: -0.22812245894785294 | P95 loss: -0.0037768357875391126 | CVaR95: -0.2400369017124905
- Median hold (min): 24.0
- Top exit reasons: prob_floor (1901), turnover_limit (1211), trailing_prob_drop (701), dry_run (548), take_profit (420)
- Top skip reasons: pnl_block (2499), turnover_limit (1211), dry_run (1093), min_hold (300), cooldown_after_loss (148)

## SOL/USDT
- Trades: 8518 (executed: 1489)
- PnL: -485.820987 | Win rate: 11.31% | Profit factor: 0.03324805502479722 | Payoff: 0.18844017063898572
- Avg win: 0.01735006775433221 | Avg loss: -0.09207202315461456 | P95 loss: -0.003959600000000004 | CVaR95: -0.0968084922328105
- Median hold (min): 18.0
- Top exit reasons: prob_floor (3483), gate_close (1666), turnover_limit (948), dry_run (745), trailing_prob_drop (675)
- Top skip reasons: pnl_block (4534), dry_run (1489), min_hold (1143), turnover_limit (948), cooldown_after_loss (249)
