## Optimization plan (dry-run)

**Evidence (20251217T015155Z forensics + alignment)**
- Portfolio PnL -964 with win rate 12.4%; prob_floor/trailing/gate_close dominate exits.
- Post-exit drift positive across exit reasons: ETH post-exit max ~0.105, BTC ~0.016, SOL ~0.006 while exit returns are negative, implying premature exits.
- Skip reasons dominated by `pnl_block` (risk blocks), and exits fire at prob_floor/time_limit with upside left.

**Changes applied**
1) **Bounded compounding sizing** (risk_limits_stage_0.yaml + trading service): initial_capital=100, equity_fraction sizing with step/ caps; base notionals trimmed (ETH 20, BTC 15, SOL 12). Should reduce blow-up risk and scale with equity growth.
2) **Exit relaxation** (final_trigger_policy.yaml + env TRADING_MODELS): exit_threshold ↓ to 0.46 (0.45 aggressive), exit_prob_drop ↓ to 0.07, min_hold_bars ↑ to 4, max_hold_minutes=90, take_profit 0.0006. Aims to reduce early prob_floor exits and allow follow-through captured in alignment drift.

**Expected impact**
- Fewer premature exits; improved MFE capture; lower turnover from longer holds; risk bounded by lower notional caps and equity fractions.
- Reduced pnl_block frequency as notional shrinks to capital-aligned levels.

**Validation loop**
1. Restart stack to load configs: `docker compose up -d trading scheduler` (ensure volumes mounted).
2. After >=24h or sufficient trades, run extractor + forensics + alignment:
   - `python3 -m scripts.extract_container_logs --container algo-data-ingestion-trading-1 --scheduler-container algo-data-ingestion-scheduler-1 --output-dir reports/log_forensics/evidence`
   - `python3 -m analysis.trading_log_forensics --audit-log reports/log_forensics/evidence/<ts>/trading_audit/audit.log --output-dir reports/log_forensics/forensics/<ts> --symbols ETH/USDT,BTC/USDT,SOL/USDT`
   - `python3 -m analysis.market_trade_alignment --trades-csv reports/log_forensics/forensics/<ts>/per_symbol_trades.csv --market-data data_lake/market/exchange=binance --output-dir reports/log_forensics/alignment/<ts> --window-mins 60`
3. Success criteria: positive PnL trend, reduced prob_floor/time_limit exit share, improved exit_return_pct vs previous baseline, controlled drawdown, no risk violations.

**Evidence (20251218T164538Z forensics + alignment)**
- Trades 1362 (executed 40), PnL -9.1037, exit reasons dominated by turnover_limit (1106) and prob_floor (141); time_limit ~0.22%.
- Alignment shows negative mean exit_return for prob_floor/time_limit/trailing across symbols with positive post-exit drift.
- Risk snapshots show turnover_1d ~408 vs max_turnover_notional 400, indicating turnover cap is binding after a handful of trades.

**Changes applied (2025-12-18 restart)**
1) **Fix trigger override precedence** (`app/trading/service.py`): exit_prob_drop now respects risk limits instead of being overwritten by manifest metadata.
2) **Raise turnover capacity** (`configs/runtime_overrides/risk_limits_stage_0.yaml`): max_turnover_per_day=12, max_orders_per_hour=180 to prevent constant turnover_limit blocks.
3) **Per-symbol exit relaxation** (`configs/runtime_overrides/risk_limits_stage_0.yaml`): ETH exit_threshold=0.43/exit_prob_drop=0.12; BTC 0.45/0.10; SOL 0.44/0.11.
4) **Longer holds** (`docker-compose.yml`, `configs/runtime_overrides/stage_0.yaml`): min_hold_bars_override ETH=6, BTC/SOL=5.

**Validation status**
- Post-restart forensics window (20251218T173921Z) shows 0 trades since 2025-12-18T17:22Z; awaiting gate_pass trades to evaluate PnL/exit_return.
