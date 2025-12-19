# Dry-run profit workflow (evidence → forensics → alignment)

_Last updated: 2025-12-19 00:11 UTC_

## Start stack (dry-run)
- Ensure `.env` present, then: `docker compose up -d trading scheduler ingestion-api redis`
- Trading uses `configs/runtime_overrides/risk_limits_stage_0.yaml` (capital 100, equity-fraction sizing) and `configs/runtime_overrides/deadlock_policy_stage_0.yaml`.

## Extract evidence bundle (real container logs/configs)
```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
python3 -m scripts.extract_container_logs \
  --container algo-data-ingestion-trading-1 \
  --scheduler-container algo-data-ingestion-scheduler-1 \
  --output-dir reports/log_forensics/evidence \
  --timestamp $TS
```
- Outputs under `reports/log_forensics/evidence/$TS/` (audit logs, metrics, contract, risk limits, trigger policy, manifest, env snapshot).

## Trading log forensics
```bash
TS=20251217T015155Z  # or latest evidence timestamp
python3 -m analysis.trading_log_forensics \
  --audit-log reports/log_forensics/evidence/$TS/trading_audit/audit.log \
  --output-dir reports/log_forensics/forensics/$TS \
  --symbols ETH/USDT,BTC/USDT,SOL/USDT
```
- Produces `forensics_summary.md/json` and `per_symbol_trades.csv` with exit/gate histograms, PnL, equity curve.

## Market alignment (trade vs OHLCV)
```bash
python3 -m analysis.market_trade_alignment \
  --trades-csv reports/log_forensics/forensics/$TS/per_symbol_trades.csv \
  --market-data data_lake/market/exchange=binance \
  --output-dir reports/log_forensics/alignment/$TS \
  --window-mins 60
```
- Outputs `alignment_summary.md/json` and `market_alignment.csv` (MFE/MAE, post-exit drift by exit_reason).

## Sizing config (bounded compounding)
- Risk limits: capital/initial_capital_usd=100, sizing_mode=equity_fraction, equity_fraction=0.2, max_equity_fraction=0.3, compounding_step_usd=5.
- Per-symbol base order notionals (scaled with equity): ETH 20, BTC 15, SOL 12; caps: ETH 45, BTC 40, SOL 30; max_total_notional=80.
- Cooldowns: 2 min after exit, 5 min after loss; daily_loss_limit_pct=5%, max_drawdown_pct=20%.

## Evaluate changes
- After a dry-run window, re-run the extractor + forensics + alignment to compare summaries and ensure positive PnL trend with controlled drawdown.
