# Week-Long Dry Run Checklist

_Last updated: 2025-11-05 14:56 UTC_

Use this checklist to track the current paper-trading exercise from start to finish. Update status daily (☐ → ☑) and capture notes in the right-hand column.

| Status | Item | Notes |
| --- | --- | --- |
| ☐ | Confirm Docker stack healthy (`scheduler`, `trading`, `ingestion-api`, `redis`, `prometheus`, `grafana`) |  |
| ☐ | Verify scheduler warm-started data (ingest + backfill rows > 0 for BTC/ETH/SOL) |  |
| ☐ | Validate gate coverage per symbol/model ≥ expected baseline (Base ≈100%, TCN ≥40%) |  |
| ☐ | Confirm decision queue is draining (Redis `trading:decisions` near-empty after initial burst) |  |
| ☐ | Check trading metrics endpoint (`http://localhost:9010/metrics`) for non-zero `trading_trade_attempts_total` and `trading_trade_notional_total` |  |
| ☐ | Ensure Redis state (`trading:positions`) reflects current open/closed positions each day |  |
| ☐ | Log realized P&L trend daily (even if zero) |  |
| ☐ | Capture Grafana screenshots: Gate Coverage, Order Fills, Turnover, P&L (start/mid/end of run) |  |
| ☐ | Review trading logs for dry-run order reasons (spread guard, invalid amount, etc.) |  |
| ☐ | Validate Prometheus alert rules remain green throughout run |  |
| ☐ | Daily check of CCXT sandbox credentials / rate limits |  |
| ☐ | Summarize notable market conditions impacting signals |  |
| ☐ | End-of-week retrospective: coverage stats, trade counts, faux P&L vs expectations |  |
| ☐ | Identify any blockers before switching to live trading (ops, compliance, risk) |  |

## Daily Log Template

Use the following headings for each day:

```
### Day N (YYYY-MM-DD)
- Stack health:
- Gate coverage snapshot:
- Trades executed (count/notional):
- Issues observed:
- Mitigations / follow-ups:
```

## Exit Criteria

- No missed inference runs (scheduler log free of persistent errors).
- Continuous trading metrics flow (no multi-hour flatlines).
- Redis state consistent with expected positions.
- Checklist items above marked complete.
