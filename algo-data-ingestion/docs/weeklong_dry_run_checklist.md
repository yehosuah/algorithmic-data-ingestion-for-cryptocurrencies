# Week-Long Dry Run Checklist

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Added sanitizer/parity coverage (multi-symbol parquet, `export_feature_slice.py`, `compare_feature_stats.py`) and Redis-backed trading defaults so the rehearsal logs now capture the same symbol-aware gates enforced by scheduler/trading.

Use this checklist to track the current paper-trading exercise from start to finish. Update status daily (☐ → ☑) and capture notes in the right-hand column.

| Status | Item | Notes |
| --- | --- | --- |
| ☐ | Confirm Docker stack healthy (`scheduler`, `trading`, `ingestion-api`, `redis`, `prometheus`, `grafana`) |  |
| ☐ | Run trigger preflight (`scripts/trigger_preflight.py` with `configs/final_trigger_policy.yaml` + promoted model) before the day’s boot |  |
| ☐ | Verify scheduler warm-started data (ingest + backfill rows > 0 for BTC/ETH/SOL) |  |
| ☐ | Validate gate coverage per symbol/model ≥ expected baseline (Base ≈100%, TCN ≥40%) |  |
| ☐ | Confirm decision queue is draining (Redis `trading:decisions` near-empty after initial burst) |  |
| ☐ | Spot-check Redis feature payloads include `hl_spread_z`/`rvol_20`/`sym_*` ranks and populated `close`/`price` in scheduler slices |  |
| ☐ | Check trading metrics endpoint (`http://localhost:9010/metrics`) for non-zero `trading_trade_attempts_total` and `trading_trade_notional_total` |  |
| ☐ | Verify new counters (`trading_safe_mode_latched`, `trading_risk_blocked_total`, `trading_intent_ledger_state_total`, `deadlock_action_taken_total`) tick and reset as expected |  |
| ☐ | Export scheduler slice + parity diff (`scripts/export_feature_slice.py`, `scripts/compare_feature_stats.py`) and attach JSON (`release/calibration/latest/*.json`) to the day’s log |  |
| ☐ | Let `scheduler` + `trading` run ≥60 min (no restarts) and record start/end values for `trading_trade_attempts_total` **and** `trading_gate_toggles_total` to prove counters advance |  |
| ☐ | Ensure Redis state (`trading:positions`) reflects current open/closed positions each day |  |
| ☐ | Log realized P&L trend daily (even if zero) |  |
| ☐ | Capture Grafana screenshots: Gate Coverage, Order Fills, Turnover, P&L (start/mid/end of run) |  |
| ☐ | Review trading logs for dry-run order reasons (spread guard, invalid amount, etc.) |  |
| ☐ | Confirm reconciliation runs emit success entries (audit `reconciliation`) and safe-mode unlatches only after consecutive successes |  |
| ☐ | Spot-check audit records for `audit_source`/`audit_run_id`/`audit_seq` and valid HMAC signatures |  |
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
