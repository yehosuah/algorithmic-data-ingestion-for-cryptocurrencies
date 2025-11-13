# Deployability Runbook (TCN Suite Refresh – Oct 2025)

_Last updated: 2025-11-13 04:43 UTC_

> Update 2025-11-13: Folded in the sanitizer + symbol-gate workflow and the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so deployability steps reference the gates/metrics enforced by scheduler + trading.

## Immediate Follow-Ups
- **Refresh shortlist** – `python3 scripts/report_shortlist.py --models-root models --out models/report_shortlist.json` to surface the updated TCN manifests alongside base/blender.
- **Freeze sanitizer + gate payload** – Re-run `training.data.sanitize_market_dataset` on the multi-symbol parquet and `scripts/compute_symbol_gate_config.py --data datasets/market_multi_3symbol_1m.parquet` so the release bundle ships with `release/symbol_gates/market_multi_3symbol_1m.json` that scheduler/trading already consume.
- **Assemble release bundle** – Package manifests, `report.json`, `tcn_gate_replay_summary.json`, `oos_replay_summary_latest.json`, `live_gate_coverage.csv`, and the forward matrix parquet per `docs/final_stretch_v1.md`.
- **CI gate coverage check** – GitHub Actions already runs `scripts/run_oos_eval.py --family tcn --stride 30` for h60/h120/h180; keep the thresholds strict (`gate_coverage < 5e-4` or `final_equity < 1.2` triggers failure) and mirror the guardrail locally before adjusting manifests.
- **Documentation touch-up** – Update `docs/oct_2025_forward_replay.md` and `TRAINING_STATUS.md` with the new TCN gate thresholds/coverage before circulating status reports.
- **Monitoring baseline update** – Append Oct 2025 figures to `live_gate_coverage.csv` so alert thresholds reflect the relaxed yet deployable gates.
- **Trading rehearsal plan** – Populate `INFER_JOBS`/`TRADING_MODELS`, review `docs/weeklong_dry_run_checklist.md`, and schedule a 7-day paper-trading run logging queue depth, trade attempts, and faux P&L.
- **Feature parity proof** – At the end of each dry run, export a scheduler slice (`scripts/export_feature_slice.py`) and store the diff vs sanitized training data via `scripts/compare_feature_stats.py --train datasets/market_multi_3symbol_1m.parquet --live /tmp/features_debug.parquet --out release/calibration/latest/tcn_parity.json`; attach the JSON to deployment notes before widening gates.

## Connecting to Live Trading
- **Manifest enforcement in inference**  
  - Load manifests with `training.infer.load_manifest_artifacts` at service startup.  
  - Apply `training.infer.apply_manifest_gates` after scoring to emit `gate_pass` alongside probabilities.
- **Scheduled inference pipeline**  
  - Populate `INFER_JOBS` for `app/scheduler/main.py`; jobs read parquet windows, score base/TCN manifests (stride 30 defaults), compute gate coverage, and enqueue decision payloads to Redis (`DECISION_QUEUE_KEY`, default `trading:decisions`).
- **Execution layer integration**  
  - Deploy `app/trading/service.py` (Docker service `trading`) to consume `trading:decisions`, enforce manifest gates/min-hold, and route dry-run orders via the CCXT adapter with spread guardrails.  
  - Persist per-symbol state using `app/trading/state.py` (file/Redis/Postgres backends) and log audit events with `app/trading/audit.py` for retrospectives.
  - Mount `TRADING_STATE_PATH` on a persistent volume (Docker Compose maps `trading-state:/app/trading_state/state.json`) so `last_timestamp` survives container restarts and stale payloads are ignored by the `register_bar` guard.  
  - Mirror the scheduler’s Redis endpoint for the trading worker, track per-model decision timestamps in `TRADING_LAST_TS_HASH` (default `trading:last_processed_ts`), and let `_handle_payload` drop anything older; before restarting the worker, run `redis-cli -u $DECISION_QUEUE_URL DEL trading:decisions` (or `LTRIM` up to the persisted timestamp) to avoid replay storms.  
  - Redis outages that close `BLPOP` sockets now trigger reconnection with exponential backoff—keep Prometheus alerts/ Grafana panels pointed at the shared Redis instance so scheduler + trading failures surface together.
- **Monitoring & alerting**  
  - Export `model_gate_coverage_ratio`, `model_probability_sigma`, `model_rss_minute_spike_share`, plus trading counters/gauges (`trading_trade_attempts_total`, `trading_gate_toggles_total`, `trading_trade_notional_total`, `trading_position_active`, `trading_realized_pnl_total`).  
  - Update `monitoring/alert.rules.yml` to watch the new coverage floor (~5 e‑4), probability σ guardrails, stale decision queues, and missing trading audit events.  
  - Publish Grafana panels for gate coverage, queue depth, turnover, faux P&L, and order execution latency (see `monitoring/grafana/dashboards/trading-overview.json`).
- **Dry-run then cutover**  
  - Execute a full paper-trading rehearsal with `TRADING_DRY_RUN=1`, capturing queue depth, trading metrics, audit stream entries, and latency.  
  - Once validated, set `TRADING_DRY_RUN=0`, rotate in live credentials, keep heightened alerting for the first week, and prepare rollback scripts (base-only mode, disable trading service) if coverage collapses.
