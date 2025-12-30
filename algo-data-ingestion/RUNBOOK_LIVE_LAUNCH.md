# Live Launch Ladder Runbook (ETH → BTC → SOL)

_Last updated: 2025-12-30 22:59 UTC_

> Update 2025-12-30: Stage-0 now layers vol-aware stop shaping (`min_stop_loss_pct`/`hard_stop_loss_pct`/`vol_stop_rvol_mult`) plus optional quote-based price monitoring (`TRADING_PRICE_MONITOR_INTERVAL_SECONDS`) to enforce exits even when decision payloads stall.
> Update 2025-12-19: Stage-0 bundle now uses equity-fraction sizing (capital 100, base notionals 20/15/12, compounding step 5, per-symbol trigger overrides with longer holds) and the dry-run profit forensics loop (`RUNBOOK_DRY_RUN_PROFIT.md`) is available for post-rehearsal signoff.
> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.


This runbook is the operator guide for staged rollout using the launch ladder in `configs/live_launch_ladder.yaml` plus the CLIs under `analysis/`.

## Prerequisites
- Env: `TRADING_INTENT_LEDGER_BACKEND=redis`, `TRADING_INTENT_LEDGER_REDIS_URL` set, `TRADING_AUDIT_HMAC_KEY` set, `TRADING_KILL_SWITCH`/`TRADING_SAFE_MODE` available.
- Redis reachable for decision queue, state, audit, and intent ledger.
- Clock sync within 5s on trading host.
- Credentials for exchange/API already loaded; models present under `models_root`.
- Audit log writable at `data_lake/trading/audit.log`; HMAC validation must pass for live.
- Ensure `docker-compose` (or systemd) can restart trading service after config changes.

## Single-command preflight (GO/NO-GO)
Use `analysis/live_readiness_check.py` to deterministically answer “are we live-run ready right now?” and write a single JSON + Markdown report bundle.

- Dry-run (no audit-based checks):
  `python3 -m analysis.live_readiness_check --deployment-contract configs/deployment_portfolio_contract.yaml --mode dry_run --no-require-shadow-preflight --output-dir reports/live_readiness`
- Live-like (recommended before promotions; requires an audit log when shadow/stage checks are enabled):
  `python3 -m analysis.live_readiness_check --deployment-contract configs/deployment_portfolio_contract.yaml --mode live_like --audit-log data_lake/trading/audit.log --lookback-hours 48 --output-dir reports/live_readiness`
- Live (strict):
  `python3 -m analysis.live_readiness_check --deployment-contract configs/deployment_portfolio_contract.yaml --mode live --audit-log data_lake/trading/audit.log --lookback-hours 48 --output-dir reports/live_readiness`

Interpretation:
- `GO` means every required check `PASS`ed; `NO_GO` means at least one required check `FAIL`ed.
- The report bundle is written under `reports/live_readiness/<stamp>_live_readiness.json` and `.md`, and links the underlying artifacts (coverage, shadow readiness, stage eval) in the same directory.

Top failure playbooks:
- **Coverage deadlock / implied trades == 0**: inspect `preflight_coverage_*.md` in the output dir; verify feature feed freshness and `gate_config.prob_gate_min` in risk limits (don’t relax silently).
- **Audit provenance/HMAC failures**: ensure audit lines include `audit_source`, `audit_run_id`, `audit_seq`, and `audit_hmac`; set/rotate `TRADING_AUDIT_HMAC_KEY` and rerun (no unauthenticated overrides in live).
- **Contract mismatch**: rerun `analysis.validate_deployment_contract`; fix missing model paths, missing risk limits per symbol, or `TRADING_MODELS` not covering every `live_symbol`.

## Stage Ladder (source of truth)
- Ladder file: `configs/live_launch_ladder.yaml`.
- Apply a stage (writes contract + overrides):  
  `python -m analysis.apply_launch_stage --stage stage_N --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml`
- Evaluate gates for a stage (GO/NO-GO):  
  `python -m analysis.evaluate_launch_stage --stage stage_N --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml --audit-log data_lake/trading/audit.log --reports-dir reports`
- Rollback to baseline:  
  `python -m analysis.rollback_to_stage --stage stage_0 --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml`
- Each stage write creates:
  - `configs/runtime_overrides/risk_limits_stage_N.yaml`
  - `configs/runtime_overrides/deadlock_policy_stage_N.yaml`
  - `configs/runtime_overrides/stage_N.yaml` (env bundle with `TRADING_MODELS`, `TRADING_SHADOW_SYMBOLS`, `TRADING_RISK_LIMITS_PATH`, `TRADING_DEADLOCK_POLICY_PATH`, `TRADING_INTENT_LEDGER_*`, `TRADING_DRY_RUN`)
  - Markdown/JSON readiness artifacts under `reports/launch_stage_eval_stage_N_*`, `reports/deadlock_drill_*`, and `reports/shadow_readiness_*`.

## GO-Live Sequence (for each promotion)
1) Apply current stage (above command) and export the env bundle from `configs/runtime_overrides/stage_N.yaml`.  
2) Run the single-command preflight (above). Add `--stage stage_N --ladder configs/live_launch_ladder.yaml` to gate stage promotions.  
3) If `NO_GO`, fix the failing check and rerun until `GO`. Archive the readiness artifacts under `reports/live_readiness/`.  
5) Start/refresh trading service (docker-compose up or `python -m app.trading.main`) with env from stage bundle. Keep `TRADING_KILL_SWITCH=1` and allow exits-only until reconciliation emits `reconciliation` audit entries with a healthy streak.  
6) Monitor metrics: Prometheus `9010` (`trade_count`, `coverage`, `deadlock_action_taken_total`, `trading_safe_mode_latched`, `trading_risk_blocked_total`, `trading_intent_ledger_state_total`, `trading_reconcile_runs_total`). Watch audit HMAC validity and Redis intent ledger health.  
7) Run gate evaluation for the next stage (command above). Review `reports/launch_stage_eval_stage_N_*.md/.json` and deadlock drill output.  
8) If status is GO, apply the next stage and repeat. If NO-GO, hold stage, fix issues, and re-evaluate.
- After each rehearsal, capture a forensics bundle per `RUNBOOK_DRY_RUN_PROFIT.md` (extract container logs, run trading log forensics, align trades vs OHLCV) so approvals include PnL/regret evidence.

## Incident Playbooks
- **Coverage deadlock / no trades**: Check `deadlock_action_taken_total`, `decision_coverage`, audit `deadlock_status`. Keep `TRADING_SAFE_MODE=1` if positions exist, apply next deadlock action from policy, or rollback stage.
- **Reconciliation mismatches**: Audit `reconciliation` events; latch `TRADING_SAFE_MODE=1`, rerun reconcile, and if mismatch persists, rollback to `stage_0`.
- **Excessive spread blocks**: Inspect audit `risk_block_reason`/`spread_bps`. Consider tightening stage sizing or reverting to prior stage; do not relax spread limits while live without evaluation.
- **Repeated order rejects / intent collisions**: Verify Redis intent ledger health; ensure `TRADING_INTENT_LEDGER_BACKEND=redis` and redis URL valid. Enable kill switch if rejects persist.
- **Loss guard & stale decisions**: The runtime risk limits now include a `loss_guard` (three losses → 90-minute cooldown with optional notional scaling) and trading honors `TRADING_DECISION_MAX_AGE_SECONDS`/`TRADING_LAST_TS_GRACE_BARS`. Look for `loss_guard` or `pnl_block` reasons in the audit stream, escalate the path if repeated skips are blocking entries, and check whether the Redis queue is emitting stale decision warnings (increase the grace/age if the scheduler burst is healthy).
- **Drawdown breach**: Immediately set `TRADING_KILL_SWITCH=1`, keep `TRADING_SAFE_MODE=1`, then rollback to `stage_0`.
- **Audit or HMAC failures**: Halt trading, rotate `TRADING_AUDIT_HMAC_KEY`, re-run contract validation, and replay the last ladder stage before re-enabling entries.

## Rollback (single action)
- Command: `python -m analysis.rollback_to_stage --stage stage_0 --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml`
- Then: `export TRADING_KILL_SWITCH=1` and `export TRADING_SAFE_MODE=1`, restart trading so exits-only behavior applies. Use stage_0 env bundle (`configs/runtime_overrides/stage_0.yaml`), keep monitoring until flat.
