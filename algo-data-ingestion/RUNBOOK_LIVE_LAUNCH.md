# Live Launch Ladder Runbook (ETH → BTC → SOL)

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.


This runbook is the operator guide for staged rollout using the launch ladder in `configs/live_launch_ladder.yaml` plus the CLIs under `analysis/`.

## Prerequisites
- Env: `TRADING_INTENT_LEDGER_BACKEND=redis`, `TRADING_INTENT_LEDGER_REDIS_URL` set, `TRADING_AUDIT_HMAC_KEY` set, `TRADING_KILL_SWITCH`/`TRADING_SAFE_MODE` available.
- Redis reachable for decision queue, state, audit, and intent ledger.
- Clock sync within 5s on trading host.
- Credentials for exchange/API already loaded; models present under `models_root`.
- Audit log writable at `data_lake/trading/audit.log`; HMAC validation must pass for live.
- Ensure `docker-compose` (or systemd) can restart trading service after config changes.

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
2) Validate contract: `python -m analysis.validate_deployment_contract --contract configs/deployment_portfolio_contract.yaml`.  
3) Coverage readiness: `python -m analysis.preflight_coverage --contract configs/deployment_portfolio_contract.yaml --output-dir reports` and `python -m analysis.shadow_readiness --contract configs/deployment_portfolio_contract.yaml --reports-dir reports`.  
4) Optional stress: `python -m analysis.preflight_symbol_promotion --stage stage_N --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml` to simulate kill/safe toggles and dedupe load.  
5) Start/refresh trading service (docker-compose up or `python -m app.trading.main`) with env from stage bundle. Keep `TRADING_KILL_SWITCH=1` and allow exits-only until reconciliation emits `reconciliation` audit entries with a healthy streak.  
6) Monitor metrics: Prometheus `9010` (`trade_count`, `coverage`, `deadlock_action_taken_total`, `trading_safe_mode_latched`, `trading_risk_blocked_total`, `trading_intent_ledger_state_total`, `trading_reconcile_runs_total`). Watch audit HMAC validity and Redis intent ledger health.  
7) Run gate evaluation for the next stage (command above). Review `reports/launch_stage_eval_stage_N_*.md/.json` and deadlock drill output.  
8) If status is GO, apply the next stage and repeat. If NO-GO, hold stage, fix issues, and re-evaluate.

## Incident Playbooks
- **Coverage deadlock / no trades**: Check `deadlock_action_taken_total`, `decision_coverage`, audit `deadlock_status`. Keep `TRADING_SAFE_MODE=1` if positions exist, apply next deadlock action from policy, or rollback stage.
- **Reconciliation mismatches**: Audit `reconciliation` events; latch `TRADING_SAFE_MODE=1`, rerun reconcile, and if mismatch persists, rollback to `stage_0`.
- **Excessive spread blocks**: Inspect audit `risk_block_reason`/`spread_bps`. Consider tightening stage sizing or reverting to prior stage; do not relax spread limits while live without evaluation.
- **Repeated order rejects / intent collisions**: Verify Redis intent ledger health; ensure `TRADING_INTENT_LEDGER_BACKEND=redis` and redis URL valid. Enable kill switch if rejects persist.
- **Drawdown breach**: Immediately set `TRADING_KILL_SWITCH=1`, keep `TRADING_SAFE_MODE=1`, then rollback to `stage_0`.
- **Audit or HMAC failures**: Halt trading, rotate `TRADING_AUDIT_HMAC_KEY`, re-run contract validation, and replay the last ladder stage before re-enabling entries.

## Rollback (single action)
- Command: `python -m analysis.rollback_to_stage --stage stage_0 --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml`
- Then: `export TRADING_KILL_SWITCH=1` and `export TRADING_SAFE_MODE=1`, restart trading so exits-only behavior applies. Use stage_0 env bundle (`configs/runtime_overrides/stage_0.yaml`), keep monitoring until flat.
