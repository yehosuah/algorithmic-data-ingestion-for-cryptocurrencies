# Expectancy Fix Ground Truth

This document is the source-of-truth snapshot for the expectancy-fix work (baseline + post-change), tied to the extracted evidence bundles.

## Baseline (pre-change)

- Evidence bundle: `reports/log_forensics/evidence/20251225T181022Z` (audit log, metrics, manifest, env snapshot, deployment contract, risk/trigger policies).
  - Audit log: `reports/log_forensics/evidence/20251225T181022Z/trading_audit/audit.log` (trade events from 2025-12-15T01:49Z → 2025-12-25T18:08Z).
  - Env snapshot: `reports/log_forensics/evidence/20251225T181022Z/env_snapshot.txt`
  - Deployment contract: `reports/log_forensics/evidence/20251225T181022Z/deployment_contract.yaml`
  - Runtime risk limits: `reports/log_forensics/evidence/20251225T181022Z/portfolio_risk_limits_1.yaml`
  - Deadlock policy: `reports/log_forensics/evidence/20251225T181022Z/deadlock_policy.yaml`
  - Active model manifest: `reports/log_forensics/evidence/20251225T181022Z/manifest/manifest.json`

## Runtime configuration (source-of-truth values)

- Trading symbols (live): BTC/USDT, ETH/USDT, SOL/USDT (see `reports/log_forensics/evidence/20251225T181022Z/deployment_contract.yaml`).
- TRADING_MODELS (from `reports/log_forensics/evidence/20251225T181022Z/env_snapshot.txt`; all `shadow_mode=false`):
  - ETH/USDT (policy_id=primary): order_notional=20, max_spread_bps=15, min_hold_bars_override=6, max_hold_minutes=90, stop_loss_pct=0.0, take_profit_pct=0.001.
  - BTC/USDT (policy_id=conservative): order_notional=15, max_spread_bps=20, min_hold_bars_override=5, max_hold_minutes=90, stop_loss_pct=0.0, take_profit_pct=0.001.
  - SOL/USDT (policy_id=conservative): order_notional=12, max_spread_bps=22, min_hold_bars_override=5, max_hold_minutes=90, stop_loss_pct=0.0, take_profit_pct=0.001.

## Risk limits (active runtime YAML)

- Risk limits at baseline extraction (captured as `reports/log_forensics/evidence/20251225T181022Z/portfolio_risk_limits_1.yaml`; the repo file `configs/runtime_overrides/risk_limits_stage_0.yaml` was updated later as part of this work).
  - Capital 200, sizing_mode `equity_fraction` (equity_fraction=0.33, max_equity_fraction=0.35, compounding_step_usd=5).
  - Caps: max_total_notional=80, max_symbol_notional 40/45/30 (BTC/ETH/SOL), max_symbol_weight=0.5 (0.4 per symbol), max_concurrent_positions=3, max_orders_per_hour=180.
  - Loss controls: daily_loss_limit_pct=5%, max_drawdown_pct=20%, cooldowns (after_exit=2m, after_loss=5m), loss_guard (3 consecutive losses → 90m cooldown, notional_scale=0.5, reset_after_profit=true).
  - Stops: min_stop_loss_pct=0.005, hard_stop_loss_pct=0.012, vol_stop_rvol_mult=3.0 (TRADING_MODELS stop_loss_pct=0.0 triggers the min/vol stop logic in runtime).
  - Execution halts: halt_if_spread_bps_gt=35, halt_if_vol_zscore_gt=5, halt_if_data_stale_seconds=180, allow_exits_during_halt=true.
  - Per-symbol risk: max_spread_bps 25/25/28 (BTC/ETH/SOL), max_entry_slippage_bps 25/25/30, max_position_age_minutes=360.
  - Trigger overrides (used by `app/trading/service.py::_resolve_manifest`): exit_threshold 0.43/0.41/0.42 and exit_prob_drop 0.15/0.16/0.14 (BTC/ETH/SOL).

## Effective trigger/guard policy (runtime)

- Model manifest gates (see `reports/log_forensics/evidence/20251225T181022Z/manifest/manifest.json`):
  - prob_gate_min: BTC/USDT=0.56, default=0.48; long_only=true; min_hold_bars=3; rvol20_max=0.002.
  - manifest threshold.value (base exit threshold): 0.47; manifest metadata exit_prob_drop: 0.10.
- Trading service effective values (see `app/trading/service.py::_resolve_manifest` + TRADING_MODELS):
  - Entry threshold: manifest prob_gate_min (BTC 0.56, ETH 0.48, SOL 0.48).
  - Exit threshold: risk_limits per-symbol override (BTC 0.43, ETH 0.41, SOL 0.42).
  - Exit_prob_drop: risk_limits per-symbol override (BTC 0.15, ETH 0.16, SOL 0.14).
  - Min hold bars: TRADING_MODELS overrides (BTC 5, ETH 6, SOL 5).
  - Take profit: TRADING_MODELS take_profit_pct=0.001 (0.10%).
  - Stop loss: bar-aware, volatility-scaled with caps (min 0.50%, hard cap 1.20%).

## Deadlock / invariants

- Deadlock policy: `configs/runtime_overrides/deadlock_policy_stage_0.yaml` (mirrored as `reports/log_forensics/evidence/20251225T181022Z/deadlock_policy.yaml`) with adjust_prob_gate_min and safe-mode action.
- Invariants (from `configs/deployment_portfolio_contract.yaml`): idempotency requires order_intent_id; reconciliation required on startup; kill/safe mode env vars `TRADING_KILL_SWITCH` / `TRADING_SAFE_MODE`.

## Baseline analysis inputs

- Audit log: `reports/log_forensics/evidence/20251225T181022Z/trading_audit/audit.log`
- Market OHLCV: `data_lake/market/exchange=binance`

## Post-change (after fixes, pre-sizing)

- Evidence bundle: `reports/log_forensics/evidence/20251225T215109Z`
  - Audit log: `reports/log_forensics/evidence/20251225T215109Z/trading_audit/audit.log`
  - Env snapshot: `reports/log_forensics/evidence/20251225T215109Z/env_snapshot.txt`
  - Runtime risk limits: `reports/log_forensics/evidence/20251225T215109Z/portfolio_risk_limits_1.yaml`
  - Deadlock policy: `reports/log_forensics/evidence/20251225T215109Z/deadlock_policy.yaml`
  - Active model manifest: `reports/log_forensics/evidence/20251225T215109Z/manifest/manifest.json`

### Runtime configuration (post-change)

- TRADING_MODELS (from `reports/log_forensics/evidence/20251225T215109Z/env_snapshot.txt`; all `shadow_mode=false`):
  - ETH/USDT (policy_id=primary): order_notional=20, max_spread_bps=8, min_hold_bars_override=3, max_hold_minutes=90, stop_loss_pct=0.004, take_profit_pct=0.0.
  - BTC/USDT (policy_id=conservative): order_notional=15, max_spread_bps=8, min_hold_bars_override=8, max_hold_minutes=90, stop_loss_pct=0.005, take_profit_pct=0.0.
  - SOL/USDT (policy_id=conservative): order_notional=12, max_spread_bps=8, min_hold_bars_override=3, max_hold_minutes=90, stop_loss_pct=0.004, take_profit_pct=0.0.

### Risk limits (post-change)

- Risk limits source: `configs/runtime_overrides/risk_limits_stage_0.yaml` (mirrored as `reports/log_forensics/evidence/20251225T215109Z/portfolio_risk_limits_1.yaml`).
  - Capital 200, sizing_mode `equity_fraction` (equity_fraction=0.33, max_equity_fraction=0.35), max_total_notional=80.
  - Trigger overrides: entry_threshold=0.48, exit_threshold=0.47, exit_prob_drop=0.15.
    - BTC override: entry_threshold=0.56, exit_threshold=0.54, exit_prob_drop=0.15.
  - Stops: min_stop_loss_pct=0.005, hard_stop_loss_pct=0.012, vol_stop_rvol_mult=3.0 (trade-level cap).
  - Execution halts: halt_if_spread_bps_gt=35, allow_exits_during_halt=true.

### Effective trigger/guard policy (post-change)

- Model manifest gates (`reports/log_forensics/evidence/20251225T215109Z/manifest/manifest.json`):
  - prob_gate_min: BTC/USDT=0.56, default=0.48; long_only=true; min_hold_bars=3; rvol20_max=0.002.
  - manifest threshold.value (base exit threshold): 0.47; manifest metadata exit_prob_drop: 0.10.
- Trading service effective values:
  - Entry threshold: risk_limits (matches manifest gates: BTC 0.56, ETH/SOL 0.48).
  - Exit threshold: risk_limits override (BTC 0.54, ETH/SOL 0.47).
  - Exit_prob_drop: risk_limits override (0.15).
  - Min hold bars: TRADING_MODELS overrides (BTC 8, ETH 3, SOL 3); stop-loss exits bypass min-hold (code fix).
  - Take profit: disabled (TRADING_MODELS take_profit_pct=0.0).
  - Spread guard: entry max_spread_bps=8; exit is allowed at a relaxed cap (halt spread 35 bps) for protective exits.

## Post-change (final sizing)

- Evidence bundle: `reports/log_forensics/evidence/20251225T223541Z`
  - Audit log: `reports/log_forensics/evidence/20251225T223541Z/trading_audit/audit.log`
  - Env snapshot: `reports/log_forensics/evidence/20251225T223541Z/env_snapshot.txt`
  - Runtime risk limits: `reports/log_forensics/evidence/20251225T223541Z/portfolio_risk_limits_1.yaml`

### Runtime configuration (final)

- TRADING_MODELS (from `reports/log_forensics/evidence/20251225T223541Z/env_snapshot.txt`; all `shadow_mode=false`):
  - ETH/USDT (policy_id=primary): order_notional=5, max_spread_bps=8, min_hold_bars_override=3, max_hold_minutes=90, stop_loss_pct=0.004, take_profit_pct=0.0.
  - BTC/USDT (policy_id=conservative): order_notional=15, max_spread_bps=8, min_hold_bars_override=8, max_hold_minutes=90, stop_loss_pct=0.005, take_profit_pct=0.0.
  - SOL/USDT (policy_id=conservative): order_notional=12, max_spread_bps=8, min_hold_bars_override=3, max_hold_minutes=90, stop_loss_pct=0.004, take_profit_pct=0.0.
