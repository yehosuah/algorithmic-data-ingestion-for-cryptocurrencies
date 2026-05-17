from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from redis import asyncio as aioredis

from app.monitoring.trading_metrics import (
    record_decision_queue_depth,
    record_decision_coverage,
    record_gate_toggle,
    record_realized_pnl,
    init_realized_pnl,
    record_reconcile_run,
    record_safe_mode,
    record_dedup_blocked,
    record_shadow_blocked,
    record_skip_reason,
    record_trade_attempt,
    record_would_trade,
    record_risk_blocked,
    record_risk_clipped,
    record_portfolio_turnover_estimate,
    record_portfolio_drawdown,
    record_portfolio_daily_pnl,
    record_orders_per_hour,
    record_concurrent_positions,
    record_deadlock_window_metrics,
    record_deadlock_portfolio_metrics,
    record_deadlock_block_reason,
    record_deadlock_action,
    set_position_active,
)
from app.trading.audit import TradingAuditLogger
from app.trading.config import TradingConfig, TradingModelConfig
from app.trading.executor import IntentLedger, IntentStatus, OrderDecision, OrderExecutor
from app.trading.deadlock import DeadlockMonitor, DeadlockPolicy
from app.trading.decision import TriggerConfig, DecisionOutcome, SAFE_MODE_ENV, KILL_SWITCH_ENV, decide_bar
from app.trading.risk import assess_and_adjust_order
from app.trading.signing import verify_decision_payload
from app.trading.state import PositionState, TradingStateStore
from training.infer import load_manifest_artifacts

logger = logging.getLogger("app.trading.service")


def _parse_timestamp(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        ts = raw
    else:
        try:
            ts = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _extract_price_from_item(item: Dict[str, object]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    features = item.get("features")
    feature_dict = features if isinstance(features, dict) else {}
    for key in ("price", "close", "mid_price", "last_price", "bid", "ask"):
        value = item.get(key)
        if value is None:
            value = feature_dict.get(key)
            if value is None and key == "price":
                value = feature_dict.get("close")
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0.0:
            return price
    return None


def _extract_spread_from_item(item: Dict[str, object]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    features = item.get("features")
    if not isinstance(features, dict):
        return None
    for key in ("hl_spread", "spread", "bid_ask_spread", "spread_bps"):
        raw_value = features.get(key)
        if raw_value is None:
            continue
        try:
            spread = float(raw_value)
        except (TypeError, ValueError):
            continue
        if key == "hl_spread":
            spread *= 1e4
        return spread
    return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_volatility_from_item(item: Dict[str, object]) -> Optional[float]:
    """
    Pull a realized volatility proxy from the feature payload to scale stops.
    """
    if not isinstance(item, dict):
        return None
    features = item.get("features") or {}
    if not isinstance(features, dict):
        return None
    for key in ("feat_realized_vol_1h", "rvol_20", "rvol20", "ret_std_20"):
        if key in features:
            try:
                vol = abs(float(features[key]))
                if math.isfinite(vol) and vol > 0:
                    return vol
            except (TypeError, ValueError):
                continue
    return None


def _entry_filter_block_reason(model_cfg: TradingModelConfig, item: Dict[str, object]) -> Optional[str]:
    """
    Optional per-model entry filters that block entries even when the model gate passes.

    These are intended to prevent trading during regime/trend conditions where the
    underlying probability signal is known to degrade.
    """
    if not isinstance(item, dict):
        return None
    features = item.get("features") or {}
    if not isinstance(features, dict):
        features = {}

    rsi_min = getattr(model_cfg, "entry_rsi_min", None)
    if rsi_min is not None:
        raw = features.get("rsi_14")
        try:
            rsi = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            rsi = None
        if rsi is None or not math.isfinite(rsi):
            return "entry_filter_rsi_missing"
        if rsi <= float(rsi_min):
            return "entry_filter_rsi"

    macd_min = getattr(model_cfg, "entry_macd_min", None)
    if macd_min is not None:
        raw = features.get("macd")
        try:
            macd = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            macd = None
        if macd is None or not math.isfinite(macd):
            return "entry_filter_macd_missing"
        if macd <= float(macd_min):
            return "entry_filter_macd"

    return None


def _resolve_symbol_value(value: Any, symbol: Optional[str]) -> Any:
    """
    Resolve per-symbol gate config entries that may provide defaults.
    """
    if not isinstance(value, dict):
        return value
    symbol_key = (symbol or "").strip()
    if symbol_key and symbol_key in value:
        return value[symbol_key]
    if "default" in value:
        return value["default"]
    for candidate in value.values():
        if candidate is not None:
            return candidate
    return None


def _extract_min_cost(market: Dict[str, object]) -> Optional[float]:
    limits = (market or {}).get("limits", {}) or {}
    try:
        cost_limit = (limits.get("cost") or {}).get("min")
        if cost_limit is None:
            cost_limit = (market or {}).get("minCost")
        return float(cost_limit) if cost_limit is not None else None
    except (TypeError, ValueError):
        return None


def _extract_min_amount(market: Dict[str, object]) -> Optional[float]:
    limits = (market or {}).get("limits", {}) or {}
    try:
        amount_limit = (limits.get("amount") or {}).get("min")
        return float(amount_limit) if amount_limit is not None else None
    except (TypeError, ValueError):
        return None


def _extract_qty_step(market: Dict[str, object]) -> Optional[float]:
    precision = (market or {}).get("precision", {}) or {}
    try:
        amount_prec = precision.get("amount")
        if amount_prec is not None and float(amount_prec) >= 0:
            return 10 ** (-float(amount_prec))
    except Exception:
        pass
    try:
        step = ((market or {}).get("limits", {}) or {}).get("amount", {}).get("step")
        return float(step) if step is not None else None
    except Exception:
        return None


def _extract_price_tick(market: Dict[str, object]) -> Optional[float]:
    precision = (market or {}).get("precision", {}) or {}
    try:
        price_prec = precision.get("price")
        if price_prec is not None and float(price_prec) >= 0:
            return 10 ** (-float(price_prec))
    except Exception:
        pass
    try:
        tick = ((market or {}).get("limits", {}) or {}).get("price", {}).get("min")
        return float(tick) if tick is not None else None
    except Exception:
        return None


@dataclass(frozen=True)
class ManifestSnapshot:
    entry_threshold: float
    exit_threshold: float
    exit_prob_drop: float
    min_hold_bars: int
    long_only: bool


def _build_order_intent_id(model_cfg: TradingModelConfig, ts: datetime, side: str) -> str:
    ts_utc = ts
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_utc = ts_utc.astimezone(timezone.utc)
    epoch_ms = int(ts_utc.timestamp() * 1000)
    symbol = (model_cfg.symbol or "").replace("/", "_")
    policy = (model_cfg.policy_id or model_cfg.model or "").replace(":", "_")
    action = "ENTER_LONG" if side.lower() == "buy" else "EXIT_LONG"
    timeframe = (model_cfg.timeframe or "").strip().lower()
    return f"{model_cfg.model}:{policy}:{symbol}:{timeframe}:{epoch_ms}:{action}"


class TradingService:
    # Log telemetry for every trade event to avoid silent sampling gaps.
    TELEMETRY_SAMPLE_LIMIT = None

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.state_store = TradingStateStore(
            config.state_path,
            backend=config.state_backend,
            redis_url=config.state_redis_url,
            redis_hash=config.state_redis_hash,
            postgres_dsn=config.state_postgres_dsn,
            postgres_table=config.state_postgres_table,
        )
        audit_backend = config.audit_backend or config.state_backend
        audit_redis_url = config.audit_redis_url or config.state_redis_url
        audit_postgres_dsn = config.audit_postgres_dsn or config.state_postgres_dsn
        audit_file_path = config.audit_log_path if audit_backend == "file" else None
        audit_hmac_key = os.getenv("TRADING_AUDIT_HMAC_KEY")
        if not config.dry_run and not audit_hmac_key:
            raise ValueError("Live trading requires TRADING_AUDIT_HMAC_KEY for audit provenance.")
        self.audit_logger = TradingAuditLogger(
            backend=audit_backend,
            redis_url=audit_redis_url,
            redis_stream=config.audit_redis_stream,
            redis_maxlen=config.audit_maxlen,
            postgres_dsn=audit_postgres_dsn,
            postgres_table=config.audit_postgres_table,
            file_path=audit_file_path,
            hmac_key=audit_hmac_key,
        )
        ledger_backend = config.intent_ledger_backend
        ledger_url = config.intent_ledger_redis_url
        if not config.dry_run:
            if ledger_backend != "redis":
                raise ValueError("Live trading requires TRADING_INTENT_LEDGER_BACKEND=redis (no fallback).")
            if not ledger_url:
                raise ValueError("Live trading requires TRADING_INTENT_LEDGER_REDIS_URL for intent ledger.")
        if ledger_backend == "redis" and not ledger_url:
            ledger_backend = "memory"
        self.intent_ledger = IntentLedger(
            backend=ledger_backend,
            redis_url=ledger_url,
            prefix=config.intent_ledger_prefix,
            lock_ttl_seconds=config.intent_lock_ttl_seconds,
        )
        self.executor = OrderExecutor(dry_run=config.dry_run, intent_ledger=self.intent_ledger)
        self._manifest_cache: Dict[Tuple[Path, Optional[str]], ManifestSnapshot] = {}
        self._model_map: Dict[Tuple[str, Optional[str]], TradingModelConfig] = {}
        for model_cfg in config.trading_models:
            symbol_key = (model_cfg.model, model_cfg.symbol)
            self._model_map[symbol_key] = model_cfg
            # Preserve backwards compatibility with single-entry configs by keeping
            # the first occurrence as the fallback when symbol is omitted.
            fallback_key = (model_cfg.model, None)
            self._model_map.setdefault(fallback_key, model_cfg)
        self.risk_limits = self._load_risk_limits()
        self._risk_capital = float(self.risk_limits.get("capital") or 0.0)
        self._risk_turnover: Dict[str, float] = defaultdict(float)
        self._risk_order_times: deque[datetime] = deque()
        self._risk_daily_pnl: Dict[str, float] = defaultdict(float)
        self._risk_equity = self._risk_capital
        self._risk_equity_peak = self._risk_equity
        self._risk_last_loss_ts: Optional[datetime] = None
        self._market_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._redis: Optional[aioredis.Redis] = None
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._reconcile_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._reconcile_healthy_streak = 0
        self._safe_mode_reason: Optional[str] = None
        self._telemetry_samples: Dict[Tuple[str, Optional[str], str], int] = defaultdict(int)
        self._last_price_monitor_ts: Optional[datetime] = None
        self.deadlock_policy = self._load_deadlock_policy()
        self.deadlock_monitor = DeadlockMonitor(self.deadlock_policy.window_minutes)
        self._deadlock_action_state: Dict[str, object] = {
            "last_action_ts": None,
            "actions_taken_today": 0,
            "day": None,
            "next_index": 0,
        }
        self._prob_gate_overrides: Dict[str, float] = {}
        self._policy_overrides: Dict[str, str] = {}
        # Initialize per-symbol gauges so Prometheus exports stable time series
        # even before the first decision payload is processed.
        for model_cfg in config.trading_models:
            state = self.state_store.get(model_cfg.state_key)
            init_realized_pnl(model_cfg.model, model_cfg.symbol)
            set_position_active(model_cfg.model, model_cfg.symbol, state.in_position)

    def _create_queue_client(self) -> aioredis.Redis:
        return aioredis.from_url(
            self.config.decision_queue_url,
            encoding="utf-8",
            decode_responses=True,
        )

    def _load_risk_limits(self) -> Dict[str, object]:
        path = self.config.risk_limits_path
        if not path.exists():
            raise FileNotFoundError(f"Risk limits file not found at {path}")
        with path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Risk limits at {path} must be a mapping.")
        return data

    def _load_deadlock_policy(self) -> DeadlockPolicy:
        payload: Dict[str, object] = {}
        if self.config.deadlock_policy_path and self.config.deadlock_policy_path.exists():
            try:
                payload = yaml.safe_load(self.config.deadlock_policy_path.read_text()) or {}
            except Exception as exc:
                logger.warning("Failed to read deadlock policy from %s: %s", self.config.deadlock_policy_path, exc)
        elif self.config.deployment_contract_path and self.config.deployment_contract_path.exists():
            try:
                contract = yaml.safe_load(self.config.deployment_contract_path.read_text()) or {}
                payload = contract.get("deadlock_policy") or {}
            except Exception as exc:
                logger.warning("Failed to read deadlock policy from contract %s: %s", self.config.deployment_contract_path, exc)
        if not payload and self.config.deadlock_policy_payload:
            try:
                payload = json.loads(self.config.deadlock_policy_payload)
            except json.JSONDecodeError:
                logger.warning("TRADING_DEADLOCK_POLICY payload is not valid JSON; ignoring.")
        if not isinstance(payload, dict):
            payload = {}
        self._deadlock_policy_raw = payload
        policy = DeadlockPolicy.from_payload(payload)
        if self.config.dry_run and not os.getenv("TRADING_ENABLE_DEADLOCK_POLICY"):
            policy.enabled = False
        return policy

    def _record_deadlock_trade_event(
        self,
        model_cfg: TradingModelConfig,
        ts: datetime,
        *,
        executed: bool,
        reason: Optional[str] = None,
    ) -> None:
        try:
            self.deadlock_monitor.record_trade(
                model=model_cfg.model,
                symbol=model_cfg.symbol,
                ts=ts,
                executed=executed,
                blocked_reason=reason,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug(
                "Deadlock monitor trade record failed for %s %s: %s",
                model_cfg.model,
                model_cfg.symbol,
                exc,
            )
        if reason:
            record_deadlock_block_reason(model_cfg.model, model_cfg.symbol, reason)

    def _deadlock_config_hash(self) -> str:
        try:
            return self.deadlock_policy.config_hash()
        except Exception:
            return "unknown"

    def _reset_deadlock_daily_state(self, now: datetime) -> None:
        state_day = self._deadlock_action_state.get("day")
        if state_day != now.date():
            self._deadlock_action_state["day"] = now.date()
            self._deadlock_action_state["actions_taken_today"] = 0

    def _can_apply_deadlock_action(self, now: datetime) -> bool:
        if not self.deadlock_policy.enabled:
            return False
        self._reset_deadlock_daily_state(now)
        last_ts = self._deadlock_action_state.get("last_action_ts")
        if isinstance(last_ts, datetime):
            delta = now - last_ts
            if delta < timedelta(minutes=max(0, self.deadlock_policy.cooldown_minutes)):
                return False
        taken_today = int(self._deadlock_action_state.get("actions_taken_today", 0))
        if taken_today >= self.deadlock_policy.max_actions_per_day:
            return False
        return True

    def _next_deadlock_action(self) -> Optional[Mapping[str, object]]:
        idx = int(self._deadlock_action_state.get("next_index", 0))
        if idx >= len(self.deadlock_policy.actions):
            return None
        action = self.deadlock_policy.actions[idx]
        if not isinstance(action, Mapping):
            return None
        return action

    async def _evaluate_deadlock(self, now: datetime) -> None:
        statuses, portfolio = self.deadlock_monitor.snapshot()
        for status in statuses:
            record_deadlock_window_metrics(
                model=status.model,
                symbol=status.symbol,
                window_label=status.window_label,
                coverage_ratio=status.coverage_ratio,
                prob_gate_pass_ratio=status.prob_gate_pass_ratio,
                trade_count=status.trade_count,
            )
        record_deadlock_portfolio_metrics(
            window_label=self.deadlock_monitor.window_label,
            trade_count=int(portfolio.get("trade_count", 0)),
            coverage_ratio=float(portfolio.get("coverage_ratio", 0.0)),
        )
        if not self.deadlock_policy.enabled:
            return
        triggers = [
            s
            for s in statuses
            if s.trade_count < self.deadlock_policy.min_trades_window
            or s.coverage_ratio < self.deadlock_policy.min_coverage_ratio_window
            or s.prob_gate_pass_ratio < self.deadlock_policy.min_coverage_ratio_window
        ]
        if not triggers:
            return
        await self._handle_deadlock_actions(triggers, portfolio, now)

    async def _handle_deadlock_actions(
        self,
        triggers: List["DeadlockStatus"],
        portfolio_metrics: Mapping[str, object],
        now: datetime,
    ) -> None:
        await self.audit_logger.log_deadlock_status(
            triggers=triggers,
            portfolio=portfolio_metrics,
            timestamp=now,
            policy_hash=self._deadlock_config_hash(),
        )
        if not self._can_apply_deadlock_action(now):
            return
        action = self._next_deadlock_action()
        if action is None:
            return
        await self._apply_deadlock_action(action, triggers, now)

    async def _apply_deadlock_action(
        self,
        action: Mapping[str, object],
        triggers: Sequence["DeadlockStatus"],
        now: datetime,
    ) -> None:
        action_name = next(iter(action.keys()), "unknown")
        changes: Dict[str, object] = {}
        if "switch_policy_id" in action:
            changes["policy_switches"] = self._apply_switch_policy_action(action["switch_policy_id"])
        elif "adjust_prob_gate_min" in action:
            changes["prob_gate_adjustments"] = self._apply_prob_gate_adjustment(
                action["adjust_prob_gate_min"],
                triggers,
            )
        elif "enter_safe_mode" in action:
            await self._set_safe_mode("deadlock_policy")
            changes["safe_mode"] = True
        if not changes:
            return
        record_deadlock_action(action_name)
        if self.deadlock_policy.audit_every_action:
            await self.audit_logger.log_deadlock_action(
                action=action_name,
                timestamp=now,
                payload=changes,
                policy_hash=self._deadlock_config_hash(),
            )
        self._deadlock_action_state["last_action_ts"] = now
        self._deadlock_action_state["actions_taken_today"] = int(
            self._deadlock_action_state.get("actions_taken_today", 0)
        ) + 1
        self._deadlock_action_state["next_index"] = int(self._deadlock_action_state.get("next_index", 0)) + 1

    def _apply_switch_policy_action(self, payload: object) -> List[Dict[str, str]]:
        if not isinstance(payload, Mapping):
            return []
        src = payload.get("from") or payload.get("src")
        dest = payload.get("to") or payload.get("target") or payload.get("id") or payload.get("policy_id")
        symbols = payload.get("symbols")
        target_symbols = None
        if isinstance(symbols, (list, tuple, set)):
            target_symbols = {str(s) for s in symbols}
        if not dest:
            return []
        changes: List[Dict[str, str]] = []
        for cfg in self.config.trading_models:
            if target_symbols and cfg.symbol not in target_symbols:
                continue
            before = self._policy_overrides.get(cfg.symbol, cfg.policy_id or cfg.model)
            if src and before != src:
                continue
            if before == dest:
                continue
            self._policy_overrides[cfg.symbol] = dest
            cfg.policy_id = dest
            changes.append({"symbol": cfg.symbol, "from": str(before), "to": str(dest)})
        return changes

    def _apply_prob_gate_adjustment(
        self,
        payload: object,
        triggers: Sequence["DeadlockStatus"],
    ) -> List[Dict[str, object]]:
        if not isinstance(payload, Mapping):
            return []
        step = payload.get("step") or payload.get("delta") or self.deadlock_policy.adjust_step
        floor = payload.get("floor", self.deadlock_policy.adjust_floor)
        if step is None:
            return []
        try:
            step_value = float(step)
        except (TypeError, ValueError):
            return []
        floor_value: Optional[float] = None
        try:
            if floor is not None:
                floor_value = float(floor)
        except (TypeError, ValueError):
            floor_value = None
        updates: List[Dict[str, object]] = []
        for status in triggers:
            current = self._prob_gate_overrides.get(status.symbol, status.prob_gate_min_used)
            if current is None:
                continue
            new_threshold = current - step_value
            if floor_value is not None:
                new_threshold = max(new_threshold, floor_value)
            new_threshold = max(new_threshold, 0.0)
            if new_threshold >= current:
                continue
            self._prob_gate_overrides[status.symbol] = new_threshold
            updates.append({"symbol": status.symbol, "from": current, "to": new_threshold})
        if updates:
            self._manifest_cache.clear()
        return updates

    async def start(self) -> None:
        if self._running:
            return
        configured_models = ", ".join(
            f"{cfg.model}@{cfg.symbol}" for cfg in self.config.trading_models
        ) or "<none>"
        logger.info(
            "Starting trading service (models=%s, dry_run=%s)",
            configured_models,
            self.config.dry_run,
        )
        shadow_map = ", ".join(
            f"{cfg.symbol}:{'shadow' if cfg.shadow_mode else 'live'}" for cfg in self.config.trading_models
        ) or "<none>"
        logger.info(
            "Shadow routing (default=%s, overrides=%s): %s",
            self.config.shadow_mode_default,
            ",".join(self.config.shadow_symbols or []) or "<none>",
            shadow_map,
        )
        self._redis = self._create_queue_client()
        try:
            await self._redis.ping()
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._redis.close()
            self._redis = None
            raise RuntimeError(
                f"Failed to connect to Redis decision queue at {self.config.decision_queue_url}"
            ) from exc
        await self._reset_stale_positions()
        await self._warm_last_processed_ts()
        if not self.config.dry_run:
            with contextlib.suppress(Exception):
                await self._reconcile_once(startup=True)
        self._running = True
        if not self.config.dry_run and self.config.reconcile_interval_seconds > 0:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Trading poll loop started (queue=%s)", self.config.decision_queue_key)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        if self._reconcile_task:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        if self._redis:
            await self._redis.close()
            self._redis = None
        await self.executor.close()
        await self.state_store.flush()
        await self.audit_logger.close()
        await self.state_store.close()

    async def _reset_stale_positions(self) -> None:
        minutes = self.config.state_stale_minutes
        if not self.config.dry_run or minutes is None:
            return
        try:
            minutes_float = float(minutes)
        except (TypeError, ValueError):
            return
        if minutes_float <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes_float)
        dirty = False
        for model_cfg in self.config.trading_models:
            state = self.state_store.get(model_cfg.state_key)
            last_ts = state.last_timestamp
            stale = last_ts is not None and last_ts < cutoff
            if state.in_position and (last_ts is None or stale):
                logger.info(
                    "Resetting stale position for %s %s (last_ts=%s cutoff=%s)",
                    model_cfg.model,
                    model_cfg.symbol,
                    last_ts.isoformat() if last_ts else "<none>",
                    cutoff.isoformat(),
                )
                state.in_position = False
                state.entry_ts = None
                state.hold_until = None
                state.bars_in_position = 0
                state.metadata.pop("open_price", None)
                state.metadata.pop("open_amount", None)
                state.metadata.pop("open_side", None)
                set_position_active(model_cfg.model, model_cfg.symbol, False)
                self.state_store.update(model_cfg.state_key, state)
                dirty = True
            elif not state.in_position and stale and state.hold_until is not None and state.hold_until < cutoff:
                logger.info(
                    "Clearing stale hold_until for %s %s (hold_until=%s cutoff=%s)",
                    model_cfg.model,
                    model_cfg.symbol,
                    state.hold_until.isoformat() if state.hold_until else "<none>",
                    cutoff.isoformat(),
                )
                state.hold_until = None
                state.bars_in_position = 0
                self.state_store.update(model_cfg.state_key, state)
                dirty = True
        if dirty:
            await self.state_store.flush()

    async def _reconnect_redis(self) -> None:
        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.close()
        if not self._running:
            self._redis = None
            return
        client = self._create_queue_client()
        try:
            await client.ping()
        except Exception as exc:
            logger.error(
                "Redis reconnect failed for %s: %s",
                self.config.decision_queue_url,
                exc,
            )
            with contextlib.suppress(Exception):
                await client.close()
            self._redis = None
            raise
        self._redis = client

    async def _read_last_processed_ts(self, state_key: str) -> Optional[datetime]:
        if not self.config.last_timestamp_hash or self._redis is None:
            return None
        try:
            raw = await self._redis.hget(self.config.last_timestamp_hash, state_key)
        except Exception as exc:
            logger.warning("Failed to read last processed timestamp for %s: %s", state_key, exc)
            return None
        if not raw:
            return None
        return _parse_timestamp(raw)

    async def _write_last_processed_ts(self, state_key: str, ts: datetime) -> None:
        if not self.config.last_timestamp_hash or self._redis is None or ts is None:
            return
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        payload = ts.astimezone(timezone.utc).isoformat()
        try:
            await self._redis.hset(self.config.last_timestamp_hash, state_key, payload)
        except Exception as exc:
            logger.warning("Failed to persist last processed timestamp for %s: %s", state_key, exc)

    async def _clear_last_processed_ts(self, state_key: str) -> None:
        if not self.config.last_timestamp_hash or self._redis is None:
            return
        try:
            await self._redis.hdel(self.config.last_timestamp_hash, state_key)
        except Exception as exc:
            logger.warning("Failed to clear last processed timestamp for %s: %s", state_key, exc)

    async def _warm_last_processed_ts(self) -> None:
        if not self.config.last_timestamp_hash or self._redis is None:
            return
        grace_bars = max(0, int(self.config.last_timestamp_grace_bars))
        if grace_bars <= 0:
            return
        now = datetime.now(timezone.utc)
        for model_cfg in self.config.trading_models:
            last_ts = await self._read_last_processed_ts(model_cfg.state_key)
            if last_ts is None:
                continue
            lag_seconds = max(0.0, (now - last_ts).total_seconds())
            bar_seconds = max(1, model_cfg.bar_seconds)
            grace_seconds = bar_seconds * grace_bars
            if lag_seconds < grace_seconds:
                continue
            await self._clear_last_processed_ts(model_cfg.state_key)
            logger.info(
                "Cleared last processed timestamp for %s after %.0fs downtime (grace=%ss, bars=%d)",
                model_cfg.state_key,
                lag_seconds,
                grace_seconds,
                grace_bars,
            )

    async def _record_queue_depth(self) -> None:
        if self._redis is None:
            return
        try:
            depth = await self._redis.llen(self.config.decision_queue_key)
        except Exception as exc:
            logger.debug(
                "Failed to sample decision queue depth for %s: %s",
                self.config.decision_queue_key,
                exc,
            )
            return
        record_decision_queue_depth(self.config.decision_queue_key, int(depth))

    def _spawn_task(self, coro: asyncio.Future) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))
        return task

    def _is_safe_mode_active(self) -> bool:
        env_flag = os.getenv(SAFE_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        return bool(env_flag or self._safe_mode_reason)

    async def _set_safe_mode(self, reason: str) -> None:
        if self._safe_mode_reason == reason:
            return
        self._safe_mode_reason = reason
        os.environ[SAFE_MODE_ENV] = "1"
        record_safe_mode(reason, True)
        await self.audit_logger.log_safe_mode(reason=reason, active=True)

    async def _clear_safe_mode(self) -> None:
        if self._safe_mode_reason is None:
            return
        reason = self._safe_mode_reason
        self._safe_mode_reason = None
        os.environ.pop(SAFE_MODE_ENV, None)
        record_safe_mode(reason, False)
        await self.audit_logger.log_safe_mode(reason=reason, active=False)

    @staticmethod
    def _mark_pending_entry(
        state: PositionState,
        *,
        intent_id: str,
        probability: float,
        price: Optional[float],
        amount: Optional[float],
        reason: Optional[str] = None,
        spread_bps: Optional[float] = None,
    ) -> None:
        state.metadata["pending_entry_intent_id"] = intent_id
        state.metadata["pending_entry_prob"] = f"{probability:.10f}"
        if price is not None:
            state.metadata["pending_entry_price"] = f"{float(price):.10f}"
        if amount is not None:
            state.metadata["pending_entry_amount"] = f"{float(amount):.10f}"
        if reason:
            state.metadata["pending_entry_reason"] = str(reason)
        if spread_bps is not None:
            state.metadata["pending_entry_spread_bps"] = f"{float(spread_bps):.4f}"

    @staticmethod
    def _mark_pending_exit(
        state: PositionState,
        *,
        intent_id: str,
        probability: float,
        price: Optional[float],
        amount: Optional[float],
        reason: Optional[str] = None,
        spread_bps: Optional[float] = None,
        trigger: Optional[str] = None,
    ) -> None:
        state.metadata["pending_exit_intent_id"] = intent_id
        state.metadata["pending_exit_prob"] = f"{probability:.10f}"
        if price is not None:
            state.metadata["pending_exit_price"] = f"{float(price):.10f}"
        if amount is not None:
            state.metadata["pending_exit_amount"] = f"{float(amount):.10f}"
        if reason:
            state.metadata["pending_exit_reason"] = str(reason)
        if spread_bps is not None:
            state.metadata["pending_exit_spread_bps"] = f"{float(spread_bps):.4f}"
        if trigger:
            state.metadata["pending_exit_trigger"] = trigger

    @staticmethod
    def _clear_pending_flags(state: PositionState, *, kind: str) -> None:
        if kind == "entry":
            for key in (
                "pending_entry_intent_id",
                "pending_entry_prob",
                "pending_entry_price",
                "pending_entry_amount",
                "pending_entry_reason",
                "pending_entry_spread_bps",
            ):
                state.metadata.pop(key, None)
        elif kind == "exit":
            for key in (
                "pending_exit_intent_id",
                "pending_exit_prob",
                "pending_exit_price",
                "pending_exit_amount",
                "pending_exit_reason",
                "pending_exit_spread_bps",
                "pending_exit_trigger",
            ):
                state.metadata.pop(key, None)

    @staticmethod
    def _compute_open_notional(state: PositionState, price_hint: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        qty = None
        try:
            qty_val = state.metadata.get("open_amount")
            if qty_val is not None:
                qty = float(qty_val)
        except (TypeError, ValueError):
            qty = None
        price = price_hint
        if price is None:
            try:
                price_val = state.metadata.get("open_price")
                if price_val is not None:
                    price = float(price_val)
            except (TypeError, ValueError):
                price = None
        if qty is None or qty <= 0 or price is None or price <= 0:
            return None, qty
        return qty * price, qty

    @staticmethod
    def _compute_pending_entry_notional(state: PositionState, price_hint: Optional[float]) -> Optional[float]:
        """
        Estimate pending entry exposure so portfolio-level caps account for in-flight intents.
        """
        if state.in_position:
            return None
        if not state.metadata.get("pending_entry_intent_id"):
            return None
        amount = None
        price = price_hint
        notional = None
        try:
            amt_val = state.metadata.get("pending_entry_amount")
            if amt_val is not None:
                amount = float(amt_val)
        except (TypeError, ValueError):
            amount = None
        try:
            price_val = state.metadata.get("pending_entry_price")
            if price_val is not None:
                price = float(price_val)
        except (TypeError, ValueError):
            price = price_hint
        try:
            notional_val = state.metadata.get("pending_entry_notional")
            if notional_val is not None:
                notional = float(notional_val)
        except (TypeError, ValueError):
            notional = None
        if notional is None and amount is not None and amount > 0 and price and price > 0:
            notional = amount * price
        if notional is None or notional <= 0:
            return None
        return notional

    def _equity_scaled_notional(self, model_cfg: TradingModelConfig) -> Optional[float]:
        limits = self.risk_limits or {}
        mode = str(limits.get("sizing_mode") or "").lower()
        if mode != "equity_fraction":
            return model_cfg.order_notional
        base_notional = model_cfg.order_notional or 0.0
        if base_notional <= 0:
            return None
        try:
            initial_capital = float(
                limits.get("initial_capital_usd")
                or limits.get("capital")
                or self._risk_capital
                or 0.0
            )
        except Exception:
            initial_capital = self._risk_capital
        equity = max(self._risk_equity, initial_capital)
        scale = equity / initial_capital if initial_capital > 0 else 1.0
        target = base_notional * scale
        try:
            equity_fraction = float(limits.get("equity_fraction") or 0.0)
        except Exception:
            equity_fraction = 0.0
        try:
            max_fraction = float(limits.get("max_equity_fraction") or equity_fraction or 0.0)
        except Exception:
            max_fraction = equity_fraction or 0.0
        if max_fraction > 0:
            target = min(target, equity * max_fraction)
        try:
            step_val = float(limits.get("compounding_step_usd") or 0.0)
        except Exception:
            step_val = 0.0
        if step_val > 0:
            target = math.floor(target / step_val) * step_val
        min_notional = None
        try:
            min_notional = limits.get("min_trade_notional")
        except Exception:
            min_notional = None
        try:
            sym_min = ((limits.get("symbols") or {}).get(model_cfg.symbol, {}) or {}).get("min_trade_notional")
            if sym_min is not None:
                min_notional = sym_min
        except Exception:
            pass
        if min_notional is not None:
            try:
                target = max(target, float(min_notional))
            except Exception:
                pass
        per_symbol_cfg: Dict[str, object] = {}
        try:
            per_symbol_cfg = (limits.get("symbols") or {}).get(model_cfg.symbol, {}) or {}
        except Exception:
            per_symbol_cfg = {}
        caps = [
            per_symbol_cfg.get("max_symbol_notional"),
            limits.get("max_notional_per_symbol_usd"),
            limits.get("max_notional_per_symbol"),
            limits.get("max_symbol_notional"),
        ]
        cap_val: Optional[float] = None
        for cap in caps:
            try:
                if cap is not None and float(cap) > 0:
                    cap_val = float(cap) if cap_val is None else min(cap_val, float(cap))
            except Exception:
                continue
        if cap_val is not None:
            target = min(target, cap_val)
        try:
            total_cap = float(limits.get("max_total_notional") or 0.0)
            if total_cap > 0:
                target = min(target, total_cap)
        except Exception:
            pass
        return target

    def _compute_stop_loss_pct(self, *, base_stop_loss: Optional[float], item: Dict[str, object]) -> Optional[float]:
        """
        Compute a bar-aware stop-loss using optional volatility scaling and a hard cap.
        """
        hard_cap = _safe_float(self.risk_limits.get("hard_stop_loss_pct"))
        min_stop = _safe_float(self.risk_limits.get("min_stop_loss_pct"))
        vol_mult = _safe_float(self.risk_limits.get("vol_stop_rvol_mult"))
        vol_value = _extract_volatility_from_item(item)

        stop = base_stop_loss
        if stop is None or stop <= 0:
            stop = min_stop
        if vol_value is not None and vol_mult is not None and vol_mult > 0:
            try:
                vol_stop = float(vol_value) * float(vol_mult)
                if stop is None or vol_stop > stop:
                    stop = vol_stop
            except Exception:
                pass
        if hard_cap is not None and hard_cap > 0 and stop is not None:
            stop = min(stop, hard_cap)
        if stop is not None and stop > 0:
            return float(stop)
        return None

    def _exit_max_spread_bps(self, model_cfg: "TradingModelConfig") -> float:
        base = _safe_float(getattr(model_cfg, "max_spread_bps", None), 0.0) or 0.0
        limit = float(base)
        symbol = getattr(model_cfg, "symbol", None)
        if symbol:
            try:
                sym_cfg = (self.risk_limits.get("symbols") or {}).get(symbol, {}) or {}
                sym_limit = _safe_float(sym_cfg.get("max_spread_bps"))
                if sym_limit is not None and sym_limit > limit:
                    limit = float(sym_limit)
            except Exception:
                pass
        global_limit = _safe_float(self.risk_limits.get("halt_if_spread_bps_gt"))
        if global_limit is not None and global_limit > limit:
            limit = float(global_limit)
        return float(limit) if limit > 0 else float(base)

    def _prune_order_times(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=1)
        while self._risk_order_times and self._risk_order_times[0] < cutoff:
            self._risk_order_times.popleft()

    def _record_order_timestamp(self, ts: datetime) -> None:
        self._risk_order_times.append(ts)
        self._prune_order_times(ts)

    def _record_turnover(self, ts: datetime, notional: Optional[float]) -> None:
        if notional is None:
            return
        day_key = ts.date().isoformat()
        self._risk_turnover[day_key] += abs(float(notional))

    def _current_turnover(self, ts: datetime) -> float:
        return float(self._risk_turnover.get(ts.date().isoformat(), 0.0))

    def _update_risk_pnl(self, ts: datetime, pnl: float) -> None:
        if pnl == 0.0:
            return
        day_key = ts.date().isoformat()
        self._risk_daily_pnl[day_key] += pnl
        self._risk_equity += pnl
        if self._risk_equity > self._risk_equity_peak:
            self._risk_equity_peak = self._risk_equity
        if pnl < 0:
            self._risk_last_loss_ts = ts

    def _get_loss_guard_cfg(self, symbol: Optional[str]) -> Dict[str, object]:
        base: Dict[str, object] = {}
        try:
            base = dict(self.risk_limits.get("loss_guard") or {})
        except Exception:
            base = {}
        sym_cfg = {}
        try:
            sym_cfg = ((self.risk_limits.get("symbols") or {}).get(symbol or "", {}) or {}).get("loss_guard") or {}
        except Exception:
            sym_cfg = {}
        if isinstance(sym_cfg, dict):
            for key, value in sym_cfg.items():
                if value is not None:
                    base[key] = value
        return base

    def _loss_guard_status(
        self,
        state: PositionState,
        ts: datetime,
        cfg: Mapping[str, object],
    ) -> Tuple[bool, int, Optional[datetime]]:
        """
        Determine if the loss-streak guard is active for this position.
        Returns (is_active, streak_count, active_until).
        """
        threshold = 0
        cooldown_minutes = 0
        try:
            threshold = int(cfg.get("max_consecutive_losses") or 0)
        except Exception:
            threshold = 0
        try:
            cooldown_minutes = int(cfg.get("cooldown_minutes") or 0)
        except Exception:
            cooldown_minutes = 0
        enabled = True
        try:
            enabled_val = cfg.get("enabled", True)
            if isinstance(enabled_val, str):
                enabled = enabled_val.strip().lower() not in {"false", "0", "no", "off"}
            else:
                enabled = bool(enabled_val)
        except Exception:
            enabled = True
        if not enabled:
            threshold = 0
        count = 0
        try:
            count = int(state.metadata.get("loss_streak_count") or 0)
        except Exception:
            count = 0
        active_until = _parse_timestamp(state.metadata.get("loss_streak_active_until"))
        if active_until and ts > active_until:
            # Expired guard window; clear it.
            state.metadata.pop("loss_streak_active_until", None)
            active_until = None
        active = False
        if threshold > 0 and count >= threshold:
            active = True
            if cooldown_minutes > 0 and active_until is None:
                active_until = ts + timedelta(minutes=cooldown_minutes)
                state.metadata["loss_streak_active_until"] = active_until.isoformat()
        if active_until and ts <= active_until:
            active = True
        return active, count, active_until

    def _update_loss_streak_after_exit(
        self,
        state: PositionState,
        ts: datetime,
        pnl_value: float,
        cfg: Mapping[str, object],
    ) -> None:
        """
        Update per-position loss streak counters after a realized exit.
        """
        enabled = True
        try:
            enabled_val = cfg.get("enabled", True)
            if isinstance(enabled_val, str):
                enabled = enabled_val.strip().lower() not in {"false", "0", "no", "off"}
            else:
                enabled = bool(enabled_val)
        except Exception:
            enabled = True
        if not enabled:
            return

        threshold = 0
        cooldown_minutes = 0
        try:
            threshold = int(cfg.get("max_consecutive_losses") or 0)
        except Exception:
            threshold = 0
        try:
            cooldown_minutes = int(cfg.get("cooldown_minutes") or 0)
        except Exception:
            cooldown_minutes = 0
        reset_on_profit = True
        try:
            reset_on_profit = bool(cfg.get("reset_after_profit", True))
        except Exception:
            reset_on_profit = True

        dirty = False
        if pnl_value < 0:
            count = 0
            try:
                count = int(state.metadata.get("loss_streak_count") or 0)
            except Exception:
                count = 0
            count += 1
            state.metadata["loss_streak_count"] = str(count)
            state.metadata["loss_streak_last_ts"] = ts.isoformat()
            active_until = _parse_timestamp(state.metadata.get("loss_streak_active_until"))
            if threshold > 0 and count >= threshold and active_until is None and cooldown_minutes > 0:
                state.metadata["loss_streak_active_until"] = (ts + timedelta(minutes=cooldown_minutes)).isoformat()
            dirty = True
        else:
            if reset_on_profit and (
                state.metadata.get("loss_streak_count") or state.metadata.get("loss_streak_active_until")
            ):
                dirty = True
            if reset_on_profit:
                state.metadata["loss_streak_count"] = "0"
                state.metadata.pop("loss_streak_active_until", None)
                state.metadata["loss_streak_last_ts"] = ts.isoformat()
        if dirty:
            state.touch(ts)

    def _loss_guard_entry_params(
        self,
        state: PositionState,
        ts: datetime,
        entry_threshold: float,
        cfg: Mapping[str, object],
    ) -> Tuple[bool, float, float, int]:
        """
        Compute entry-time adjustments when a loss streak is active.
        Returns (guard_active, required_prob, notional_scale, streak_count).
        """
        active, count, _ = self._loss_guard_status(state, ts, cfg)
        prob_buffer = 0.0
        notional_scale = 1.0
        try:
            prob_buffer = float(cfg.get("prob_buffer") or 0.0)
        except Exception:
            prob_buffer = 0.0
        try:
            notional_scale = float(cfg.get("notional_scale") or cfg.get("notional_scale_factor") or 1.0)
        except Exception:
            notional_scale = 1.0
        if notional_scale <= 0:
            notional_scale = 1.0
        required_prob = entry_threshold
        if active and prob_buffer > 0:
            required_prob = entry_threshold + prob_buffer
        return active, required_prob, notional_scale, count


    async def _get_symbol_limits(self, model_cfg: TradingModelConfig) -> Dict[str, object]:
        symbol_cfg = (self.risk_limits.get("symbols") or {}).get(model_cfg.symbol, {}) if self.risk_limits else {}
        limits: Dict[str, object] = {
            "min_trade_notional": symbol_cfg.get("min_trade_notional", self.risk_limits.get("min_trade_notional")),
            "qty_step": symbol_cfg.get("qty_step"),
            "price_tick": symbol_cfg.get("price_tick"),
            "max_position_age_minutes": symbol_cfg.get("max_position_age_minutes"),
        }
        key = (model_cfg.exchange, model_cfg.symbol)
        if key not in self._market_cache:
            market_info = {}
            if hasattr(self.executor, "get_market_info"):
                try:
                    market_info = await self.executor.get_market_info(model_cfg.exchange, model_cfg.symbol)
                except Exception as exc:
                    logger.debug(
                        "Failed to fetch market metadata for %s %s: %s", model_cfg.exchange, model_cfg.symbol, exc
                    )
            self._market_cache[key] = market_info
        market_info = self._market_cache.get(key, {}) or {}
        limits.setdefault("min_trade_notional", _extract_min_cost(market_info))
        qty_step = limits.get("qty_step")
        price_tick = limits.get("price_tick")
        if qty_step is None:
            limits["qty_step"] = _extract_qty_step(market_info)
        if price_tick is None:
            limits["price_tick"] = _extract_price_tick(market_info)
        min_amount = _extract_min_amount(market_info)
        if min_amount is not None and (limits.get("min_amount") is None):
            limits["min_amount"] = min_amount
        return limits

    async def _build_symbol_state(
        self,
        model_cfg: TradingModelConfig,
        state: PositionState,
        *,
        ts: datetime,
        price: Optional[float],
        item: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        limits = await self._get_symbol_limits(model_cfg)
        open_notional, open_qty = self._compute_open_notional(state, price)
        last_exit_ts = int(state.last_exit_ts.timestamp()) if state.last_exit_ts else None
        last_entry_ts = int(state.entry_ts.timestamp()) if state.entry_ts else None
        last_bar_ts = int(ts.timestamp()) if ts else int(state.last_timestamp.timestamp()) if state.last_timestamp else None
        vol_zscore = None
        if isinstance(item, dict):
            features = item.get("features") or {}
            for key in ("vol_zscore", "rvol20_z", "rvol_z"):
                if key in features:
                    try:
                        vol_zscore = float(features[key])
                        break
                    except (TypeError, ValueError):
                        vol_zscore = None

        max_age_min = None
        try:
            max_age_min = limits.get("max_position_age_minutes") or state.metadata.get("max_position_age_minutes")
            if max_age_min is not None:
                max_age_min = float(max_age_min)
        except Exception:
            max_age_min = None

        return {
            "symbol": model_cfg.symbol,
            "in_position": state.in_position,
            "open_notional": open_notional,
            "open_qty": open_qty,
            "last_exit_ts": last_exit_ts,
            "last_entry_ts": last_entry_ts,
            "last_bar_ts": last_bar_ts,
            "missing_price_bars": price is None,
            "vol_zscore": vol_zscore,
            "qty_step": limits.get("qty_step"),
            "price_tick": limits.get("price_tick"),
            "min_trade_notional": limits.get("min_trade_notional"),
            "min_amount": limits.get("min_amount"),
            "max_position_age_seconds": max_age_min * 60 if max_age_min is not None else None,
        }

    def _build_portfolio_state(
        self,
        *,
        ts: datetime,
        price_hints: Dict[str, Optional[float]],
    ) -> Dict[str, object]:
        open_symbols: List[str] = []
        pending_symbols: List[str] = []
        gross = 0.0
        net = 0.0
        pending_total = 0.0
        for model_cfg in self.config.trading_models:
            state = self.state_store.get(model_cfg.state_key)
            price_hint = price_hints.get(model_cfg.symbol)
            exposure, _ = self._compute_open_notional(state, price_hint)
            pending_notional = self._compute_pending_entry_notional(state, price_hint)
            symbol_gross = 0.0
            symbol_net = 0.0
            has_position = False
            if exposure:
                symbol_gross += abs(exposure)
                symbol_net += exposure
                has_position = True
            if pending_notional:
                symbol_gross += abs(pending_notional)
                symbol_net += pending_notional
                pending_total += abs(pending_notional)
                pending_symbols.append(model_cfg.symbol)
                has_position = True
            if has_position:
                open_symbols.append(model_cfg.symbol)
                gross += symbol_gross
                net += symbol_net
        self._prune_order_times(ts)
        turnover_today = self._current_turnover(ts)
        daily_pnl = self._risk_daily_pnl.get(ts.date().isoformat(), 0.0)
        daily_pnl_pct = (daily_pnl / self._risk_capital) if self._risk_capital else 0.0
        drawdown_pct = 0.0
        if self._risk_equity_peak > 0:
            drawdown_pct = max(0.0, (self._risk_equity_peak - self._risk_equity) / self._risk_equity_peak)

        record_portfolio_turnover_estimate(turnover_today / self._risk_capital if self._risk_capital else 0.0)
        record_portfolio_daily_pnl(daily_pnl_pct)
        record_portfolio_drawdown(drawdown_pct)
        record_orders_per_hour("*", len(self._risk_order_times))
        record_concurrent_positions(len(open_symbols))

        kill_switch_active = os.getenv(KILL_SWITCH_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

        return {
            "capital": self._risk_capital,
            "gross_exposure": gross,
            "net_exposure": net,
            "open_symbols": open_symbols,
            "open_positions": len(open_symbols),
            "turnover_1d": turnover_today,
            "orders_last_hour": len(self._risk_order_times),
            "pending_notional": pending_total,
            "pending_symbols": pending_symbols,
            "daily_pnl_pct": daily_pnl_pct,
            "drawdown_pct": drawdown_pct,
            "last_loss_ts": int(self._risk_last_loss_ts.timestamp()) if self._risk_last_loss_ts else None,
            "safe_mode": self._is_safe_mode_active(),
            "kill_switch": kill_switch_active,
            "reconciliation_latched": bool(self._safe_mode_reason),
        }

    async def _filter_stale_decisions(
        self,
        model_cfg: TradingModelConfig,
        state: PositionState,
        items: List[Tuple[datetime, Dict[str, object]]],
        *,
        now: Optional[datetime] = None,
    ) -> List[Tuple[datetime, Dict[str, object]]]:
        if not items:
            return items
        max_age_seconds = None
        try:
            max_age_seconds = (
                int(self.config.decision_max_age_seconds) if self.config.decision_max_age_seconds else None
            )
        except Exception:
            max_age_seconds = None
        now_ts = now or datetime.now(timezone.utc)
        cutoff = state.last_timestamp
        redis_cutoff = await self._read_last_processed_ts(model_cfg.state_key)
        if redis_cutoff is not None and (cutoff is None or redis_cutoff > cutoff):
            cutoff = redis_cutoff
        if cutoff is None and max_age_seconds is None:
            return items
        fresh: List[Tuple[datetime, Dict[str, object]]] = []
        aged_out = 0
        for ts, payload in items:
            if max_age_seconds and ts < (now_ts - timedelta(seconds=max_age_seconds)):
                aged_out += 1
                continue
            if cutoff is not None and ts <= cutoff:
                continue
            fresh.append((ts, payload))
        dropped = len(items) - len(fresh)
        if dropped or aged_out:
            logger.info(
                "Dropping %s stale decision(s) for %s %s (<= %s, aged_out=%s, max_age=%s)",
                dropped or 0,
                model_cfg.model,
                model_cfg.symbol,
                cutoff.isoformat() if cutoff else "<none>",
                aged_out,
                max_age_seconds,
            )
        return fresh

    async def _poll_loop(self) -> None:
        timeout = max(1, int(self.config.redis_poll_timeout))
        iteration = 0
        while self._running:
            iteration += 1
            await self._maybe_price_monitor(datetime.now(timezone.utc))
            if self._redis is None:
                try:
                    await self._reconnect_redis()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await asyncio.sleep(timeout)
                    continue
            try:
                assert self._redis is not None
                await self._record_queue_depth()
                if iteration % 10 == 0:
                    logger.debug("poll_loop heartbeat (queue=%s)", self.config.decision_queue_key)
                result = await self._redis.blpop(
                    self.config.decision_queue_key,
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Redis BLPOP failed: %s", exc)
                try:
                    await self._reconnect_redis()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                await asyncio.sleep(timeout)
                continue
            if result is None:
                continue
            _, payload = result
            try:
                await self._handle_payload(payload)
            except Exception as exc:
                logger.exception("Failed to process decision payload: %s", exc)

    async def _maybe_price_monitor(self, now: datetime) -> None:
        interval = 0
        try:
            interval = int(getattr(self.config, "price_monitor_interval_seconds", 0) or 0)
        except Exception:
            interval = 0
        if interval <= 0:
            return
        last = self._last_price_monitor_ts
        if last is not None and (now - last).total_seconds() < float(interval):
            return
        self._last_price_monitor_ts = now
        await self._price_monitor(now)

    async def _price_monitor(self, now: datetime) -> None:
        """
        Periodically enforce price-based exits even if no new decision payload arrives.

        This is deliberately conservative:
        - Only ever attempts exits (never opens new positions).
        - Disables probability-based exits (stale prob/gate), but still honors stop-loss,
          take-profit, profit-trailing, and max-hold.
        """
        quote_fetch = getattr(self.executor, "fetch_quote", None)
        if not callable(quote_fetch):
            return

        for model_cfg in self.config.trading_models:
            state = self.state_store.get(model_cfg.state_key)
            if not state.in_position:
                continue
            if state.metadata.get("pending_exit_intent_id"):
                continue

            manifest = self._resolve_manifest({}, model_cfg.model, symbol_hint=model_cfg.symbol)
            if manifest is None:
                continue

            policy_id = self._policy_overrides.get(model_cfg.symbol, model_cfg.policy_id or model_cfg.model)
            timeframe = model_cfg.timeframe
            decision_namespace = f"{model_cfg.symbol}:{timeframe}:{policy_id}:{model_cfg.model}"

            entry_price = _safe_float(state.metadata.get("open_price"))
            entry_amount = _safe_float(state.metadata.get("open_amount"))
            if entry_price is None or entry_price <= 0 or entry_amount is None or entry_amount <= 0:
                continue

            quote: Optional[Dict[str, float]] = None
            try:
                quote = await quote_fetch(exchange=model_cfg.exchange, symbol=model_cfg.symbol)
            except Exception:
                quote = None
            if not isinstance(quote, dict):
                continue
            quote_bid = quote.get("bid")
            if quote_bid is None or not math.isfinite(float(quote_bid)) or float(quote_bid) <= 0:
                continue
            current_price = float(quote_bid)
            spread_bps = None
            try:
                spread_val = quote.get("spread_bps")
                spread_bps = float(spread_val) if spread_val is not None else None
            except Exception:
                spread_bps = None

            # Update peak price metadata so profit-trailing stays accurate between decision bars.
            try:
                peak = _safe_float(state.metadata.get("open_peak_price"))
                if peak is None or peak <= 0:
                    peak = entry_price
                peak = max(float(peak), current_price)
                state.metadata["open_peak_price"] = f"{float(peak):.10f}"
            except Exception:
                pass

            min_hold_bars = model_cfg.min_hold_bars_override or manifest.min_hold_bars
            min_hold_seconds = max(1, int(min_hold_bars)) * max(1, int(model_cfg.bar_seconds))
            max_hold_seconds = None
            if model_cfg.max_hold_minutes is not None:
                try:
                    max_hold_seconds = max(1, int(model_cfg.max_hold_minutes) * 60)
                except Exception:
                    max_hold_seconds = None
            if max_hold_seconds is not None and max_hold_seconds < min_hold_seconds:
                max_hold_seconds = min_hold_seconds

            # Use a looser "hard" stop for the price monitor to avoid exiting on transient intrabar wicks.
            base_stop_loss_pct = _safe_float(self.risk_limits.get("hard_stop_loss_pct"))
            if base_stop_loss_pct is None or base_stop_loss_pct <= 0:
                base_stop_loss_pct = _safe_float(getattr(model_cfg, "stop_loss_pct", None))
            if base_stop_loss_pct is None or base_stop_loss_pct <= 0:
                base_stop_loss_pct = _safe_float(self.risk_limits.get("min_stop_loss_pct"), 0.008) or 0.008

            take_profit_pct = _safe_float(getattr(model_cfg, "take_profit_pct", None))
            if take_profit_pct is not None and take_profit_pct <= 0:
                take_profit_pct = None

            profit_trailing_start_pct = _safe_float(getattr(model_cfg, "profit_trailing_start_pct", None))
            if profit_trailing_start_pct is not None and profit_trailing_start_pct <= 0:
                profit_trailing_start_pct = None
            profit_trailing_stop_pct = _safe_float(getattr(model_cfg, "profit_trailing_stop_pct", None))
            if profit_trailing_stop_pct is not None and profit_trailing_stop_pct <= 0:
                profit_trailing_stop_pct = None

            trigger_cfg = TriggerConfig(
                entry_threshold=manifest.entry_threshold,
                exit_threshold=manifest.exit_threshold,
                exit_prob_drop=manifest.exit_prob_drop,
                min_hold_bars=min_hold_bars,
                bar_seconds=model_cfg.bar_seconds,
                long_only=manifest.long_only,
                max_hold_seconds=max_hold_seconds,
                stop_loss_pct=base_stop_loss_pct,
                take_profit_pct=take_profit_pct,
                profit_trailing_start_pct=profit_trailing_start_pct,
                profit_trailing_stop_pct=profit_trailing_stop_pct,
                max_spread_bps=model_cfg.max_spread_bps,
            )

            fee_estimate_bps = _safe_float(self.risk_limits.get("transaction_cost_bps"))
            slippage_estimate_bps = _safe_float(self.risk_limits.get("slippage_bps"))

            active_stop_loss_pct = self._compute_stop_loss_pct(base_stop_loss=base_stop_loss_pct, item={})
            outcome = decide_bar(
                ts=now,
                probability=float(state.last_probability or manifest.entry_threshold),
                gate_pass=bool(state.last_gate) if state.last_gate is not None else True,
                state=state,
                cfg=trigger_cfg,
                current_price=current_price,
                entry_price=entry_price,
                entry_amount=entry_amount,
                spread_bps=spread_bps,
                include_spread_cost=False,
                safe_mode_active=self._is_safe_mode_active(),
                fee_estimate_bps=fee_estimate_bps,
                slippage_estimate_bps=slippage_estimate_bps,
                stop_loss_override=active_stop_loss_pct,
                disable_prob_exits=True,
            )
            try:
                outcome.exit_context["decision_price_source"] = "price_monitor_quote_bid"
                outcome.exit_context["quote_bid"] = quote.get("bid")
                outcome.exit_context["quote_ask"] = quote.get("ask")
                outcome.exit_context["quote_mid"] = quote.get("mid")
                outcome.exit_context["quote_spread_bps"] = quote.get("spread_bps")
            except Exception:
                pass

            if not outcome.should_exit:
                continue

            # Execute exit using the same risk/audit paths as decision-driven exits.
            threshold_for_audit = (
                manifest.exit_threshold
                if outcome.exit_trigger in {"prob_floor", "prob_trailing"}
                else manifest.entry_threshold
            )
            intent_id = _build_order_intent_id(model_cfg, now, "sell")
            portfolio_state = self._build_portfolio_state(ts=now, price_hints={model_cfg.symbol: current_price})
            portfolio_state["orders_last_hour"] = int(portfolio_state.get("orders_last_hour", 0)) + 1
            symbol_state = await self._build_symbol_state(
                model_cfg,
                state,
                ts=now,
                price=current_price,
                item=None,
            )
            desired_qty = entry_amount
            desired_notional = entry_price * entry_amount
            risk_result = assess_and_adjust_order(
                symbol=model_cfg.symbol,
                action="EXIT_LONG",
                desired_notional=desired_notional,
                desired_qty=desired_qty,
                price=current_price,
                spread_bps=spread_bps,
                now_ts=int(now.timestamp()),
                portfolio_state=portfolio_state,
                symbol_state=symbol_state,
                risk_cfg=self.risk_limits,
            )
            if not risk_result.get("allowed"):
                reason = risk_result.get("block_reason") or "risk_blocked"
                record_risk_blocked(model_cfg.symbol, reason)
                record_skip_reason(model_cfg.model, model_cfg.symbol, reason)
                state.metadata["last_exit_reason"] = reason
                state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timeframe=timeframe,
                    timestamp=now,
                    side="sell",
                    gate_pass=bool(state.last_gate) if state.last_gate is not None else True,
                    probability=float(state.last_probability or 0.0),
                    threshold=threshold_for_audit,
                    decision=OrderDecision(
                        executed=False,
                        price_used=current_price,
                        amount=desired_qty,
                        reason=reason,
                        blocked_reason=reason,
                        order_intent_id=intent_id,
                        notional=risk_result.get("final_notional"),
                    ),
                    policy_id=policy_id,
                    risk_payload=risk_result,
                    decision_namespace=decision_namespace,
                    extra=self._build_exit_audit_payload(
                        outcome=outcome,
                        entry_price=entry_price,
                        entry_amount=entry_amount,
                        decision=None,
                        current_price=current_price,
                        spread_bps=spread_bps,
                        fee_estimate_bps=fee_estimate_bps,
                        slippage_estimate_bps=slippage_estimate_bps,
                    ),
                )
                continue

            final_notional = risk_result.get("final_notional", desired_notional)
            final_qty = risk_result.get("final_qty", desired_qty)
            for clip_reason in risk_result.get("clip_reasons", []):
                record_risk_clipped(model_cfg.symbol, clip_reason)
            self._record_turnover(now, final_notional)
            self._record_order_timestamp(now)
            decision = await self.executor.submit(
                exchange=model_cfg.exchange,
                symbol=model_cfg.symbol,
                side="sell",
                order_amount=final_qty,
                order_notional=final_notional,
                max_spread_bps=self._exit_max_spread_bps(model_cfg),
                shadow_mode=bool(model_cfg.shadow_mode),
                order_intent_id=intent_id,
            )
            if decision.order_intent_id is None:
                decision.order_intent_id = intent_id
            if decision.dedup_blocked:
                record_dedup_blocked(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                record_skip_reason(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                state.metadata["last_exit_reason"] = decision.reason or "duplicate_intent"
                state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timeframe=timeframe,
                    timestamp=now,
                    side="sell",
                    gate_pass=bool(state.last_gate) if state.last_gate is not None else True,
                    probability=float(state.last_probability or 0.0),
                    threshold=threshold_for_audit,
                    decision=decision,
                    policy_id=policy_id,
                    risk_payload=risk_result,
                    decision_namespace=decision_namespace,
                    extra=self._build_exit_audit_payload(
                        outcome=outcome,
                        entry_price=entry_price,
                        entry_amount=entry_amount,
                        decision=decision,
                        current_price=current_price,
                        spread_bps=spread_bps,
                        fee_estimate_bps=fee_estimate_bps,
                        slippage_estimate_bps=slippage_estimate_bps,
                    ),
                )
                continue

            fill_confirmed = self.config.dry_run or decision.intent_status in {IntentStatus.FILLED.value}
            exit_payload = self._build_exit_audit_payload(
                outcome=outcome,
                entry_price=entry_price,
                entry_amount=entry_amount,
                decision=decision,
                current_price=current_price,
                spread_bps=spread_bps,
                fee_estimate_bps=fee_estimate_bps,
                slippage_estimate_bps=slippage_estimate_bps,
            )
            await self.audit_logger.log_trade(
                model=model_cfg.model,
                symbol=model_cfg.symbol,
                timeframe=timeframe,
                timestamp=now,
                side="sell",
                gate_pass=bool(state.last_gate) if state.last_gate is not None else True,
                probability=float(state.last_probability or 0.0),
                threshold=threshold_for_audit,
                decision=decision,
                policy_id=policy_id,
                risk_payload=risk_result,
                decision_namespace=decision_namespace,
                extra=exit_payload,
            )
            if decision.executed and fill_confirmed:
                pnl_value = 0.0
                exit_price = float(decision.price_used or 0.0)
                exit_amount = float(decision.amount or entry_amount or 0.0)
                if entry_price > 0.0 and entry_amount > 0.0 and exit_price > 0.0:
                    qty = min(entry_amount, exit_amount if exit_amount > 0.0 else entry_amount)
                    pnl_value = (exit_price - entry_price) * qty
                    record_realized_pnl(model_cfg.model, model_cfg.symbol, pnl_value)
                    self._update_risk_pnl(now, pnl_value)
                    self._update_loss_streak_after_exit(state, now, pnl_value, self._get_loss_guard_cfg(model_cfg.symbol))
                    state.metadata["last_realized_pnl"] = f"{pnl_value:.10f}"
                state.metadata.pop("open_price", None)
                state.metadata.pop("open_amount", None)
                state.metadata.pop("open_side", None)
                state.metadata.pop("open_entry_prob", None)
                state.metadata.pop("open_peak_price", None)
                state.mark_exit(now)
                state.metadata["last_exit_reason"] = decision.reason or ""
                state.metadata["last_exit_spread_bps"] = (
                    f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                )
                state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                set_position_active(model_cfg.model, model_cfg.symbol, False)
                await self.state_store.flush()
            elif decision.executed:
                self._mark_pending_exit(
                    state,
                    intent_id=intent_id,
                    probability=float(state.last_probability or 0.0),
                    price=decision.price_used,
                    amount=decision.amount,
                    reason=decision.reason,
                    spread_bps=decision.spread_bps,
                    trigger=outcome.exit_trigger,
                )
                await self.state_store.flush()


    def _signed_decisions_required(self) -> bool:
        configured = self.config.require_signed_decisions
        if configured is not None:
            return bool(configured)
        return not bool(self.config.dry_run)

    def _verify_decision_message(self, message: Mapping[str, object]) -> bool:
        secret = self.config.decision_hmac_secret or os.getenv("DECISION_HMAC_SECRET", "")
        require_signature = self._signed_decisions_required()
        if not secret:
            if require_signature:
                logger.error(
                    "Rejecting decision payload because signed decisions are required but no HMAC secret is configured"
                )
                record_skip_reason("unknown", "unknown", "unsigned_decision")
                return False
            return True
        if verify_decision_payload(message, secret):
            return True
        model = str(message.get("model") or "unknown")
        symbol = str(message.get("symbol") or "unknown")
        logger.warning(
            "Rejecting decision payload with missing or invalid signature for model=%s symbol=%s",
            model,
            symbol,
        )
        record_skip_reason(model, symbol, "invalid_decision_signature")
        return False

    async def _handle_payload(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Discarding malformed payload: %s", raw[:80])
            return

        if not self._verify_decision_message(message):
            return

        model_label = str(message.get("model") or "")
        if not model_label:
            logger.debug("Decision payload missing model label")
            return
        symbol_hint = message.get("symbol")
        if isinstance(symbol_hint, str):
            symbol_hint = symbol_hint.strip() or None
        elif symbol_hint is not None:
            symbol_hint = str(symbol_hint)
        model_cfg = self._model_map.get((model_label, symbol_hint)) or self._model_map.get((model_label, None))
        if model_cfg is None:
            logger.info(
                "Ignoring payload for unconfigured model '%s' (symbol=%s)",
                model_label,
                symbol_hint or "<none>",
            )
            return

        expected_policy_id = self._policy_overrides.get(model_cfg.symbol, model_cfg.policy_id or model_label)
        incoming_policy_id = str(message.get("policy_id") or "").strip()
        policy_id = incoming_policy_id or expected_policy_id
        if incoming_policy_id and expected_policy_id and incoming_policy_id != expected_policy_id:
            logger.warning(
                "Policy id mismatch for %s %s (payload=%s, config=%s); using configured policy id",
                model_cfg.model,
                model_cfg.symbol,
                incoming_policy_id,
                expected_policy_id,
            )
            policy_id = expected_policy_id
        incoming_timeframe = str(message.get("timeframe") or "").strip()
        timeframe = incoming_timeframe or model_cfg.timeframe
        if incoming_timeframe and incoming_timeframe != model_cfg.timeframe:
            logger.warning(
                "Timeframe mismatch for %s %s (payload=%s, config=%s); skipping payload",
                model_cfg.model,
                model_cfg.symbol,
                incoming_timeframe,
                model_cfg.timeframe,
            )
            return
        decision_namespace = message.get("decision_namespace") or (
            f"{model_cfg.symbol}:{timeframe}:{policy_id}:{model_cfg.model}"
        )
        shadow_mode_flag = bool(model_cfg.shadow_mode or message.get("shadow_mode"))

        manifest = self._resolve_manifest(message, model_label, symbol_hint=model_cfg.symbol)
        if manifest is None:
            logger.warning("Unable to resolve manifest for model '%s'; skipping", model_label)
            return

        items = message.get("items")
        if not items:
            ts = message.get("timestamp")
            if ts is not None:
                synthetic = {
                    "timestamp": ts,
                    "probability": message.get("probability"),
                    "gate_pass": message.get("gate_pass"),
                }
                if "features" in message:
                    synthetic["features"] = message["features"]
                items = [synthetic]
            else:
                items = []
        if not isinstance(items, list):
            logger.debug("Payload items not a list for model '%s'", model_label)
            return

        parsed: List[Tuple[datetime, Dict[str, object]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ts = _parse_timestamp(item.get("timestamp"))
            if ts is None:
                continue
            parsed.append((ts, item))
        if not parsed:
            return
        parsed.sort(key=lambda pair: pair[0])

        state = self.state_store.get(model_cfg.state_key)
        set_position_active(model_cfg.model, model_cfg.symbol, state.in_position)
        parsed = await self._filter_stale_decisions(
            model_cfg,
            state,
            parsed,
            now=datetime.now(timezone.utc),
        )
        if not parsed:
            return

        min_hold_bars = int(model_cfg.min_hold_bars_override or manifest.min_hold_bars or 1)
        min_hold_seconds = max(1, min_hold_bars) * model_cfg.bar_seconds
        max_hold_minutes_cfg = model_cfg.max_hold_minutes
        max_hold_seconds: Optional[int] = None
        # Enforce the stricter of model-config max hold and symbol risk limit (if present).
        symbol_limits = (self.risk_limits.get("symbols") or {}).get(model_cfg.symbol or "", {})
        max_age_minutes_limit = None
        try:
            max_age_minutes_limit = float(symbol_limits.get("max_position_age_minutes")) if symbol_limits else None
        except Exception:
            max_age_minutes_limit = None
        if max_hold_minutes_cfg is not None:
            try:
                max_hold_seconds = max(1, int(max_hold_minutes_cfg) * 60)
            except (TypeError, ValueError):
                max_hold_seconds = None
            if max_hold_seconds is not None and max_hold_seconds < min_hold_seconds:
                logger.warning(
                    "Configured max_hold_minutes (%s) for %s %s shorter than min_hold window; using min_hold instead.",
                    max_hold_minutes_cfg,
                    model_cfg.model,
                    model_cfg.symbol,
                )
                max_hold_seconds = min_hold_seconds
        if max_age_minutes_limit is not None:
            try:
                limit_seconds = max(1, int(max_age_minutes_limit * 60))
                max_hold_seconds = limit_seconds if max_hold_seconds is None else min(max_hold_seconds, limit_seconds)
            except Exception:
                pass
        entry_threshold = float(manifest.entry_threshold)
        exit_threshold = float(manifest.exit_threshold)
        exit_prob_drop = float(manifest.exit_prob_drop)
        dirty = False
        try:
            base_stop_loss_pct = (
                float(model_cfg.stop_loss_pct) if getattr(model_cfg, "stop_loss_pct", None) is not None else None
            )
        except (TypeError, ValueError):
            base_stop_loss_pct = None
        # Fail-safe: apply a small stop-loss if none configured or it was disabled (<=0).
        if base_stop_loss_pct is None or base_stop_loss_pct <= 0.0:
            base_stop_loss_pct = _safe_float(self.risk_limits.get("min_stop_loss_pct"), 0.008)
            if base_stop_loss_pct is None or base_stop_loss_pct <= 0.0:
                base_stop_loss_pct = 0.008
        try:
            take_profit_pct = (
                float(model_cfg.take_profit_pct)
                if getattr(model_cfg, "take_profit_pct", None) is not None
                else None
            )
        except (TypeError, ValueError):
            take_profit_pct = None
        if take_profit_pct is not None and take_profit_pct <= 0.0:
            take_profit_pct = None
        try:
            profit_trailing_start_pct = (
                float(model_cfg.profit_trailing_start_pct)
                if getattr(model_cfg, "profit_trailing_start_pct", None) is not None
                else None
            )
        except (TypeError, ValueError):
            profit_trailing_start_pct = None
        if profit_trailing_start_pct is not None and profit_trailing_start_pct <= 0.0:
            profit_trailing_start_pct = None
        try:
            profit_trailing_stop_pct = (
                float(model_cfg.profit_trailing_stop_pct)
                if getattr(model_cfg, "profit_trailing_stop_pct", None) is not None
                else None
            )
        except (TypeError, ValueError):
            profit_trailing_stop_pct = None
        if profit_trailing_stop_pct is not None and profit_trailing_stop_pct <= 0.0:
            profit_trailing_stop_pct = None

        trigger_cfg = TriggerConfig(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            exit_prob_drop=exit_prob_drop,
            min_hold_bars=min_hold_bars,
            bar_seconds=model_cfg.bar_seconds,
            long_only=manifest.long_only,
            max_hold_seconds=max_hold_seconds,
            stop_loss_pct=base_stop_loss_pct,
            take_profit_pct=take_profit_pct,
            profit_trailing_start_pct=profit_trailing_start_pct,
            profit_trailing_stop_pct=profit_trailing_stop_pct,
            max_spread_bps=model_cfg.max_spread_bps,
        )

        loss_guard_cfg = self._get_loss_guard_cfg(model_cfg.symbol)

        for ts, item in parsed:
            probability = float(item.get("probability") or 0.0)
            gate_pass = bool(item.get("gate_pass"))
            if not state.register_bar(ts):
                continue
            dirty = True
            record_decision_coverage(model_cfg.model, model_cfg.symbol, gate_pass)
            try:
                self.deadlock_monitor.record_gate(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    ts=ts,
                    gate_pass=gate_pass,
                    probability=probability,
                    prob_gate_min=entry_threshold,
                    timeframe=timeframe,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Deadlock monitor gate record failed for %s %s: %s",
                    model_cfg.model,
                    model_cfg.symbol,
                    exc,
                )
            previous_gate = state.last_gate
            state.last_probability = probability
            if previous_gate is None or bool(previous_gate) != gate_pass:
                await self.audit_logger.log_gate_toggle(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timestamp=ts,
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=entry_threshold,
                    previous_gate=previous_gate,
                )
                record_gate_toggle(model_cfg.model, model_cfg.symbol, gate_pass)
            state.last_gate = gate_pass

            stored_entry_prob: Optional[float] = None
            if state.metadata.get("open_entry_prob") is not None:
                try:
                    stored_entry_prob = float(state.metadata["open_entry_prob"])
                except (TypeError, ValueError):
                    stored_entry_prob = None

            entry_price: Optional[float] = None
            if state.metadata.get("open_price") is not None:
                try:
                    entry_price = float(state.metadata["open_price"])
                except (TypeError, ValueError):
                    entry_price = None
                else:
                    if entry_price <= 0.0:
                        entry_price = None
            entry_amount: Optional[float] = None
            if state.metadata.get("open_amount") is not None:
                try:
                    entry_amount = float(state.metadata["open_amount"])
                except (TypeError, ValueError):
                    entry_amount = None
                else:
                    if entry_amount <= 0.0:
                        entry_amount = None
            exit_qty: Optional[float] = None
            feature_price = _extract_price_from_item(item)
            feature_spread_bps = _extract_spread_from_item(item)
            current_price = feature_price
            spread_bps = feature_spread_bps
            decision_price_source = "feature"
            quote: Optional[Dict[str, float]] = None
            quote_fetch = getattr(self.executor, "fetch_quote", None)
            if callable(quote_fetch):
                try:
                    quote = await quote_fetch(exchange=model_cfg.exchange, symbol=model_cfg.symbol)
                except Exception:
                    quote = None
            if isinstance(quote, dict):
                quote_bid = quote.get("bid")
                quote_ask = quote.get("ask")
                quote_spread = quote.get("spread_bps")
                if state.in_position and quote_bid is not None and quote_bid > 0:
                    current_price = float(quote_bid)
                    decision_price_source = "quote_bid"
                elif not state.in_position and quote_ask is not None and quote_ask > 0:
                    current_price = float(quote_ask)
                    decision_price_source = "quote_ask"
                if quote_spread is not None and math.isfinite(float(quote_spread)):
                    spread_bps = float(quote_spread)

            if state.in_position and current_price is not None and current_price > 0:
                try:
                    peak_val = _safe_float(state.metadata.get("open_peak_price"))
                    if peak_val is None or peak_val <= 0:
                        peak_val = entry_price if entry_price is not None and entry_price > 0 else current_price
                    peak_val = max(float(peak_val), float(current_price))
                    state.metadata["open_peak_price"] = f"{float(peak_val):.10f}"
                except Exception:
                    pass
            fee_estimate_bps = None
            try:
                fee_estimate_bps = float(self.risk_limits.get("transaction_cost_bps"))
            except Exception:
                fee_estimate_bps = None
            slippage_estimate_bps = None
            try:
                slippage_estimate_bps = float(self.risk_limits.get("slippage_bps"))
            except Exception:
                slippage_estimate_bps = None
            active_stop_loss_pct = self._compute_stop_loss_pct(
                base_stop_loss=base_stop_loss_pct,
                item=item,
            )

            outcome = decide_bar(
                ts=ts,
                probability=probability,
                gate_pass=gate_pass,
                state=state,
                cfg=trigger_cfg,
                current_price=current_price,
                entry_price=entry_price,
                entry_amount=entry_amount,
                spread_bps=spread_bps,
                include_spread_cost=not decision_price_source.startswith("quote"),
                safe_mode_active=self._is_safe_mode_active(),
                fee_estimate_bps=fee_estimate_bps,
                slippage_estimate_bps=slippage_estimate_bps,
                stop_loss_override=active_stop_loss_pct,
                disable_prob_exits=bool(getattr(model_cfg, "disable_prob_exits", False)),
            )
            try:
                outcome.exit_context["decision_price_source"] = decision_price_source
                outcome.exit_context["price_feature"] = feature_price
                outcome.exit_context["spread_feature_bps"] = feature_spread_bps
                if isinstance(quote, dict):
                    outcome.exit_context["quote_bid"] = quote.get("bid")
                    outcome.exit_context["quote_ask"] = quote.get("ask")
                    outcome.exit_context["quote_mid"] = quote.get("mid")
                    outcome.exit_context["quote_spread_bps"] = quote.get("spread_bps")
            except Exception:
                pass

            if outcome.entry_block_reason and not outcome.should_exit:
                logger.warning(
                    "Entry blocked for %s %s due to %s flag",
                    model_cfg.model,
                    model_cfg.symbol,
                    outcome.entry_block_reason,
                )
                record_skip_reason(model_cfg.model, model_cfg.symbol, outcome.entry_block_reason)
                state.metadata["last_entry_reason"] = outcome.entry_block_reason
                state.metadata["last_entry_trigger"] = outcome.entry_block_reason
                dirty = True
                continue

            if outcome.should_enter:
                record_would_trade(model_cfg.model, model_cfg.symbol, "buy")
                entry_filter_reason = _entry_filter_block_reason(model_cfg, item)
                if entry_filter_reason:
                    record_skip_reason(model_cfg.model, model_cfg.symbol, entry_filter_reason)
                    state.metadata["last_entry_reason"] = entry_filter_reason
                    state.metadata["last_entry_trigger"] = entry_filter_reason
                    state.metadata["entry_filter_blocked_at"] = ts.isoformat()
                    dirty = True
                    blocked_decision = OrderDecision(
                        executed=False,
                        price_used=current_price,
                        amount=None,
                        reason=entry_filter_reason,
                        blocked_reason=entry_filter_reason,
                    )
                    await self.audit_logger.log_trade(
                        model=model_cfg.model,
                        symbol=model_cfg.symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        side="buy",
                        gate_pass=gate_pass,
                        probability=probability,
                        threshold=entry_threshold,
                        decision=blocked_decision,
                        policy_id=policy_id,
                        decision_namespace=decision_namespace,
                        extra={
                            "entry_filter_rsi_min": getattr(model_cfg, "entry_rsi_min", None),
                            "entry_filter_macd_min": getattr(model_cfg, "entry_macd_min", None),
                        },
                    )
                    self._record_deadlock_trade_event(
                        model_cfg,
                        ts,
                        executed=False,
                        reason=entry_filter_reason,
                    )
                    continue
                guard_active, required_prob, notional_scale, loss_count = self._loss_guard_entry_params(
                    state=state,
                    ts=ts,
                    entry_threshold=entry_threshold,
                    cfg=loss_guard_cfg,
                )
                if guard_active and probability < required_prob:
                    reason = "loss_guard"
                    record_skip_reason(model_cfg.model, model_cfg.symbol, reason)
                    state.metadata["last_entry_reason"] = reason
                    state.metadata["last_entry_trigger"] = reason
                    state.metadata["loss_guard_blocked_at"] = ts.isoformat()
                    dirty = True
                    blocked_decision = OrderDecision(
                        executed=False,
                        price_used=current_price,
                        amount=None,
                        reason=reason,
                        blocked_reason=reason,
                    )
                    await self.audit_logger.log_trade(
                        model=model_cfg.model,
                        symbol=model_cfg.symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        side="buy",
                        gate_pass=gate_pass,
                        probability=probability,
                        threshold=entry_threshold,
                        decision=blocked_decision,
                        policy_id=policy_id,
                        risk_payload={"loss_streak_count": loss_count, "loss_guard_active": guard_active},
                        decision_namespace=decision_namespace,
                    )
                    self._record_deadlock_trade_event(model_cfg, ts, executed=False, reason=reason)
                    continue
                intent_id = _build_order_intent_id(model_cfg, ts, "buy")
                portfolio_state = self._build_portfolio_state(ts=ts, price_hints={model_cfg.symbol: current_price})
                portfolio_state["orders_last_hour"] = int(portfolio_state.get("orders_last_hour", 0)) + 1
                symbol_state = await self._build_symbol_state(
                    model_cfg,
                    state,
                    ts=ts,
                    price=current_price,
                    item=item,
                )
                desired_notional = self._equity_scaled_notional(model_cfg)
                desired_qty = model_cfg.order_amount
                if guard_active and notional_scale and notional_scale < 0.999:
                    if desired_notional is not None:
                        desired_notional = float(desired_notional) * float(notional_scale)
                    elif desired_qty is not None:
                        desired_qty = float(desired_qty) * float(notional_scale)
                if desired_notional is None and desired_qty is not None and current_price:
                    desired_notional = float(desired_qty) * float(current_price)
                risk_result = assess_and_adjust_order(
                    symbol=model_cfg.symbol,
                    action="ENTER_LONG",
                    desired_notional=desired_notional or 0.0,
                    desired_qty=desired_qty,
                    price=current_price,
                    spread_bps=spread_bps,
                    now_ts=int(ts.timestamp()),
                    portfolio_state=portfolio_state,
                    symbol_state=symbol_state,
                    risk_cfg=self.risk_limits,
                )
                if not risk_result["allowed"]:
                    reason = risk_result.get("block_reason") or "risk_blocked"
                    record_risk_blocked(model_cfg.symbol, reason)
                    record_skip_reason(model_cfg.model, model_cfg.symbol, reason)
                    state.metadata["last_entry_reason"] = reason
                    self._record_deadlock_trade_event(model_cfg, ts, executed=False, reason=reason)
                    dirty = True
                    await self.audit_logger.log_trade(
                        model=model_cfg.model,
                        symbol=model_cfg.symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        side="buy",
                        gate_pass=gate_pass,
                        probability=probability,
                        threshold=entry_threshold,
                        decision=OrderDecision(
                            executed=False,
                            price_used=current_price,
                            amount=desired_qty,
                            reason=reason,
                            blocked_reason=reason,
                            order_intent_id=intent_id,
                            notional=risk_result.get("final_notional"),
                        ),
                        policy_id=policy_id,
                        risk_payload=risk_result,
                        decision_namespace=decision_namespace,
                    )
                    continue
                final_notional = risk_result.get("final_notional", desired_notional)
                final_qty = risk_result.get("final_qty", desired_qty)
                for clip_reason in risk_result.get("clip_reasons", []):
                    record_risk_clipped(model_cfg.symbol, clip_reason)
                self._record_turnover(ts, final_notional)
                self._record_order_timestamp(ts)
                decision = await self.executor.submit(
                    exchange=model_cfg.exchange,
                    symbol=model_cfg.symbol,
                    side="buy",
                    order_amount=final_qty,
                    order_notional=final_notional,
                    max_spread_bps=model_cfg.max_spread_bps,
                    shadow_mode=shadow_mode_flag,
                    order_intent_id=intent_id,
                )
                if decision.order_intent_id is None:
                    decision.order_intent_id = intent_id
                if decision.dedup_blocked:
                    record_dedup_blocked(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                    record_skip_reason(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                    state.metadata["last_entry_reason"] = decision.reason or "duplicate_intent"
                    self._record_deadlock_trade_event(
                        model_cfg,
                        ts,
                        executed=False,
                        reason=decision.reason or "duplicate_intent",
                    )
                    dirty = True
                    continue
                if decision.shadow_mode:
                    record_shadow_blocked(model_cfg.model, model_cfg.symbol, "buy")
                record_trade_attempt(
                    model_cfg.model,
                    model_cfg.symbol,
                    "buy",
                    decision.executed,
                    decision.price_used,
                    decision.amount,
                )
                trade_executed = bool(decision.executed or decision.shadow_mode)
                self._record_deadlock_trade_event(
                    model_cfg,
                    ts,
                    executed=trade_executed,
                    reason=None if trade_executed else (decision.reason or decision.blocked_reason),
                )
                entry_amount_for_exit = entry_amount
                exit_payload = self._build_exit_audit_payload(
                    outcome=outcome,
                    entry_price=entry_price,
                    entry_amount=entry_amount_for_exit,
                    decision=decision,
                    current_price=current_price,
                    spread_bps=spread_bps,
                    fee_estimate_bps=fee_estimate_bps,
                    slippage_estimate_bps=slippage_estimate_bps,
                )
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    side="buy",
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=entry_threshold,
                    decision=decision,
                    policy_id=policy_id,
                    risk_payload=risk_result,
                    decision_namespace=decision_namespace,
                )
                fill_confirmed = self.config.dry_run or decision.intent_status in {IntentStatus.FILLED.value}
                if decision.executed and fill_confirmed:
                    if decision.price_used is not None:
                        state.metadata["open_price"] = f"{float(decision.price_used):.10f}"
                        state.metadata["open_peak_price"] = f"{float(decision.price_used):.10f}"
                    if decision.amount is not None:
                        state.metadata["open_amount"] = f"{float(decision.amount):.10f}"
                    state.metadata["open_side"] = "long"
                    state.mark_entry(ts, min_hold_seconds)
                    state.metadata["open_entry_prob"] = f"{probability:.10f}"
                    state.metadata["last_entry_reason"] = decision.reason or ""
                    state.metadata["last_entry_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata.pop("last_exit_trigger", None)
                    set_position_active(model_cfg.model, model_cfg.symbol, True)
                    dirty = True
                    market_price = current_price if current_price is not None else decision.price_used
                    self._log_trade_telemetry(
                        kind="entry",
                        model_cfg=model_cfg,
                        probability=probability,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        gate_pass=gate_pass,
                        decision=decision,
                        item=item,
                        entry_prob=probability,
                        current_price=market_price,
                        entry_price=decision.price_used,
                    )
                elif decision.executed:
                    self._mark_pending_entry(
                        state,
                        intent_id=intent_id,
                        probability=probability,
                        price=decision.price_used,
                        amount=decision.amount,
                        reason=decision.reason,
                        spread_bps=decision.spread_bps,
                    )
                    state.metadata["last_entry_reason"] = decision.reason or "submitted"
                    state.metadata["last_entry_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata.pop("open_entry_prob", None)
                    state.metadata.pop("open_peak_price", None)
                    dirty = True
                    if decision.exchange_order_id:
                        self._spawn_task(
                            self._monitor_order_fill(
                                model_cfg=model_cfg,
                                state_key=model_cfg.state_key,
                                intent_id=intent_id,
                                order_id=decision.exchange_order_id,
                                side="buy",
                                probability=probability,
                                min_hold_seconds=min_hold_seconds,
                                price_hint=decision.price_used,
                                amount_hint=decision.amount,
                            )
                        )
                else:
                    logger.info(
                        "Entry order skipped for %s %s (prob=%.4f threshold=%.4f reason=%s)",
                        model_cfg.model,
                        model_cfg.symbol,
                        probability,
                        entry_threshold,
                        decision.reason or "unknown",
                    )
                    record_skip_reason(model_cfg.model, model_cfg.symbol, decision.reason or "unknown")
                    state.metadata.pop("open_entry_prob", None)
                    state.metadata.pop("open_peak_price", None)
                    state.metadata["last_entry_reason"] = decision.reason or ""
                    state.metadata["last_entry_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    dirty = True
                    self._record_deadlock_trade_event(
                        model_cfg,
                        ts,
                        executed=False,
                        reason=decision.reason or "skipped",
                    )
            elif outcome.should_exit:
                if self._is_safe_mode_active() and not self.config.safe_mode_allow_exits:
                    record_skip_reason(model_cfg.model, model_cfg.symbol, "safe_mode")
                    state.metadata["last_exit_reason"] = "safe_mode"
                    dirty = True
                    exit_decision = OrderDecision(
                        executed=False,
                        price_used=current_price,
                        amount=entry_amount,
                        reason="safe_mode",
                        blocked_reason="safe_mode",
                    )
                    exit_payload = self._build_exit_audit_payload(
                        outcome=outcome,
                        entry_price=entry_price,
                        entry_amount=entry_amount,
                        decision=exit_decision,
                        current_price=current_price,
                        spread_bps=spread_bps,
                        fee_estimate_bps=fee_estimate_bps,
                        slippage_estimate_bps=slippage_estimate_bps,
                    )
                    await self.audit_logger.log_trade(
                        model=model_cfg.model,
                        symbol=model_cfg.symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        side="sell",
                        gate_pass=gate_pass,
                        probability=probability,
                        threshold=exit_threshold if outcome.exit_reason_primary in {"prob_floor", "prob_trailing", "trailing_prob_drop"} else entry_threshold,
                        decision=exit_decision,
                        policy_id=policy_id,
                        risk_payload=None,
                        decision_namespace=decision_namespace,
                        extra=exit_payload,
                    )
                    continue
                if outcome.exit_trigger == "time_limit":
                    logger.info(
                        "Time-based exit triggered for %s %s after %.1f minutes in position",
                        model_cfg.model,
                        model_cfg.symbol,
                        ((ts - (state.entry_ts or ts)).total_seconds() / 60.0),
                    )
                elif outcome.exit_trigger == "stop_loss":
                    logger.info(
                        "Stop-loss exit triggered for %s %s at price %.6f (entry %.6f, threshold %.3f%%)",
                        model_cfg.model,
                        model_cfg.symbol,
                        (current_price or 0.0),
                        (entry_price or 0.0),
                        (active_stop_loss_pct or 0.0) * 100.0,
                    )
                elif outcome.exit_trigger == "take_profit":
                    logger.info(
                        "Take-profit exit triggered for %s %s at price %.6f (entry %.6f, threshold %.3f%%)",
                        model_cfg.model,
                        model_cfg.symbol,
                        (current_price or 0.0),
                        (entry_price or 0.0),
                        (take_profit_pct or 0.0) * 100.0,
                    )
                record_would_trade(model_cfg.model, model_cfg.symbol, "sell")
                if outcome.skip_execution:
                    record_skip_reason(model_cfg.model, model_cfg.symbol, outcome.skip_reason or outcome.exit_trigger)
                    state.metadata["last_exit_reason"] = outcome.skip_reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{spread_bps:.4f}" if spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                    dirty = True
                    continue
                intent_id = _build_order_intent_id(model_cfg, ts, "sell")
                exit_qty: Optional[float] = None
                try:
                    exit_qty = float(state.metadata.get("open_amount") or model_cfg.order_amount or 0.0)
                except (TypeError, ValueError):
                    exit_qty = model_cfg.order_amount
                exit_notional = model_cfg.order_notional
                if exit_notional is None and exit_qty is not None and current_price:
                    exit_notional = float(exit_qty) * float(current_price)
                portfolio_state = self._build_portfolio_state(ts=ts, price_hints={model_cfg.symbol: current_price})
                portfolio_state["orders_last_hour"] = int(portfolio_state.get("orders_last_hour", 0)) + 1
                symbol_state = await self._build_symbol_state(
                    model_cfg,
                    state,
                    ts=ts,
                    price=current_price,
                    item=item,
                )
                risk_result = assess_and_adjust_order(
                    symbol=model_cfg.symbol,
                    action="EXIT_LONG",
                    desired_notional=exit_notional or 0.0,
                    desired_qty=exit_qty,
                    price=current_price,
                    spread_bps=spread_bps,
                    now_ts=int(ts.timestamp()),
                    portfolio_state=portfolio_state,
                    symbol_state=symbol_state,
                    risk_cfg=self.risk_limits,
                )
                if not risk_result["allowed"]:
                    reason = risk_result.get("block_reason") or "risk_blocked"
                    record_risk_blocked(model_cfg.symbol, reason)
                    record_skip_reason(model_cfg.model, model_cfg.symbol, reason)
                    state.metadata["last_exit_reason"] = reason
                    self._record_deadlock_trade_event(model_cfg, ts, executed=False, reason=reason)
                    dirty = True
                    blocked_decision = OrderDecision(
                        executed=False,
                        price_used=current_price,
                        amount=exit_qty,
                        reason=reason,
                        blocked_reason=reason,
                        order_intent_id=intent_id,
                        notional=risk_result.get("final_notional"),
                    )
                    exit_payload = self._build_exit_audit_payload(
                        outcome=outcome,
                        entry_price=entry_price,
                        entry_amount=entry_amount,
                        decision=blocked_decision,
                        current_price=current_price,
                        spread_bps=spread_bps,
                        fee_estimate_bps=fee_estimate_bps,
                        slippage_estimate_bps=slippage_estimate_bps,
                    )
                    await self.audit_logger.log_trade(
                        model=model_cfg.model,
                        symbol=model_cfg.symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        side="sell",
                        gate_pass=gate_pass,
                        probability=probability,
                        threshold=exit_threshold if outcome.exit_trigger in {"prob_floor", "prob_trailing"} else entry_threshold,
                        decision=blocked_decision,
                        policy_id=policy_id,
                        risk_payload=risk_result,
                        decision_namespace=decision_namespace,
                        extra=exit_payload,
                    )
                    continue
                final_exit_notional = risk_result.get("final_notional", exit_notional)
                final_exit_qty = risk_result.get("final_qty", exit_qty)
                for clip_reason in risk_result.get("clip_reasons", []):
                    record_risk_clipped(model_cfg.symbol, clip_reason)
                pnl_blocked = False
                expected_pnl = None
                expected_net = None
                prices_from_quote = decision_price_source.startswith("quote")
                if entry_price is not None and current_price is not None:
                    qty_for_pnl = entry_amount
                    if qty_for_pnl is None or qty_for_pnl <= 0:
                        qty_for_pnl = final_exit_qty
                    try:
                        if qty_for_pnl is not None and qty_for_pnl > 0:
                            expected_pnl = (float(current_price) - float(entry_price)) * float(qty_for_pnl)
                            expected_net = expected_pnl
                    except Exception:
                        expected_pnl = None
                        expected_net = None
                    notional_hint = None
                    try:
                        if qty_for_pnl is not None:
                            notional_hint = float(qty_for_pnl) * float(entry_price)
                    except Exception:
                        notional_hint = None
                    total_cost_bps = 0.0
                    try:
                        if fee_estimate_bps is not None and math.isfinite(float(fee_estimate_bps)):
                            total_cost_bps += float(fee_estimate_bps) * 2.0
                    except Exception:
                        pass
                    try:
                        if slippage_estimate_bps is not None and math.isfinite(float(slippage_estimate_bps)):
                            total_cost_bps += float(slippage_estimate_bps)
                    except Exception:
                        pass
                    if not prices_from_quote:
                        try:
                            if spread_bps is not None and math.isfinite(float(spread_bps)):
                                total_cost_bps += float(spread_bps)
                        except Exception:
                            pass
                    if expected_net is not None and notional_hint is not None and total_cost_bps:
                        expected_net = expected_net - (notional_hint * total_cost_bps / 1e4)

                self._record_turnover(ts, final_exit_notional)
                self._record_order_timestamp(ts)
                decision = await self.executor.submit(
                    exchange=model_cfg.exchange,
                    symbol=model_cfg.symbol,
                    side="sell",
                    order_amount=final_exit_qty,
                    order_notional=final_exit_notional,
                    max_spread_bps=self._exit_max_spread_bps(model_cfg),
                    shadow_mode=shadow_mode_flag,
                    order_intent_id=intent_id,
                )
                if decision.order_intent_id is None:
                    decision.order_intent_id = intent_id
                if decision.dedup_blocked:
                    record_dedup_blocked(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                    record_skip_reason(model_cfg.model, model_cfg.symbol, decision.reason or "duplicate_intent")
                    state.metadata["last_exit_reason"] = decision.reason or "duplicate_intent"
                    self._record_deadlock_trade_event(
                        model_cfg,
                        ts,
                        executed=False,
                        reason=decision.reason or "duplicate_intent",
                    )
                    dirty = True
                    continue
                if decision.shadow_mode:
                    record_shadow_blocked(model_cfg.model, model_cfg.symbol, "sell")
                record_trade_attempt(
                    model_cfg.model,
                    model_cfg.symbol,
                    "sell",
                    decision.executed,
                    decision.price_used,
                    decision.amount,
                )
                if decision.executed and entry_price is not None:
                    try:
                        qty_for_pnl = entry_amount
                        if qty_for_pnl is None or qty_for_pnl <= 0:
                            qty_for_pnl = decision.amount
                        if qty_for_pnl is not None and qty_for_pnl > 0 and decision.price_used is not None:
                            expected_pnl = (float(decision.price_used) - float(entry_price)) * float(qty_for_pnl)
                            expected_net = expected_pnl
                    except Exception:
                        expected_pnl = None
                    notional_hint = None
                    try:
                        if qty_for_pnl is not None and entry_price is not None:
                            notional_hint = float(qty_for_pnl) * float(entry_price)
                    except Exception:
                        notional_hint = None
                    total_cost_bps = 0.0
                    try:
                        if fee_estimate_bps is not None and math.isfinite(float(fee_estimate_bps)):
                            total_cost_bps += float(fee_estimate_bps) * 2.0
                    except Exception:
                        pass
                    try:
                        if slippage_estimate_bps is not None and math.isfinite(float(slippage_estimate_bps)):
                            total_cost_bps += float(slippage_estimate_bps)
                    except Exception:
                        pass
                    if not prices_from_quote:
                        try:
                            if spread_bps is not None and math.isfinite(float(spread_bps)):
                                total_cost_bps += float(spread_bps)
                        except Exception:
                            pass
                    if expected_net is not None and notional_hint is not None and total_cost_bps:
                        expected_net = expected_net - (notional_hint * total_cost_bps / 1e4)
                trade_executed = bool(decision.executed or decision.shadow_mode)
                self._record_deadlock_trade_event(
                    model_cfg,
                    ts,
                    executed=trade_executed,
                    reason=None if trade_executed else (decision.reason or decision.blocked_reason),
                )
                exit_payload = self._build_exit_audit_payload(
                    outcome=outcome,
                    entry_price=entry_price,
                    entry_amount=entry_amount,
                    decision=decision,
                    current_price=current_price,
                    spread_bps=spread_bps,
                    fee_estimate_bps=fee_estimate_bps,
                    slippage_estimate_bps=slippage_estimate_bps,
                )
                if expected_pnl is not None:
                    exit_payload["pnl_expected"] = expected_pnl
                    exit_payload["pnl_expected_net"] = expected_net
                    exit_payload["pnl_blocked"] = pnl_blocked
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    side="sell",
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=exit_threshold if outcome.exit_trigger in {"prob_floor", "prob_trailing"} else entry_threshold,
                    decision=decision,
                    policy_id=policy_id,
                    risk_payload=risk_result,
                    decision_namespace=decision_namespace,
                    extra=exit_payload,
                )
                fill_confirmed = self.config.dry_run or decision.intent_status in {IntentStatus.FILLED.value}
                if decision.executed and fill_confirmed:
                    pnl_value = 0.0
                    try:
                        entry_price = float(state.metadata.get("open_price") or 0.0)
                        entry_amount = float(state.metadata.get("open_amount") or 0.0)
                    except (TypeError, ValueError):
                        entry_price = 0.0
                        entry_amount = 0.0
                    exit_price = float(decision.price_used or 0.0)
                    exit_amount = float(decision.amount or entry_amount or 0.0)
                    if entry_price > 0.0 and entry_amount > 0.0 and exit_price > 0.0:
                        qty = min(entry_amount, exit_amount if exit_amount > 0.0 else entry_amount)
                        pnl_value = (exit_price - entry_price) * qty
                        record_realized_pnl(model_cfg.model, model_cfg.symbol, pnl_value)
                        self._update_risk_pnl(ts, pnl_value)
                        self._update_loss_streak_after_exit(state, ts, pnl_value, loss_guard_cfg)
                        state.metadata["last_realized_pnl"] = f"{pnl_value:.10f}"
                    state.metadata.pop("open_price", None)
                    state.metadata.pop("open_amount", None)
                    state.metadata.pop("open_side", None)
                    state.mark_exit(ts)
                    state.metadata["last_exit_reason"] = decision.reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                    state.metadata.pop("open_entry_prob", None)
                    state.metadata.pop("open_peak_price", None)
                    set_position_active(model_cfg.model, model_cfg.symbol, False)
                    dirty = True
                    market_price = current_price if current_price is not None else decision.price_used
                    self._log_trade_telemetry(
                        kind="exit",
                        model_cfg=model_cfg,
                        probability=probability,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        gate_pass=gate_pass,
                        decision=decision,
                        item=item,
                        entry_prob=state.metadata.get("open_entry_prob"),
                        current_price=market_price,
                        entry_price=entry_price,
                        exit_trigger=outcome.exit_trigger,
                    )
                elif decision.executed:
                    self._mark_pending_exit(
                        state,
                        intent_id=intent_id,
                        probability=probability,
                        price=decision.price_used,
                        amount=decision.amount,
                        reason=decision.reason,
                        spread_bps=decision.spread_bps,
                        trigger=outcome.exit_trigger,
                    )
                    state.metadata["last_exit_reason"] = decision.reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                    dirty = True
                    if decision.exchange_order_id:
                        self._spawn_task(
                            self._monitor_order_fill(
                                model_cfg=model_cfg,
                                state_key=model_cfg.state_key,
                                intent_id=intent_id,
                                order_id=decision.exchange_order_id,
                                side="sell",
                                probability=probability,
                                min_hold_seconds=min_hold_seconds,
                                price_hint=decision.price_used,
                                amount_hint=decision.amount,
                            )
                        )
                else:
                    logger.info(
                        "Exit order skipped for %s %s (prob=%.4f threshold=%.4f reason=%s)",
                        model_cfg.model,
                        model_cfg.symbol,
                        probability,
                        exit_threshold if outcome.exit_trigger in {"prob_floor", "prob_trailing"} else entry_threshold,
                        decision.reason or "unknown",
                    )
                    record_skip_reason(model_cfg.model, model_cfg.symbol, decision.reason or "unknown")
                    state.metadata["last_exit_reason"] = decision.reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = outcome.exit_trigger or ""
                    dirty = True
                    self._record_deadlock_trade_event(
                        model_cfg,
                        ts,
                        executed=False,
                        reason=decision.reason or "skipped",
                    )
            elif outcome.exit_armed:
                if outcome.skip_execution and outcome.skip_reason:
                    skip_reason = outcome.skip_reason
                elif outcome.exit_blocked_by_hold:
                    skip_reason = "min_hold"
                elif getattr(outcome, "exit_blocked_by_pnl", False):
                    skip_reason = "pnl_block"
                else:
                    skip_reason = outcome.exit_reason_primary or "exit_armed"
                record_skip_reason(model_cfg.model, model_cfg.symbol, skip_reason)
                state.metadata["last_exit_reason"] = skip_reason
                state.metadata["last_exit_spread_bps"] = (
                    f"{spread_bps:.4f}" if spread_bps is not None else ""
                )
                state.metadata["last_exit_trigger"] = outcome.exit_reason_primary or ""
                exit_decision = OrderDecision(
                    executed=False,
                    price_used=current_price,
                    amount=entry_amount,
                    reason=skip_reason,
                    blocked_reason=skip_reason,
                    order_intent_id=None,
                    notional=None,
                )
                exit_payload = self._build_exit_audit_payload(
                    outcome=outcome,
                    entry_price=entry_price,
                    entry_amount=entry_amount,
                    decision=exit_decision,
                    current_price=current_price,
                    spread_bps=spread_bps,
                    fee_estimate_bps=fee_estimate_bps,
                    slippage_estimate_bps=slippage_estimate_bps,
                )
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    side="sell",
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=exit_threshold if outcome.exit_reason_primary in {"prob_floor", "prob_trailing", "trailing_prob_drop"} else entry_threshold,
                    decision=exit_decision,
                    policy_id=policy_id,
                    risk_payload=None,
                    decision_namespace=decision_namespace,
                    extra=exit_payload,
                )
                dirty = True

        if dirty:
            self.state_store.update(model_cfg.state_key, state)
            await self.state_store.flush()
            if state.last_timestamp is not None:
                await self._write_last_processed_ts(model_cfg.state_key, state.last_timestamp)
        await self._evaluate_deadlock(parsed[-1][0] if parsed else datetime.now(timezone.utc))

    @staticmethod
    def _extract_order_values(
        order: Dict[str, object],
        *,
        price_hint: Optional[float],
        amount_hint: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        price = price_hint
        amount = amount_hint
        if isinstance(order, dict):
            try:
                avg_price = order.get("average")
                if avg_price:
                    price = float(avg_price)
            except Exception:
                pass
            try:
                raw_price = order.get("price")
                if raw_price:
                    price = float(raw_price)
            except Exception:
                pass
            try:
                filled = order.get("filled")
                if filled is not None:
                    amount = float(filled)
            except Exception:
                pass
            try:
                raw_amount = order.get("amount")
                if raw_amount is not None and (amount is None or amount <= 0):
                    amount = float(raw_amount)
            except Exception:
                pass
        return price, amount

    async def _handle_unfilled_intent(
        self,
        *,
        model_cfg: TradingModelConfig,
        state_key: str,
        side: str,
        status: str,
    ) -> None:
        state = self.state_store.get(state_key)
        if side == "buy":
            state.metadata["last_entry_reason"] = status
            self._clear_pending_flags(state, kind="entry")
        else:
            state.metadata["last_exit_reason"] = status
            self._clear_pending_flags(state, kind="exit")
        self.state_store.update(state_key, state)
        await self.state_store.flush()

    async def _apply_entry_fill(
        self,
        *,
        model_cfg: TradingModelConfig,
        state_key: str,
        order: Dict[str, object],
        probability: float,
        min_hold_seconds: int,
        price_hint: Optional[float],
        amount_hint: Optional[float],
    ) -> None:
        state = self.state_store.get(state_key)
        price, amount = self._extract_order_values(order, price_hint=price_hint, amount_hint=amount_hint)
        now = datetime.now(timezone.utc)
        if price is not None:
            state.metadata["open_price"] = f"{float(price):.10f}"
        if amount is not None:
            state.metadata["open_amount"] = f"{float(amount):.10f}"
        state.metadata["open_side"] = "long"
        prob_value = state.metadata.get("pending_entry_prob")
        if prob_value is None:
            prob_value = f"{probability:.10f}"
        state.metadata["open_entry_prob"] = str(prob_value)
        state.metadata["last_entry_reason"] = state.metadata.get("pending_entry_reason") or "filled"
        if state.metadata.get("pending_entry_spread_bps"):
            state.metadata["last_entry_spread_bps"] = state.metadata.get("pending_entry_spread_bps", "")
        self._clear_pending_flags(state, kind="entry")
        state.mark_entry(now, min_hold_seconds)
        set_position_active(model_cfg.model, model_cfg.symbol, True)
        self.state_store.update(state_key, state)
        await self.state_store.flush()

    async def _apply_exit_fill(
        self,
        *,
        model_cfg: TradingModelConfig,
        state_key: str,
        order: Dict[str, object],
        probability: float,
        exit_trigger: Optional[str],
        price_hint: Optional[float],
        amount_hint: Optional[float],
    ) -> None:
        state = self.state_store.get(state_key)
        exit_price, exit_amount = self._extract_order_values(order, price_hint=price_hint, amount_hint=amount_hint)
        pnl_value = 0.0
        try:
            entry_price = float(state.metadata.get("open_price") or 0.0)
            entry_amount = float(state.metadata.get("open_amount") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
            entry_amount = 0.0
        if entry_price > 0.0 and entry_amount > 0.0 and exit_price is not None:
            qty = min(entry_amount, exit_amount if exit_amount and exit_amount > 0.0 else entry_amount)
            pnl_value = (float(exit_price) - entry_price) * qty
            record_realized_pnl(model_cfg.model, model_cfg.symbol, pnl_value)
            self._update_risk_pnl(datetime.now(timezone.utc), pnl_value)
            state.metadata["last_realized_pnl"] = f"{pnl_value:.10f}"
        state.metadata.pop("open_price", None)
        state.metadata.pop("open_amount", None)
        state.metadata.pop("open_side", None)
        state.mark_exit(datetime.now(timezone.utc))
        state.metadata["last_exit_reason"] = state.metadata.get("pending_exit_reason") or "filled"
        state.metadata["last_exit_spread_bps"] = state.metadata.get("pending_exit_spread_bps", "")
        state.metadata["last_exit_trigger"] = state.metadata.get("pending_exit_trigger") or exit_trigger or ""
        state.metadata.pop("open_entry_prob", None)
        self._clear_pending_flags(state, kind="exit")
        set_position_active(model_cfg.model, model_cfg.symbol, False)
        self.state_store.update(state_key, state)
        await self.state_store.flush()

    async def _monitor_order_fill(
        self,
        *,
        model_cfg: TradingModelConfig,
        state_key: str,
        intent_id: str,
        order_id: str,
        side: str,
        probability: float,
        min_hold_seconds: int,
        price_hint: Optional[float],
        amount_hint: Optional[float],
    ) -> None:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.config.order_monitor_max_seconds)
        backoff = max(1, int(self.config.order_monitor_backoff_seconds))
        while self._running and datetime.now(timezone.utc) < deadline:
            await asyncio.sleep(backoff)
            try:
                order = await self.executor.fetch_order(model_cfg.exchange, model_cfg.symbol, order_id)
            except Exception as exc:
                logger.debug("Order monitor fetch failed for %s %s: %s", model_cfg.symbol, order_id, exc)
                backoff = min(backoff * 2, self.config.order_monitor_backoff_seconds * 4)
                continue
            status_raw = str(order.get("status") or "").lower() if isinstance(order, dict) else ""
            exchange_order_id = str(order.get("id") or order_id) if isinstance(order, dict) else order_id
            if status_raw in {"closed", "filled"}:
                await self.intent_ledger.set_status(intent_id, IntentStatus.FILLED, exchange_order_id=exchange_order_id)
                if side == "buy":
                    await self._apply_entry_fill(
                        model_cfg=model_cfg,
                        state_key=state_key,
                        order=order if isinstance(order, dict) else {},
                        probability=probability,
                        min_hold_seconds=min_hold_seconds,
                        price_hint=price_hint,
                        amount_hint=amount_hint,
                    )
                else:
                    await self._apply_exit_fill(
                        model_cfg=model_cfg,
                        state_key=state_key,
                        order=order if isinstance(order, dict) else {},
                        probability=probability,
                        exit_trigger=None,
                        price_hint=price_hint,
                        amount_hint=amount_hint,
                    )
                return
            if status_raw in {"canceled", "cancelled"}:
                await self.intent_ledger.set_status(
                    intent_id, IntentStatus.CANCELED, exchange_order_id=exchange_order_id
                )
                await self._handle_unfilled_intent(
                    model_cfg=model_cfg,
                    state_key=state_key,
                    side=side,
                    status="canceled",
                )
                return
            if status_raw in {"rejected"}:
                await self.intent_ledger.set_status(
                    intent_id, IntentStatus.REJECTED, exchange_order_id=exchange_order_id
                )
                await self._handle_unfilled_intent(
                    model_cfg=model_cfg,
                    state_key=state_key,
                    side=side,
                    status="rejected",
                )
                return
            await self.intent_ledger.set_status(intent_id, IntentStatus.ACKED, exchange_order_id=exchange_order_id)
            backoff = min(backoff * 2, self.config.order_monitor_backoff_seconds * 4)

        await self.intent_ledger.set_status(intent_id, IntentStatus.EXPIRED, exchange_order_id=order_id)
        await self._handle_unfilled_intent(
            model_cfg=model_cfg,
            state_key=state_key,
            side=side,
            status="expired",
        )

    async def _reconcile_once(self, *, startup: bool = False) -> bool:
        """
        Compare internal state to exchange truth; latch safe mode on divergence.
        """
        healthy = True
        report: Dict[str, Any] = {"startup": startup, "results": []}
        for model_cfg in self.config.trading_models:
            state = self.state_store.get(model_cfg.state_key)
            adapter = await self.executor.get_adapter(model_cfg.exchange)
            exposure_base = 0.0
            exposure_notional: Optional[float] = None
            price_sample: Optional[float] = None
            open_orders: list = []
            status = "ok"
            try:
                balance = await adapter.fetch_balance()
                base_symbol = (model_cfg.symbol or "").split("/")[0]
                free_map = balance.get("free") if isinstance(balance, dict) else {}
                used_map = balance.get("used") if isinstance(balance, dict) else {}
                exposure_base = float((free_map or {}).get(base_symbol, 0.0)) + float(
                    (used_map or {}).get(base_symbol, 0.0)
                )
                ticker = await adapter.fetch_ticker(model_cfg.symbol)
                bid = float(ticker.get("bid") or 0.0)
                ask = float(ticker.get("ask") or 0.0)
                if bid > 0 and ask > 0:
                    price_sample = (bid + ask) / 2.0
                    exposure_notional = price_sample * exposure_base
                open_orders = await adapter.fetch_open_orders(model_cfg.symbol)
            except Exception as exc:
                logger.warning("Reconciliation fetch failed for %s %s: %s", model_cfg.model, model_cfg.symbol, exc)
                healthy = False
                status = f"error:{exc.__class__.__name__}"
            exchange_in_position = False
            dust = False
            if exposure_notional is not None:
                dust = abs(exposure_notional) < float(self.config.reconcile_dust_notional)
                exchange_in_position = not dust and abs(exposure_notional) > 0.0
            else:
                dust = abs(exposure_base) < 1e-8
                exchange_in_position = not dust and abs(exposure_base) > 0.0
            internal_in_position = bool(state.in_position)
            internal_pending = bool(
                state.metadata.get("pending_entry_intent_id") or state.metadata.get("pending_exit_intent_id")
            )
            orders_open = len(open_orders or [])
            if status.startswith("error"):
                mismatch_reason = status
            elif internal_in_position != exchange_in_position and not dust:
                mismatch_reason = "exposure_mismatch"
                healthy = False
            elif orders_open > 0 and not internal_pending:
                mismatch_reason = "orphan_open_orders"
                healthy = False
            else:
                mismatch_reason = "ok"
            report["results"].append(
                {
                    "symbol": model_cfg.symbol,
                    "exchange": model_cfg.exchange,
                    "internal_in_position": internal_in_position,
                    "internal_pending": internal_pending,
                    "exchange_in_position": exchange_in_position,
                    "open_orders_exchange": orders_open,
                    "open_orders_internal": 1 if internal_pending else 0,
                    "exposure_base": exposure_base,
                    "exposure_notional": exposure_notional,
                    "price": price_sample,
                    "dust": dust,
                    "status": mismatch_reason,
                }
            )
        record_reconcile_run(healthy)
        await self.audit_logger.log_reconciliation(payload=report, timestamp=datetime.now(timezone.utc))
        if healthy:
            self._reconcile_healthy_streak += 1
            if self._safe_mode_reason and self._reconcile_healthy_streak >= self.config.reconcile_healthy_streak:
                await self._clear_safe_mode()
        else:
            self._reconcile_healthy_streak = 0
            await self._set_safe_mode("reconciliation_failed")
        return healthy

    async def _reconcile_loop(self) -> None:
        interval = max(1, int(self.config.reconcile_interval_seconds))
        while self._running:
            try:
                await self._reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Periodic reconciliation failed: %s", exc)
                self._reconcile_healthy_streak = 0
                try:
                    await self._set_safe_mode("reconciliation_error")
                except Exception:
                    pass
                record_reconcile_run(False)
            await asyncio.sleep(interval)

    def _log_trade_telemetry(
        self,
        *,
        kind: str,
        model_cfg: TradingModelConfig,
        probability: float,
        entry_threshold: float,
        exit_threshold: float,
        gate_pass: bool,
        decision: OrderDecision,
        item: Dict[str, object],
        entry_prob: Optional[float] = None,
        current_price: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_trigger: Optional[str] = None,
    ) -> None:
        key = (model_cfg.model, model_cfg.symbol, kind)
        count = self._telemetry_samples[key]
        self._telemetry_samples[key] = count + 1

        feature_spread = _extract_spread_from_item(item)

        def _fmt(value: Optional[float], precision: int = 4) -> str:
            if value is None:
                return "na"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "na"
            if not math.isfinite(numeric):
                return "na"
            return f"{numeric:.{precision}f}"

        logger.info(
            "Trade telemetry [%s #%d] %s %s gate=%s prob=%s entry_thr=%.4f exit_thr=%.4f "
            "entry_prob=%s exit_prob=%s market_price=%s execution_price=%s entry_price=%s "
            "feature_spread=%s exec_spread_bps=%s exit_trigger=%s reason=%s",
            kind,
            count + 1,
            model_cfg.model,
            model_cfg.symbol or "<none>",
            gate_pass,
            _fmt(probability),
            entry_threshold,
            exit_threshold,
            _fmt(entry_prob),
            _fmt(probability if kind == "exit" else None),
            _fmt(current_price, precision=6),
            _fmt(decision.price_used, precision=6),
            _fmt(entry_price, precision=6),
            _fmt(feature_spread, precision=6),
            _fmt(decision.spread_bps),
            exit_trigger or "",
            decision.reason or "",
        )

    def _build_exit_audit_payload(
        self,
        *,
        outcome: DecisionOutcome,
        entry_price: Optional[float],
        entry_amount: Optional[float],
        decision: Optional[OrderDecision] = None,
        current_price: Optional[float] = None,
        spread_bps: Optional[float] = None,
        fee_estimate_bps: Optional[float] = None,
        slippage_estimate_bps: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if outcome.exit_context:
            payload.update(outcome.exit_context)
        payload.update(
            {
                "exit_reason_primary": outcome.exit_reason_primary,
                "exit_reason_all": list(outcome.exit_reasons or []),
                "exit_armed": bool(outcome.exit_armed),
                "exit_blocked_by_hold": bool(outcome.exit_blocked_by_hold),
                "exit_blocked_by_pnl": bool(getattr(outcome, "exit_blocked_by_pnl", False)),
            }
        )
        entry_price_val = entry_price if entry_price is not None and entry_price > 0 else None
        entry_amount_val = entry_amount if entry_amount is not None and entry_amount > 0 else None
        decision_price = None
        if decision and decision.price_used is not None:
            try:
                decision_price = float(decision.price_used)
            except (TypeError, ValueError):
                decision_price = None
        if decision_price is None and current_price is not None:
            try:
                decision_price = float(current_price)
            except (TypeError, ValueError):
                decision_price = None
        decision_amount = None
        if decision and decision.amount is not None:
            try:
                decision_amount = float(decision.amount)
            except (TypeError, ValueError):
                decision_amount = None

        qty = None
        if entry_amount_val is not None and entry_amount_val > 0:
            qty = entry_amount_val
            if decision_amount is not None and decision_amount > 0:
                qty = min(entry_amount_val, decision_amount)

        if entry_price_val is not None and decision_price is not None and qty is not None:
            pnl_gross = (decision_price - entry_price_val) * qty
            payload["pnl_gross"] = pnl_gross
            total_cost_bps = 0.0
            trade_executed = bool(decision and decision.executed and not decision.shadow_mode)
            try:
                if fee_estimate_bps is not None and math.isfinite(float(fee_estimate_bps)):
                    total_cost_bps += float(fee_estimate_bps) * 2.0
            except Exception:
                pass
            try:
                if slippage_estimate_bps is not None and math.isfinite(float(slippage_estimate_bps)):
                    total_cost_bps += float(slippage_estimate_bps)
            except Exception:
                pass
            if not trade_executed:
                try:
                    if spread_bps is not None and math.isfinite(float(spread_bps)):
                        total_cost_bps += float(spread_bps)
                except Exception:
                    pass
            pnl_net = pnl_gross - (entry_price_val * qty * total_cost_bps / 1e4) if total_cost_bps else pnl_gross
            payload["pnl_net_estimate"] = pnl_net
            if trade_executed:
                payload["pnl_net_realized"] = pnl_net
        payload.setdefault("spread_bps_now", spread_bps if spread_bps is not None else None)
        if decision:
            payload["decision_price_used"] = decision.price_used
            payload["decision_amount"] = decision.amount
            payload["decision_reason"] = decision.reason
        return payload

    def _resolve_manifest(
        self,
        payload: Dict[str, object],
        model_label: str,
        *,
        symbol_hint: Optional[str] = None,
    ) -> Optional[ManifestSnapshot]:
        artifact = payload.get("artifact_dir")
        path = Path(str(artifact)) if artifact else self.config.models_root / model_label
        if not path.is_absolute():
            path = (self.config.models_root / path).resolve()
        path = path.expanduser().resolve()
        symbol = symbol_hint or payload.get("symbol")
        if isinstance(symbol, str):
            symbol = symbol.strip() or None
        else:
            symbol = None
        cache_key = (path, symbol)
        cached = self._manifest_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            artifacts = load_manifest_artifacts(path, model_label=model_label)
        except Exception as exc:
            logger.exception("Failed to load manifest for %s: %s", model_label, exc)
            return None
        infer_cfg = artifacts.gate_config.get("inference") or {}
        entry_threshold = _resolve_symbol_value(infer_cfg.get("prob_gate_min"), symbol)
        threshold_meta = artifacts.manifest.get("threshold")
        exit_threshold = None
        if isinstance(threshold_meta, dict):
            exit_threshold = threshold_meta.get("value")
        if entry_threshold is None and exit_threshold is not None:
            entry_threshold = exit_threshold
        if entry_threshold is None:
            entry_threshold = 0.5
        if exit_threshold is None:
            exit_threshold = entry_threshold
        metadata = artifacts.manifest.get("metadata") or {}
        exit_prob_drop = float(metadata.get("exit_prob_drop", 0.15))
        trigger_overrides = self.risk_limits.get("trigger_overrides") or {}
        symbol_overrides: Dict[str, object] = {}
        if symbol:
            try:
                symbol_overrides = (
                    (self.risk_limits.get("symbols") or {}).get(symbol, {}) or {}
                ).get("trigger_overrides") or {}
            except Exception:
                symbol_overrides = {}

        def _override_value(key: str, value: float, cast: type = float) -> float:
            raw = symbol_overrides.get(key)
            if raw is None:
                raw = trigger_overrides.get(key)
            if raw is None:
                return value
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return value

        entry_threshold = float(_override_value("entry_threshold", entry_threshold))
        exit_threshold = float(_override_value("exit_threshold", exit_threshold))
        exit_prob_drop = float(_override_value("exit_prob_drop", exit_prob_drop))
        if exit_prob_drop < 0:
            exit_prob_drop = 0.0
        if exit_threshold > entry_threshold:
            exit_threshold = entry_threshold
        override_key = symbol if symbol is not None else None
        if override_key is not None and override_key in self._prob_gate_overrides:
            entry_threshold = max(self._prob_gate_overrides[override_key], 0.0)
            exit_threshold = min(exit_threshold, entry_threshold) if exit_threshold is not None else entry_threshold
        min_hold_raw = _resolve_symbol_value(infer_cfg.get("min_hold_bars"), symbol)
        try:
            min_hold_bars = int(min_hold_raw or 1)
        except (TypeError, ValueError):
            min_hold_bars = 1
        long_only_raw = _resolve_symbol_value(infer_cfg.get("long_only"), symbol)
        long_only = bool(True if long_only_raw is None else long_only_raw)
        snapshot = ManifestSnapshot(
            entry_threshold=float(entry_threshold),
            exit_threshold=float(exit_threshold),
            exit_prob_drop=float(exit_prob_drop),
            min_hold_bars=min_hold_bars,
            long_only=long_only,
        )
        self._manifest_cache[cache_key] = snapshot
        return snapshot
