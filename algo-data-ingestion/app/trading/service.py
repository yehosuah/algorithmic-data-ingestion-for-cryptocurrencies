from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from redis import asyncio as aioredis

from app.monitoring.trading_metrics import (
    record_decision_queue_depth,
    record_gate_toggle,
    record_realized_pnl,
    record_trade_attempt,
    set_position_active,
)
from app.trading.audit import TradingAuditLogger
from app.trading.config import TradingConfig, TradingModelConfig
from app.trading.executor import OrderDecision, OrderExecutor
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
        return spread
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


@dataclass(frozen=True)
class ManifestSnapshot:
    entry_threshold: float
    exit_threshold: float
    exit_prob_drop: float
    min_hold_bars: int
    long_only: bool


class TradingService:
    TELEMETRY_SAMPLE_LIMIT = 5

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
        self.audit_logger = TradingAuditLogger(
            backend=audit_backend,
            redis_url=audit_redis_url,
            redis_stream=config.audit_redis_stream,
            redis_maxlen=config.audit_maxlen,
            postgres_dsn=audit_postgres_dsn,
            postgres_table=config.audit_postgres_table,
            file_path=audit_file_path,
        )
        self.executor = OrderExecutor(dry_run=config.dry_run)
        self._manifest_cache: Dict[Tuple[Path, Optional[str]], ManifestSnapshot] = {}
        self._model_map: Dict[Tuple[str, Optional[str]], TradingModelConfig] = {}
        for model_cfg in config.trading_models:
            symbol_key = (model_cfg.model, model_cfg.symbol)
            self._model_map[symbol_key] = model_cfg
            # Preserve backwards compatibility with single-entry configs by keeping
            # the first occurrence as the fallback when symbol is omitted.
            fallback_key = (model_cfg.model, None)
            self._model_map.setdefault(fallback_key, model_cfg)
        self._redis: Optional[aioredis.Redis] = None
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._telemetry_samples: Dict[Tuple[str, Optional[str], str], int] = defaultdict(int)

    def _create_queue_client(self) -> aioredis.Redis:
        return aioredis.from_url(
            self.config.decision_queue_url,
            encoding="utf-8",
            decode_responses=True,
        )

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
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
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

    async def _filter_stale_decisions(
        self,
        model_cfg: TradingModelConfig,
        state: PositionState,
        items: List[Tuple[datetime, Dict[str, object]]],
    ) -> List[Tuple[datetime, Dict[str, object]]]:
        if not items:
            return items
        cutoff = state.last_timestamp
        redis_cutoff = await self._read_last_processed_ts(model_cfg.state_key)
        if redis_cutoff is not None and (cutoff is None or redis_cutoff > cutoff):
            cutoff = redis_cutoff
        if cutoff is None:
            return items
        fresh: List[Tuple[datetime, Dict[str, object]]] = []
        for ts, payload in items:
            if ts > cutoff:
                fresh.append((ts, payload))
        dropped = len(items) - len(fresh)
        if dropped:
            logger.info(
                "Dropping %s stale decision(s) for %s %s (<= %s)",
                dropped,
                model_cfg.model,
                model_cfg.symbol,
                cutoff.isoformat(),
            )
        return fresh

    async def _poll_loop(self) -> None:
        timeout = max(1, int(self.config.redis_poll_timeout))
        while self._running:
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
            await self._handle_payload(payload)

    async def _handle_payload(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Discarding malformed payload: %s", raw[:80])
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
        parsed = await self._filter_stale_decisions(model_cfg, state, parsed)
        if not parsed:
            return

        min_hold_bars = int(model_cfg.min_hold_bars_override or manifest.min_hold_bars or 1)
        min_hold_seconds = max(1, min_hold_bars) * model_cfg.bar_seconds
        max_hold_minutes_cfg = model_cfg.max_hold_minutes
        max_hold_seconds: Optional[int] = None
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
        entry_threshold = float(manifest.entry_threshold)
        exit_threshold = float(manifest.exit_threshold)
        exit_prob_drop = float(manifest.exit_prob_drop)
        dirty = False
        try:
            stop_loss_pct = (
                float(model_cfg.stop_loss_pct) if getattr(model_cfg, "stop_loss_pct", None) is not None else None
            )
        except (TypeError, ValueError):
            stop_loss_pct = None
        try:
            take_profit_pct = (
                float(model_cfg.take_profit_pct)
                if getattr(model_cfg, "take_profit_pct", None) is not None
                else None
            )
        except (TypeError, ValueError):
            take_profit_pct = None
        if stop_loss_pct is not None and stop_loss_pct <= 0.0:
            stop_loss_pct = None
        if take_profit_pct is not None and take_profit_pct <= 0.0:
            take_profit_pct = None

        for ts, item in parsed:
            probability = float(item.get("probability") or 0.0)
            gate_pass = bool(item.get("gate_pass"))
            if not state.register_bar(ts):
                continue
            dirty = True
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
            current_price = _extract_price_from_item(item)
            exit_due_to_stop_loss = False
            exit_due_to_take_profit = False
            if (
                state.in_position
                and entry_price is not None
                and current_price is not None
            ):
                if stop_loss_pct is not None and current_price <= entry_price * (1.0 - stop_loss_pct):
                    exit_due_to_stop_loss = True
                elif take_profit_pct is not None and current_price >= entry_price * (1.0 + take_profit_pct):
                    exit_due_to_take_profit = True

            force_time_exit = False
            if (
                state.in_position
                and max_hold_seconds is not None
                and state.entry_ts is not None
                and (ts - state.entry_ts).total_seconds() >= max_hold_seconds
            ):
                force_time_exit = True

            ready_for_exit = state.ready_for_exit(ts)
            if force_time_exit and not ready_for_exit:
                ready_for_exit = True
            if (exit_due_to_stop_loss or exit_due_to_take_profit) and not ready_for_exit:
                ready_for_exit = True

            should_enter = (
                gate_pass
                and probability >= entry_threshold
                and manifest.long_only
                and state.ready_for_entry(ts)
            )

            exit_due_to_prob_floor = probability < exit_threshold
            exit_due_to_gate = (not gate_pass) or exit_due_to_prob_floor
            exit_due_to_trailing = False
            if state.in_position and stored_entry_prob is not None:
                exit_due_to_trailing = (stored_entry_prob - probability) >= exit_prob_drop

            exit_trigger: Optional[str] = None
            if force_time_exit:
                exit_trigger = "time_limit"
            elif exit_due_to_stop_loss:
                exit_trigger = "stop_loss"
            elif exit_due_to_take_profit:
                exit_trigger = "take_profit"
            elif exit_due_to_prob_floor:
                exit_trigger = "prob_floor"
            elif not gate_pass:
                exit_trigger = "gate_close"
            elif exit_due_to_trailing:
                exit_trigger = "prob_trailing"

            should_exit = (
                state.in_position
                and ready_for_exit
                and (
                    exit_due_to_gate
                    or exit_due_to_trailing
                    or force_time_exit
                    or exit_due_to_stop_loss
                    or exit_due_to_take_profit
                )
            )

            if should_enter:
                decision = await self.executor.submit(
                    exchange=model_cfg.exchange,
                    symbol=model_cfg.symbol,
                    side="buy",
                    order_amount=model_cfg.order_amount,
                    order_notional=model_cfg.order_notional,
                    max_spread_bps=model_cfg.max_spread_bps,
                )
                record_trade_attempt(
                    model_cfg.model,
                    model_cfg.symbol,
                    "buy",
                    decision.executed,
                    decision.price_used,
                    decision.amount,
                )
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timestamp=ts,
                    side="buy",
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=entry_threshold,
                    decision=decision,
                )
                if decision.executed:
                    if decision.price_used is not None:
                        state.metadata["open_price"] = f"{float(decision.price_used):.10f}"
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
                        current_price=current_price,
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
                    state.metadata.pop("open_entry_prob", None)
                    state.metadata["last_entry_reason"] = decision.reason or ""
                    state.metadata["last_entry_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    dirty = True
            elif should_exit:
                if exit_trigger == "time_limit":
                    logger.info(
                        "Time-based exit triggered for %s %s after %.1f minutes in position",
                        model_cfg.model,
                        model_cfg.symbol,
                        ((ts - (state.entry_ts or ts)).total_seconds() / 60.0),
                    )
                elif exit_trigger == "stop_loss":
                    logger.info(
                        "Stop-loss exit triggered for %s %s at price %.6f (entry %.6f, threshold %.3f%%)",
                        model_cfg.model,
                        model_cfg.symbol,
                        (current_price or 0.0),
                        (entry_price or 0.0),
                        (stop_loss_pct or 0.0) * 100.0,
                    )
                elif exit_trigger == "take_profit":
                    logger.info(
                        "Take-profit exit triggered for %s %s at price %.6f (entry %.6f, threshold %.3f%%)",
                        model_cfg.model,
                        model_cfg.symbol,
                        (current_price or 0.0),
                        (entry_price or 0.0),
                        (take_profit_pct or 0.0) * 100.0,
                    )
                decision = await self.executor.submit(
                    exchange=model_cfg.exchange,
                    symbol=model_cfg.symbol,
                    side="sell",
                    order_amount=model_cfg.order_amount,
                    order_notional=model_cfg.order_notional,
                    max_spread_bps=model_cfg.max_spread_bps,
                )
                record_trade_attempt(
                    model_cfg.model,
                    model_cfg.symbol,
                    "sell",
                    decision.executed,
                    decision.price_used,
                    decision.amount,
                )
                await self.audit_logger.log_trade(
                    model=model_cfg.model,
                    symbol=model_cfg.symbol,
                    timestamp=ts,
                    side="sell",
                    gate_pass=gate_pass,
                    probability=probability,
                    threshold=exit_threshold if exit_trigger in {"prob_floor", "prob_trailing"} else entry_threshold,
                    decision=decision,
                )
                if decision.executed:
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
                        state.metadata["last_realized_pnl"] = f"{pnl_value:.10f}"
                    state.metadata.pop("open_price", None)
                    state.metadata.pop("open_amount", None)
                    state.metadata.pop("open_side", None)
                    state.mark_exit(ts)
                    state.metadata["last_exit_reason"] = decision.reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = exit_trigger or ""
                    state.metadata.pop("open_entry_prob", None)
                    set_position_active(model_cfg.model, model_cfg.symbol, False)
                    dirty = True
                    self._log_trade_telemetry(
                        kind="exit",
                        model_cfg=model_cfg,
                        probability=probability,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        gate_pass=gate_pass,
                        decision=decision,
                        item=item,
                        entry_prob=stored_entry_prob,
                        current_price=current_price,
                        entry_price=entry_price,
                        exit_trigger=exit_trigger,
                    )
                else:
                    logger.info(
                        "Exit order skipped for %s %s (prob=%.4f threshold=%.4f reason=%s)",
                        model_cfg.model,
                        model_cfg.symbol,
                        probability,
                        exit_threshold if exit_trigger in {"prob_floor", "prob_trailing"} else entry_threshold,
                        decision.reason or "unknown",
                    )
                    state.metadata["last_exit_reason"] = decision.reason or ""
                    state.metadata["last_exit_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata["last_exit_trigger"] = exit_trigger or ""
                    dirty = True

        if dirty:
            self.state_store.update(model_cfg.state_key, state)
            await self.state_store.flush()
            if state.last_timestamp is not None:
                await self._write_last_processed_ts(model_cfg.state_key, state.last_timestamp)

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
        if count >= self.TELEMETRY_SAMPLE_LIMIT:
            return
        self._telemetry_samples[key] = count + 1

        feature_spread = _extract_spread_from_item(item)

        def _fmt(value: Optional[float], precision: int = 4) -> str:
            return f"{value:.{precision}f}" if value is not None else "na"

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
