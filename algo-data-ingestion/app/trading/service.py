from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from redis import asyncio as aioredis

from app.monitoring.trading_metrics import (
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


@dataclass(frozen=True)
class ManifestSnapshot:
    threshold: float
    min_hold_bars: int
    long_only: bool


class TradingService:
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
        self._manifest_cache: Dict[Path, ManifestSnapshot] = {}
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
        self._redis = aioredis.from_url(
            self.config.decision_queue_url,
            encoding="utf-8",
            decode_responses=True,
        )
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

    async def _poll_loop(self) -> None:
        assert self._redis is not None
        timeout = max(1, int(self.config.redis_poll_timeout))
        while self._running:
            try:
                result = await self._redis.blpop(
                    self.config.decision_queue_key,
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Redis BLPOP failed: %s", exc)
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

        manifest = self._resolve_manifest(message, model_label)
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
        threshold = float(manifest.threshold)

        state = self.state_store.get(model_cfg.state_key)
        set_position_active(model_cfg.model, model_cfg.symbol, state.in_position)
        dirty = False

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
                    threshold=threshold,
                    previous_gate=previous_gate,
                )
                record_gate_toggle(model_cfg.model, model_cfg.symbol, gate_pass)
            state.last_gate = gate_pass
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

            should_enter = (
                gate_pass
                and probability >= threshold
                and manifest.long_only
                and state.ready_for_entry(ts)
            )

            exit_due_to_gate = (not gate_pass) or (probability < threshold)
            exit_trigger: Optional[str] = None
            if exit_due_to_gate:
                exit_trigger = "gate_close"
            if force_time_exit:
                exit_trigger = "time_limit"

            should_exit = (
                state.in_position
                and ready_for_exit
                and (exit_due_to_gate or force_time_exit)
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
                    threshold=threshold,
                    decision=decision,
                )
                if decision.executed:
                    if decision.price_used is not None:
                        state.metadata["open_price"] = f"{float(decision.price_used):.10f}"
                    if decision.amount is not None:
                        state.metadata["open_amount"] = f"{float(decision.amount):.10f}"
                    state.metadata["open_side"] = "long"
                    state.mark_entry(ts, min_hold_seconds)
                    state.metadata["last_entry_reason"] = decision.reason or ""
                    state.metadata["last_entry_spread_bps"] = (
                        f"{decision.spread_bps:.4f}" if decision.spread_bps is not None else ""
                    )
                    state.metadata.pop("last_exit_trigger", None)
                    set_position_active(model_cfg.model, model_cfg.symbol, True)
                    dirty = True
                else:
                    logger.info(
                        "Entry order skipped for %s %s (prob=%.4f threshold=%.4f reason=%s)",
                        model_cfg.model,
                        model_cfg.symbol,
                        probability,
                        threshold,
                        decision.reason or "unknown",
                    )
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
                    threshold=threshold,
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
                    set_position_active(model_cfg.model, model_cfg.symbol, False)
                    dirty = True
                else:
                    logger.info(
                        "Exit order skipped for %s %s (prob=%.4f threshold=%.4f reason=%s)",
                        model_cfg.model,
                        model_cfg.symbol,
                        probability,
                        threshold,
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

    def _resolve_manifest(self, payload: Dict[str, object], model_label: str) -> Optional[ManifestSnapshot]:
        artifact = payload.get("artifact_dir")
        path = Path(str(artifact)) if artifact else self.config.models_root / model_label
        if not path.is_absolute():
            path = (self.config.models_root / path).resolve()
        path = path.expanduser().resolve()
        cached = self._manifest_cache.get(path)
        if cached is not None:
            return cached
        try:
            artifacts = load_manifest_artifacts(path, model_label=model_label)
        except Exception as exc:
            logger.exception("Failed to load manifest for %s: %s", model_label, exc)
            return None
        infer_cfg = artifacts.gate_config.get("inference") or {}
        threshold = infer_cfg.get("prob_gate_min")
        if threshold is None:
            threshold_meta = artifacts.manifest.get("threshold")
            if isinstance(threshold_meta, dict):
                threshold = threshold_meta.get("value")
        if threshold is None:
            threshold = 0.5
        min_hold_bars = int(infer_cfg.get("min_hold_bars") or 1)
        long_only = bool(infer_cfg.get("long_only", True))
        snapshot = ManifestSnapshot(
            threshold=float(threshold),
            min_hold_bars=min_hold_bars,
            long_only=long_only,
        )
        self._manifest_cache[path] = snapshot
        return snapshot
