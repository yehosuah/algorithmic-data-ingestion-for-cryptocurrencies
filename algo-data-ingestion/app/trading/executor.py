from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from redis import asyncio as aioredis

from app.adapters.ccxt_adapter import CCXTAdapter
from app.monitoring.trading_metrics import record_intent_status

try:
    from ccxt.base.errors import ExchangeNotAvailable, NetworkError
except Exception:  # pragma: no cover - ccxt may be missing in some test environments
    ExchangeNotAvailable = NetworkError = Exception  # type: ignore

logger = logging.getLogger("app.trading.executor")


def _extract_min_cost(market: Dict[str, Any]) -> Optional[float]:
    try:
        cost_limit = (market or {}).get("limits", {}).get("cost", {}) or {}
        value = cost_limit.get("min")
        if value is None:
            value = (market or {}).get("minCost")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_min_amount(market: Dict[str, Any]) -> Optional[float]:
    try:
        amount_limit = (market or {}).get("limits", {}).get("amount", {}) or {}
        value = amount_limit.get("min")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class IntentStatus(str, Enum):
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKED = "acked"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class IntentLedger:
    """
    Lightweight ledger with atomic intent locks (Redis preferred, memory fallback).
    """

    def __init__(
        self,
        *,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        prefix: str = "trading:intent",
        lock_ttl_seconds: int = 6 * 3600,
    ) -> None:
        self.backend = backend
        self.redis_url = redis_url
        self.prefix = prefix.rstrip(":")
        self.lock_ttl_seconds = max(60, int(lock_ttl_seconds))
        self._redis: Optional[aioredis.Redis] = None
        self._local: Dict[str, Dict[str, Any]] = {}
        self._local_lock = asyncio.Lock()

    @property
    def _lock_prefix(self) -> str:
        return f"{self.prefix}:lock"

    @property
    def _ledger_prefix(self) -> str:
        return f"{self.prefix}:ledger"

    async def _ensure_redis(self) -> aioredis.Redis:
        if self._redis is not None:
            return self._redis
        if not self.redis_url or self.backend != "redis":
            raise RuntimeError("Redis backend not configured for IntentLedger")
        self._redis = aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        try:
            await self._redis.ping()
        except Exception:
            with contextlib.suppress(Exception):
                await self._redis.close()
            self._redis = None
            raise
        return self._redis

    async def acquire(self, intent_id: str) -> bool:
        """
        Acquire the intent lock. Returns False when already held (duplicate).
        """
        if self.backend == "redis":
            try:
                client = await self._ensure_redis()
                result = await client.set(
                    f"{self._lock_prefix}:{intent_id}",
                    "1",
                    ex=self.lock_ttl_seconds,
                    nx=True,
                )
                if result:
                    await self.set_status(intent_id, IntentStatus.PENDING_SUBMIT)
                return bool(result)
            except Exception as exc:
                logger.warning("Intent ledger redis fallback to memory due to error: %s", exc)
                self.backend = "memory"

        async with self._local_lock:
            entry = self._local.get(intent_id)
            now = datetime.now(timezone.utc).timestamp()
            if entry and entry.get("_lock_expires", 0) > now:
                return False
            self._local[intent_id] = {
                "status": IntentStatus.PENDING_SUBMIT.value,
                "_lock_expires": now + self.lock_ttl_seconds,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            record_intent_status(IntentStatus.PENDING_SUBMIT.value)
            return True

    async def get_status(self, intent_id: str) -> Optional[str]:
        if self.backend == "redis":
            try:
                client = await self._ensure_redis()
                status = await client.hget(f"{self._ledger_prefix}:{intent_id}", "status")
                return status
            except Exception:
                self.backend = "memory"
        async with self._local_lock:
            entry = self._local.get(intent_id)
            if not entry:
                return None
            return str(entry.get("status"))

    async def set_status(
        self,
        intent_id: str,
        status: IntentStatus,
        *,
        exchange_order_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if exchange_order_id:
            payload["exchange_order_id"] = exchange_order_id
        if error:
            payload["error"] = error
        if self.backend == "redis":
            try:
                client = await self._ensure_redis()
                await client.hset(f"{self._ledger_prefix}:{intent_id}", mapping=payload)
                await client.expire(f"{self._ledger_prefix}:{intent_id}", self.lock_ttl_seconds)
                record_intent_status(status.value)
                return
            except Exception as exc:
                logger.warning("Intent ledger redis set_status fallback to memory: %s", exc)
                self.backend = "memory"
        async with self._local_lock:
            entry = self._local.get(intent_id, {})
            entry.update(payload)
            entry["_lock_expires"] = datetime.now(timezone.utc).timestamp() + self.lock_ttl_seconds
            self._local[intent_id] = entry
            record_intent_status(status.value)

    async def close(self) -> None:
        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.close()
            self._redis = None


@dataclass
class OrderDecision:
    executed: bool
    price_used: Optional[float] = None
    amount: Optional[float] = None
    spread_bps: Optional[float] = None
    reason: Optional[str] = None
    order_payload: Optional[Dict[str, Any]] = None
    order_intent_id: Optional[str] = None
    shadow_mode: bool = False
    blocked_reason: Optional[str] = None
    notional: Optional[float] = None
    dedup_blocked: bool = False
    intent_status: Optional[str] = None
    exchange_order_id: Optional[str] = None


class OrderExecutor:
    """
    Thin wrapper that encapsulates CCXT order submission with spread guardrails.
    """

    def __init__(self, *, dry_run: bool = True, intent_ledger: Optional[IntentLedger] = None) -> None:
        self.dry_run = dry_run
        self._adapters: Dict[str, CCXTAdapter] = {}
        self._intent_ledger = intent_ledger

    async def _get_adapter(self, exchange: str) -> CCXTAdapter:
        adapter = self._adapters.get(exchange)
        if adapter is None:
            adapter = CCXTAdapter(exchange)
            self._adapters[exchange] = adapter
        return adapter

    async def get_adapter(self, exchange: str) -> CCXTAdapter:
        """
        Expose adapter for reconciliation or monitoring.
        """
        return await self._get_adapter(exchange)

    async def get_market_info(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch cached market metadata for a symbol (requires ensure_markets to have been called).
        """
        adapter = await self._get_adapter(exchange)
        await adapter.ensure_markets()
        try:
            markets = getattr(adapter.client, "markets", {}) or {}
            if isinstance(markets, dict):
                return dict(markets.get(symbol, {}) or {})
        except Exception:
            return {}
        return {}

    async def fetch_order(self, exchange: str, symbol: str, order_id: str) -> Dict[str, Any]:
        adapter = await self._get_adapter(exchange)
        return await adapter.fetch_order(order_id=order_id, symbol=symbol)

    async def submit(
        self,
        *,
        exchange: str,
        symbol: str,
        side: str,
        order_amount: Optional[float],
        order_notional: Optional[float],
        max_spread_bps: float,
        shadow_mode: bool = False,
        order_intent_id: Optional[str] = None,
    ) -> OrderDecision:
        if self._intent_ledger and order_intent_id:
            acquired = await self._intent_ledger.acquire(order_intent_id)
            if not acquired:
                status = await self._intent_ledger.get_status(order_intent_id)
                return OrderDecision(
                    executed=False,
                    reason="duplicate_intent",
                    blocked_reason="duplicate_intent",
                    order_intent_id=order_intent_id,
                    dedup_blocked=True,
                    intent_status=status,
                )

        async def _update_status(status: IntentStatus, *, exchange_order_id: Optional[str] = None, error: Optional[str] = None) -> None:
            if self._intent_ledger and order_intent_id:
                await self._intent_ledger.set_status(
                    order_intent_id,
                    status,
                    exchange_order_id=exchange_order_id,
                    error=error,
                )

        adapter = await self._get_adapter(exchange)
        try:
            ticker = await adapter.fetch_ticker(symbol)
        except (ExchangeNotAvailable, NetworkError) as exc:
            logger.warning(
                "Ticker fetch unavailable for %s %s: %s",
                exchange,
                symbol,
                exc,
            )
            await _update_status(IntentStatus.ERROR, error=str(exc))
            return OrderDecision(
                executed=False,
                reason="ticker_unavailable",
                order_intent_id=order_intent_id,
                intent_status=IntentStatus.ERROR.value if not self.dry_run else IntentStatus.REJECTED.value,
            )
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid <= 0 or ask <= 0:
            # Fallback to the freshest order book snapshot when the ticker
            # payload drops bid/ask (common on sandbox endpoints).
            try:
                order_book = await adapter.fetch_order_book(symbol, limit=5)
            except Exception:
                order_book = None
            if order_book is not None:
                best_bid = 0.0
                best_ask = 0.0
                for row in order_book.itertuples(index=False):
                    side = getattr(row, "side", "").lower()
                    price = float(getattr(row, "price", 0.0) or 0.0)
                    if price <= 0:
                        continue
                    if side == "bid":
                        best_bid = max(best_bid, price)
                    elif side == "ask":
                        if best_ask == 0.0 or price < best_ask:
                            best_ask = price
                if best_bid > 0 and best_ask > 0 and best_bid < best_ask:
                    logger.debug(
                        "Recovered bid/ask for %s %s via order book fallback (%.6f/%.6f)",
                        exchange,
                        symbol,
                        best_bid,
                        best_ask,
                    )
                    bid, ask = best_bid, best_ask
        if bid <= 0 or ask <= 0:
            await _update_status(IntentStatus.REJECTED)
            return OrderDecision(
                executed=False,
                reason="invalid_bid_ask",
                order_intent_id=order_intent_id,
                intent_status=IntentStatus.REJECTED.value,
            )
        spread = ask - bid
        mid = (ask + bid) / 2.0
        spread_bps = (spread / mid) * 1e4 if mid > 0 else None
        if spread_bps is not None and spread_bps > max_spread_bps:
            await _update_status(IntentStatus.REJECTED)
            return OrderDecision(
                executed=False,
                spread_bps=spread_bps,
                price_used=ask if side.lower() == "buy" else bid,
                reason="spread_threshold",
                order_intent_id=order_intent_id,
                intent_status=IntentStatus.REJECTED.value,
            )

        price = ask if side.lower() == "buy" else bid
        amount = order_amount
        if amount is None and order_notional is not None and price > 0:
            amount = order_notional / price
        if amount is None or amount <= 0:
            await _update_status(IntentStatus.REJECTED)
            return OrderDecision(
                executed=False,
                price_used=price,
                spread_bps=spread_bps,
                reason="invalid_amount",
                order_intent_id=order_intent_id,
                intent_status=IntentStatus.REJECTED.value,
            )
        notional = (amount * price) if price and amount else None

        if shadow_mode:
            logger.info(
                "Shadow mode blocking %s order for %s %s @ %s (amount %.6f, spread %.4f bps)",
                side.upper(),
                exchange,
                symbol,
                price,
                amount,
                spread_bps or 0.0,
            )
            await _update_status(IntentStatus.CANCELED)
            return OrderDecision(
                executed=False,
                price_used=price,
                amount=amount,
                spread_bps=spread_bps,
                reason="shadow_mode",
                blocked_reason="shadow_mode",
                shadow_mode=True,
                order_payload={"status": "shadow_blocked"},
                order_intent_id=order_intent_id,
                notional=notional,
                intent_status=IntentStatus.CANCELED.value,
            )

        if self.dry_run:
            logger.info(
                "DRY RUN %s order for %s %s @ %s (amount %.6f, spread %.4f bps)",
                side.upper(),
                exchange,
                symbol,
                price,
                amount,
                spread_bps or 0.0,
            )
            await _update_status(IntentStatus.FILLED)
            return OrderDecision(
                executed=True,
                price_used=price,
                amount=amount,
                spread_bps=spread_bps,
                reason="dry_run",
                order_payload={"status": "dry_run"},
                order_intent_id=order_intent_id,
                notional=notional,
                intent_status=IntentStatus.FILLED.value,
            )

        try:
            await adapter.ensure_markets()
            market_info = await self.get_market_info(exchange, symbol)
            amount_precise = adapter.amount_to_precision(symbol, amount)
            if amount_precise <= 0:
                return OrderDecision(
                    executed=False,
                    price_used=price,
                    spread_bps=spread_bps,
                    reason="zero_amount_after_precision",
                )
            notional_precise = price * amount_precise if price else notional
            min_amount = _extract_min_amount(market_info)
            min_cost = _extract_min_cost(market_info)
            if min_amount is not None and amount_precise < min_amount:
                await _update_status(IntentStatus.REJECTED)
                return OrderDecision(
                    executed=False,
                    price_used=price,
                    spread_bps=spread_bps,
                    amount=amount_precise,
                    reason="min_amount",
                    order_intent_id=order_intent_id,
                    notional=notional_precise,
                    intent_status=IntentStatus.REJECTED.value,
                )
            if min_cost is not None and notional_precise is not None and notional_precise < min_cost:
                await _update_status(IntentStatus.REJECTED)
                return OrderDecision(
                    executed=False,
                    price_used=price,
                    spread_bps=spread_bps,
                    amount=amount_precise,
                    reason="min_notional",
                    order_intent_id=order_intent_id,
                    notional=notional_precise,
                    intent_status=IntentStatus.REJECTED.value,
                )

            order = await adapter.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount_precise,
            )
            order_id = str(order.get("id") or order.get("orderId") or "") if isinstance(order, dict) else ""
            await _update_status(
                IntentStatus.SUBMITTED,
                exchange_order_id=order_id or None,
            )
            logger.info(
                "Executed %s order on %s %s amount %.6f (spread %.4f bps)",
                side.upper(),
                exchange,
                symbol,
                amount_precise,
                spread_bps or 0.0,
            )
            return OrderDecision(
                executed=True,
                price_used=price,
                amount=amount_precise,
                spread_bps=spread_bps,
                order_payload=order,
                order_intent_id=order_intent_id,
                notional=notional_precise if notional_precise is not None else price * amount_precise if price else None,
                intent_status=IntentStatus.SUBMITTED.value,
                exchange_order_id=order_id or None,
            )
        except Exception as exc:  # pragma: no cover - trade submission failure path
            logger.exception("Live trade submission failed: %s", exc)
            await _update_status(IntentStatus.ERROR, error=str(exc))
            return OrderDecision(
                executed=False,
                price_used=price,
                spread_bps=spread_bps,
                reason=f"order_error:{exc.__class__.__name__}",
                order_intent_id=order_intent_id,
                notional=notional if notional is not None else price * amount if price and amount else None,
                intent_status=IntentStatus.ERROR.value,
            )

    async def close(self) -> None:
        for adapter in self._adapters.values():
            try:
                await adapter.close()
            except Exception:
                continue
        self._adapters.clear()
        if self._intent_ledger is not None:
            await self._intent_ledger.close()
