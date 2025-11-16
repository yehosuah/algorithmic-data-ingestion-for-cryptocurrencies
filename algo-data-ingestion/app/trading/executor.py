from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.adapters.ccxt_adapter import CCXTAdapter

try:
    from ccxt.base.errors import ExchangeNotAvailable, NetworkError
except Exception:  # pragma: no cover - ccxt may be missing in some test environments
    ExchangeNotAvailable = NetworkError = Exception  # type: ignore

logger = logging.getLogger("app.trading.executor")


@dataclass
class OrderDecision:
    executed: bool
    price_used: Optional[float] = None
    amount: Optional[float] = None
    spread_bps: Optional[float] = None
    reason: Optional[str] = None
    order_payload: Optional[Dict[str, Any]] = None


class OrderExecutor:
    """
    Thin wrapper that encapsulates CCXT order submission with spread guardrails.
    """

    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._adapters: Dict[str, CCXTAdapter] = {}

    async def _get_adapter(self, exchange: str) -> CCXTAdapter:
        adapter = self._adapters.get(exchange)
        if adapter is None:
            adapter = CCXTAdapter(exchange)
            self._adapters[exchange] = adapter
        return adapter

    async def submit(
        self,
        *,
        exchange: str,
        symbol: str,
        side: str,
        order_amount: Optional[float],
        order_notional: Optional[float],
        max_spread_bps: float,
    ) -> OrderDecision:
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
            return OrderDecision(
                executed=False,
                reason="ticker_unavailable",
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
            return OrderDecision(
                executed=False,
                reason="invalid_bid_ask",
            )
        spread = ask - bid
        mid = (ask + bid) / 2.0
        spread_bps = (spread / mid) * 1e4 if mid > 0 else None
        if spread_bps is not None and spread_bps > max_spread_bps:
            return OrderDecision(
                executed=False,
                spread_bps=spread_bps,
                price_used=ask if side.lower() == "buy" else bid,
                reason="spread_threshold",
            )

        price = ask if side.lower() == "buy" else bid
        amount = order_amount
        if amount is None and order_notional is not None and price > 0:
            amount = order_notional / price
        if amount is None or amount <= 0:
            return OrderDecision(
                executed=False,
                price_used=price,
                spread_bps=spread_bps,
                reason="invalid_amount",
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
            return OrderDecision(
                executed=True,
                price_used=price,
                amount=amount,
                spread_bps=spread_bps,
                reason="dry_run",
                order_payload={"status": "dry_run"},
            )

        try:
            await adapter.ensure_markets()
            amount_precise = adapter.amount_to_precision(symbol, amount)
            if amount_precise <= 0:
                return OrderDecision(
                    executed=False,
                    price_used=price,
                    spread_bps=spread_bps,
                    reason="zero_amount_after_precision",
                )

            order = await adapter.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount_precise,
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
            )
        except Exception as exc:  # pragma: no cover - trade submission failure path
            logger.exception("Live trade submission failed: %s", exc)
            return OrderDecision(
                executed=False,
                price_used=price,
                spread_bps=spread_bps,
                reason=f"order_error:{exc.__class__.__name__}",
            )

    async def close(self) -> None:
        for adapter in self._adapters.values():
            try:
                await adapter.close()
            except Exception:
                continue
        self._adapters.clear()
