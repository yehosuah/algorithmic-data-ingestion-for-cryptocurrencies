from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Gauge

TRADING_GATE_TOGGLES = Counter(
    "trading_gate_toggles_total",
    "Count of gate toggles recorded by the trading service.",
    labelnames=("model", "symbol", "state"),
)

TRADING_TRADE_ATTEMPTS = Counter(
    "trading_trade_attempts_total",
    "Number of trade submissions issued by the trading service.",
    labelnames=("model", "symbol", "side", "executed"),
)

TRADING_TRADE_NOTIONAL = Counter(
    "trading_trade_notional_total",
    "Total notional value routed through the trading service, bucketed by execution outcome.",
    labelnames=("model", "symbol", "side", "executed"),
)

TRADING_REALIZED_PNL = Gauge(
    "trading_realized_pnl_total",
    "Cumulative realized P&L (quote currency) for completed positions.",
    labelnames=("model", "symbol"),
)

TRADING_POSITION_ACTIVE = Gauge(
    "trading_position_active",
    "Whether the trading service currently holds an open position (1=yes, 0=no).",
    labelnames=("model", "symbol"),
)


def record_gate_toggle(model: str, symbol: str, gate_pass: bool) -> None:
    state = "open" if gate_pass else "closed"
    TRADING_GATE_TOGGLES.labels(model=model, symbol=symbol, state=state).inc()


def record_trade_attempt(
    model: str,
    symbol: str,
    side: str,
    executed: bool,
    price: Optional[float],
    amount: Optional[float],
) -> None:
    executed_label = "yes" if executed else "no"
    TRADING_TRADE_ATTEMPTS.labels(model=model, symbol=symbol, side=side, executed=executed_label).inc()
    if price is None or amount is None:
        return
    notional = abs(float(price) * float(amount))
    TRADING_TRADE_NOTIONAL.labels(model=model, symbol=symbol, side=side, executed=executed_label).inc(notional)


def record_realized_pnl(model: str, symbol: str, pnl: float) -> None:
    if pnl == 0.0:
        return
    metric = TRADING_REALIZED_PNL.labels(model=model, symbol=symbol)
    if pnl > 0:
        metric.inc(pnl)
    else:
        metric.dec(abs(pnl))


def set_position_active(model: str, symbol: str, active: bool) -> None:
    TRADING_POSITION_ACTIVE.labels(model=model, symbol=symbol).set(1.0 if active else 0.0)
