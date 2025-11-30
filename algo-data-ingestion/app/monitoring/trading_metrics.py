from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

from prometheus_client import Counter, Gauge

LIVE_OBSERVABILITY_COUNTERS = ("trade_count", "coverage", "skips_by_reason", "deadlock_action_taken_total")

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

TRADING_DECISION_QUEUE_DEPTH = Gauge(
    "trading_decision_queue_depth",
    "Depth of the Redis decision queue consumed by the trading service.",
    labelnames=("queue",),
)

TRADING_GATE_COVERAGE_RATIO = Gauge(
    "trading_gate_coverage_ratio",
    "Share of decisions that passed gate checks per symbol.",
    labelnames=("model", "symbol"),
)

TRADING_DECISIONS_TOTAL = Counter(
    "trading_decisions_total",
    "Number of decisions consumed by the trading service, labeled by gate outcome.",
    labelnames=("model", "symbol", "gate_pass"),
)

TRADING_WOULD_TRADE = Counter(
    "trading_would_trade_total",
    "Number of times the service decided to enter/exit (pre-execution).",
    labelnames=("model", "symbol", "side"),
)

TRADING_SHADOW_BLOCKED = Counter(
    "trading_shadow_blocked_total",
    "Orders blocked at the executor boundary due to shadow mode.",
    labelnames=("model", "symbol", "side"),
)

TRADING_SKIPS_BY_REASON = Counter(
    "trading_skips_by_reason_total",
    "Orders skipped or blocked, bucketed by reason.",
    labelnames=("model", "symbol", "reason"),
)

TRADING_DEDUP_BLOCKED = Counter(
    "trading_dedup_blocked_orders_total",
    "Orders blocked at submit-time due to duplicate intent ids.",
    labelnames=("model", "symbol", "reason"),
)

TRADING_RECONCILE_RUNS = Counter(
    "trading_reconcile_runs_total",
    "Number of reconciliation cycles executed.",
    labelnames=("status",),
)

TRADING_SAFE_MODE_LATCHED = Gauge(
    "trading_safe_mode_latched",
    "Current safe-mode latch state (1=latched).",
    labelnames=("reason",),
)

TRADING_INTENT_LEDGER_STATE = Counter(
    "trading_intent_ledger_state_total",
    "Counts of intent ledger state transitions.",
    labelnames=("state",),
)

TRADING_RISK_BLOCKED = Counter(
    "trading_risk_blocked_total",
    "Orders blocked by runtime risk checks, grouped by symbol and reason.",
    labelnames=("symbol", "reason"),
)

TRADING_RISK_CLIPPED = Counter(
    "trading_risk_clipped_total",
    "Orders clipped (size reduced) by runtime risk checks.",
    labelnames=("symbol", "clip_reason"),
)

TRADING_PORTFOLIO_TURNOVER = Gauge(
    "trading_portfolio_turnover_estimate",
    "Estimated turnover fraction of capital over the trailing window.",
    labelnames=("window",),
)

TRADING_PORTFOLIO_DRAWDOWN = Gauge(
    "trading_portfolio_drawdown_pct",
    "Portfolio drawdown as a fraction of peak equity.",
    labelnames=(),
)

TRADING_PORTFOLIO_DAILY_PNL = Gauge(
    "trading_portfolio_daily_pnl_pct",
    "Realized daily P&L as a fraction of reference capital.",
    labelnames=(),
)

TRADING_ORDERS_PER_HOUR = Gauge(
    "trading_orders_per_hour",
    "Order submissions observed in the trailing 1h window.",
    labelnames=("symbol",),
)

TRADING_CONCURRENT_POSITIONS = Gauge(
    "trading_concurrent_positions",
    "Number of concurrent open positions across symbols.",
    labelnames=(),
)

TRADING_DEADLOCK_COVERAGE_RATIO = Gauge(
    "trading_deadlock_coverage_ratio",
    "Rolling coverage ratio within the deadlock detector window.",
    labelnames=("model", "symbol", "window"),
)

TRADING_DEADLOCK_PROB_GATE_RATIO = Gauge(
    "trading_deadlock_prob_gate_ratio",
    "Rolling fraction of probabilities above the prob_gate_min threshold.",
    labelnames=("model", "symbol", "window"),
)

TRADING_DEADLOCK_TRADE_COUNT = Gauge(
    "trading_deadlock_trade_count_window",
    "Trade attempts observed in the rolling deadlock window.",
    labelnames=("model", "symbol", "window"),
)

TRADING_DEADLOCK_PORTFOLIO_TRADES = Gauge(
    "trading_deadlock_portfolio_trades_window",
    "Portfolio-level trade attempts observed in the rolling window.",
    labelnames=("window",),
)

TRADING_DEADLOCK_PORTFOLIO_COVERAGE = Gauge(
    "trading_deadlock_portfolio_coverage_ratio",
    "Average coverage ratio across symbols in the rolling window.",
    labelnames=("window",),
)

TRADING_DEADLOCK_BLOCKED_REASON = Counter(
    "trading_deadlock_blocked_by_reason_total",
    "Counts of blocked order attempts tracked by the deadlock detector.",
    labelnames=("model", "symbol", "reason"),
)

TRADING_DEADLOCK_ACTIONS = Counter(
    "trading_deadlock_action_taken_total",
    "Deadlock mitigation actions executed by the trading service.",
    labelnames=("action",),
)

_COVERAGE_COUNTS: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"total": 0, "pass": 0})


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


def record_decision_queue_depth(queue: str, depth: int) -> None:
    TRADING_DECISION_QUEUE_DEPTH.labels(queue=queue).set(max(0, int(depth)))


def record_decision_coverage(model: str, symbol: str, gate_pass: bool) -> None:
    key = (model, symbol)
    stats = _COVERAGE_COUNTS[key]
    stats["total"] += 1
    if gate_pass:
        stats["pass"] += 1
    ratio = float(stats["pass"]) / float(stats["total"]) if stats["total"] else 0.0
    gate_label = "yes" if gate_pass else "no"
    TRADING_DECISIONS_TOTAL.labels(model=model, symbol=symbol, gate_pass=gate_label).inc()
    TRADING_GATE_COVERAGE_RATIO.labels(model=model, symbol=symbol).set(ratio)


def record_would_trade(model: str, symbol: str, side: str) -> None:
    TRADING_WOULD_TRADE.labels(model=model, symbol=symbol, side=side).inc()


def record_shadow_blocked(model: str, symbol: str, side: str) -> None:
    TRADING_SHADOW_BLOCKED.labels(model=model, symbol=symbol, side=side).inc()


def record_skip_reason(model: str, symbol: str, reason: Optional[str]) -> None:
    if reason is None:
        return
    reason_label = str(reason).strip()
    if not reason_label:
        return
    TRADING_SKIPS_BY_REASON.labels(model=model, symbol=symbol, reason=reason_label).inc()


def record_dedup_blocked(model: str, symbol: str, reason: str) -> None:
    TRADING_DEDUP_BLOCKED.labels(model=model, symbol=symbol, reason=reason or "duplicate").inc()


def record_reconcile_run(success: bool) -> None:
    TRADING_RECONCILE_RUNS.labels(status="success" if success else "failure").inc()
    if success:
        TRADING_SAFE_MODE_LATCHED.labels(reason="").set(0.0)


def record_safe_mode(reason: str, active: bool) -> None:
    TRADING_SAFE_MODE_LATCHED.labels(reason=reason or "unknown").set(1.0 if active else 0.0)


def record_intent_status(state: str) -> None:
    if not state:
        return
    TRADING_INTENT_LEDGER_STATE.labels(state=state).inc()


def record_risk_blocked(symbol: str, reason: str) -> None:
    if not reason:
        return
    TRADING_RISK_BLOCKED.labels(symbol=symbol or "*", reason=reason).inc()


def record_risk_clipped(symbol: str, clip_reason: str) -> None:
    if not clip_reason:
        return
    TRADING_RISK_CLIPPED.labels(symbol=symbol or "*", clip_reason=clip_reason).inc()


def record_portfolio_turnover_estimate(turnover_fraction: float) -> None:
    TRADING_PORTFOLIO_TURNOVER.labels(window="1d").set(max(0.0, float(turnover_fraction or 0.0)))


def record_portfolio_drawdown(drawdown_pct: float) -> None:
    TRADING_PORTFOLIO_DRAWDOWN.set(max(0.0, float(drawdown_pct or 0.0)))


def record_portfolio_daily_pnl(pnl_pct: float) -> None:
    TRADING_PORTFOLIO_DAILY_PNL.set(float(pnl_pct or 0.0))


def record_orders_per_hour(symbol: str, count: int) -> None:
    TRADING_ORDERS_PER_HOUR.labels(symbol=symbol or "*").set(max(0, int(count or 0)))


def record_concurrent_positions(count: int) -> None:
    TRADING_CONCURRENT_POSITIONS.set(max(0, int(count or 0)))


def record_deadlock_window_metrics(
    *,
    model: str,
    symbol: str,
    window_label: str,
    coverage_ratio: float,
    prob_gate_pass_ratio: float,
    trade_count: int,
) -> None:
    TRADING_DEADLOCK_COVERAGE_RATIO.labels(model=model, symbol=symbol, window=window_label).set(
        max(0.0, float(coverage_ratio))
    )
    TRADING_DEADLOCK_PROB_GATE_RATIO.labels(model=model, symbol=symbol, window=window_label).set(
        max(0.0, float(prob_gate_pass_ratio))
    )
    TRADING_DEADLOCK_TRADE_COUNT.labels(model=model, symbol=symbol, window=window_label).set(max(0, int(trade_count)))


def record_deadlock_portfolio_metrics(
    *,
    window_label: str,
    trade_count: int,
    coverage_ratio: float,
) -> None:
    TRADING_DEADLOCK_PORTFOLIO_TRADES.labels(window=window_label).set(max(0, int(trade_count)))
    TRADING_DEADLOCK_PORTFOLIO_COVERAGE.labels(window=window_label).set(max(0.0, float(coverage_ratio)))


def record_deadlock_block_reason(model: str, symbol: str, reason: str) -> None:
    if not reason:
        return
    TRADING_DEADLOCK_BLOCKED_REASON.labels(model=model, symbol=symbol, reason=reason).inc()


def record_deadlock_action(action: str) -> None:
    TRADING_DEADLOCK_ACTIONS.labels(action=action or "unknown").inc()
