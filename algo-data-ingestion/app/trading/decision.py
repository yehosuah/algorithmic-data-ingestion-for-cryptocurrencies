from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

from app.trading.state import PositionState

KILL_SWITCH_ENV = "TRADING_KILL_SWITCH"
SAFE_MODE_ENV = "TRADING_SAFE_MODE"
_FLAG_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TriggerConfig:
    entry_threshold: float
    exit_threshold: float
    exit_prob_drop: float
    min_hold_bars: int
    bar_seconds: int
    long_only: bool = True
    max_hold_seconds: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_spread_bps: Optional[float] = None


@dataclass(frozen=True)
class DecisionOutcome:
    should_enter: bool
    should_exit: bool
    exit_trigger: Optional[str] = None
    skip_execution: bool = False
    skip_reason: Optional[str] = None
    entry_block_reason: Optional[str] = None
    exit_reason_primary: Optional[str] = None
    exit_reasons: Tuple[str, ...] = field(default_factory=tuple)
    exit_armed: bool = False
    exit_blocked_by_hold: bool = False
    exit_blocked_by_pnl: bool = False
    exit_context: Dict[str, Any] = field(default_factory=dict)


def _is_finite(val: Optional[float]) -> bool:
    try:
        return val is not None and math.isfinite(float(val))
    except Exception:
        return False


def decide_bar(
    *,
    ts: datetime,
    probability: float,
    gate_pass: bool,
    state: PositionState,
    cfg: TriggerConfig,
    current_price: Optional[float] = None,
    entry_price: Optional[float] = None,
    entry_amount: Optional[float] = None,
    spread_bps: Optional[float] = None,
    include_spread_cost: bool = True,
    safe_mode_active: bool = False,
    fee_estimate_bps: Optional[float] = None,
    slippage_estimate_bps: Optional[float] = None,
) -> DecisionOutcome:
    """
    Pure decision function mirrored from the live trading loop.

    Inputs:
        ts: bar timestamp
        probability: model probability for this bar
        gate_pass: manifest gate decision for this bar
        state: current PositionState (not mutated here)
        cfg: trigger configuration (thresholds/holds/stops)
        current_price: last price/close/mid if available
        entry_price: stored position entry price (if any)
        spread_bps: instantaneous spread in basis points (send-time guard)
    Output:
        DecisionOutcome describing entry/exit intent and any guard skip reason.
    """
    should_enter = False
    should_exit = False
    exit_trigger: Optional[str] = None
    skip_execution = False
    skip_reason: Optional[str] = None
    entry_block_reason: Optional[str] = None

    # Execution guard (spread) – informational for offline; live executor enforces too.
    if _is_finite(spread_bps) and cfg.max_spread_bps is not None:
        try:
            if float(spread_bps) > float(cfg.max_spread_bps):
                skip_execution = True
                skip_reason = "spread_threshold"
        except Exception:
            pass

    if not state.in_position:
        safe_mode_flag = safe_mode_active or os.getenv(SAFE_MODE_ENV, "").strip().lower() in _FLAG_TRUE_VALUES
        if os.getenv(KILL_SWITCH_ENV, "").strip().lower() in _FLAG_TRUE_VALUES:
            entry_block_reason = "kill_switch"
        elif safe_mode_flag:
            entry_block_reason = "safe_mode"

    min_hold_seconds = max(1, int(cfg.min_hold_bars)) * max(1, int(cfg.bar_seconds))
    max_hold_seconds = cfg.max_hold_seconds if cfg.max_hold_seconds and cfg.max_hold_seconds > 0 else None

    # Entry conditions
    if (
        gate_pass
        and probability >= cfg.entry_threshold
        and cfg.long_only
        and state.ready_for_entry(ts)
        and entry_block_reason is None
    ):
        should_enter = True

    # Exit conditions
    exit_due_to_prob_floor = probability < cfg.exit_threshold
    # Gate closure should only force an exit when the probability gate still passes but another
    # non-probability predicate fails (e.g. volatility/spread filters). This preserves hysteresis
    # between `entry_threshold` and `exit_threshold` instead of exiting on minor probability jitter.
    exit_due_to_gate_close = (not gate_pass) and probability >= cfg.entry_threshold
    exit_due_to_gate = exit_due_to_prob_floor or exit_due_to_gate_close
    exit_due_to_trailing = False
    if state.in_position and _is_finite(state.metadata.get("open_entry_prob")):
        try:
            stored_entry_prob = float(state.metadata.get("open_entry_prob") or 0.0)
            exit_due_to_trailing = (stored_entry_prob - probability) >= cfg.exit_prob_drop
        except Exception:
            exit_due_to_trailing = False
    exit_due_to_stop = False
    exit_due_to_take_profit = False
    if (
        state.in_position
        and _is_finite(entry_price)
        and _is_finite(current_price)
    ):
        ep = float(entry_price)  # type: ignore[arg-type]
        cp = float(current_price)  # type: ignore[arg-type]
        if cfg.stop_loss_pct is not None and cfg.stop_loss_pct > 0 and cp <= ep * (1.0 - cfg.stop_loss_pct):
            exit_due_to_stop = True
        if cfg.take_profit_pct is not None and cfg.take_profit_pct > 0 and cp >= ep * (1.0 + cfg.take_profit_pct):
            exit_due_to_take_profit = True

    force_time_exit = False
    if state.in_position and max_hold_seconds is not None and state.entry_ts is not None:
        force_time_exit = (ts - state.entry_ts).total_seconds() >= max_hold_seconds

    ready_for_exit = state.ready_for_exit(ts)
    if (exit_due_to_stop or exit_due_to_take_profit or force_time_exit) and not ready_for_exit:
        ready_for_exit = True

    exit_reasons: List[str] = []
    if state.in_position:
        if exit_due_to_stop:
            exit_reasons.append("stop_loss")
        if exit_due_to_take_profit:
            exit_reasons.append("take_profit")
        if force_time_exit:
            exit_reasons.append("time_exit")
        if exit_due_to_prob_floor:
            exit_reasons.append("prob_floor")
        if exit_due_to_gate_close and "gate_close" not in exit_reasons:
            exit_reasons.append("gate_close")
        if exit_due_to_trailing:
            exit_reasons.append("trailing_prob_drop")

    exit_armed = bool(exit_reasons)
    exit_blocked_by_hold = False
    exit_blocked_by_pnl = False

    if state.in_position and ready_for_exit:
        if force_time_exit:
            exit_trigger = "time_limit"
        elif exit_due_to_stop:
            exit_trigger = "stop_loss"
        elif exit_due_to_take_profit:
            exit_trigger = "take_profit"
        elif exit_due_to_prob_floor:
            exit_trigger = "prob_floor"
        elif exit_due_to_gate_close:
            exit_trigger = "gate_close"
        elif exit_due_to_trailing:
            exit_trigger = "prob_trailing"

        should_exit = bool(
            exit_due_to_gate
            or exit_due_to_trailing
            or force_time_exit
            or exit_due_to_stop
            or exit_due_to_take_profit
        )

    # Honor min-hold: if state says hold_until is in future, we cannot exit yet.
    if state.in_position and state.hold_until is not None and ts < state.hold_until:
        if should_exit:
            # Defer exit until hold expires.
            should_exit = False
            exit_trigger = None
        if exit_armed:
            exit_blocked_by_hold = True

    prob_entry: Optional[float] = None
    try:
        prob_entry = float(state.metadata.get("open_entry_prob")) if state.metadata.get("open_entry_prob") else None
    except Exception:
        prob_entry = None

    entry_amt_val: Optional[float] = None
    try:
        entry_amt_val = float(entry_amount) if entry_amount is not None else None
    except Exception:
        entry_amt_val = None
    entry_price_val: Optional[float] = None
    try:
        entry_price_val = float(entry_price) if entry_price is not None else None
    except Exception:
        entry_price_val = None
    current_price_val: Optional[float] = None
    try:
        current_price_val = float(current_price) if current_price is not None else None
    except Exception:
        current_price_val = None

    pnl_gross = None
    pnl_net_estimate = None
    pnl_notional = None
    if (
        entry_price_val is not None
        and entry_price_val > 0
        and current_price_val is not None
        and entry_amt_val is not None
        and entry_amt_val > 0
    ):
        pnl_notional = entry_price_val * entry_amt_val
        pnl_gross = (current_price_val - entry_price_val) * entry_amt_val
        total_cost_bps = 0.0
        if _is_finite(fee_estimate_bps):
            total_cost_bps += float(fee_estimate_bps) * 2.0
        if _is_finite(slippage_estimate_bps):
            total_cost_bps += float(slippage_estimate_bps)
        if include_spread_cost and _is_finite(spread_bps):
            total_cost_bps += float(spread_bps)
        if total_cost_bps and pnl_notional is not None:
            pnl_net_estimate = pnl_gross - (pnl_notional * total_cost_bps / 1e4)

    pnl_for_exit = pnl_net_estimate if pnl_net_estimate is not None else pnl_gross
    if pnl_for_exit is not None and pnl_for_exit < 0:
        exit_blocked_by_pnl = True

    if (
        exit_blocked_by_pnl
        and should_exit
        and exit_trigger in {"prob_floor", "gate_close", "prob_trailing", "take_profit"}
    ):
        should_exit = False
        exit_trigger = None

    exit_reason_primary = exit_trigger or (exit_reasons[0] if exit_reasons else None)

    exit_context = {
        "prob_now": float(probability),
        "prob_entry": prob_entry,
        "entry_threshold": float(cfg.entry_threshold),
        "exit_threshold": float(cfg.exit_threshold),
        "exit_prob_drop": float(cfg.exit_prob_drop),
        "min_hold_bars": int(cfg.min_hold_bars),
        "bar_seconds": int(cfg.bar_seconds),
        "price_now": current_price_val,
        "entry_price": entry_price_val,
        "entry_amount": entry_amt_val,
        "entry_ts": state.entry_ts.isoformat() if state.entry_ts else None,
        "spread_bps_now": float(spread_bps) if _is_finite(spread_bps) else None,
        "fee_estimate_bps": float(fee_estimate_bps) if _is_finite(fee_estimate_bps) else None,
        "slippage_estimate_bps": float(slippage_estimate_bps) if _is_finite(slippage_estimate_bps) else None,
        "pnl_gross": pnl_gross,
        "pnl_net_estimate": pnl_net_estimate,
        "pnl_notional": pnl_notional,
        "exit_blocked_by_hold": bool(exit_blocked_by_hold),
        "exit_blocked_by_pnl": bool(exit_blocked_by_pnl),
    }

    return DecisionOutcome(
        should_enter=should_enter,
        should_exit=should_exit,
        exit_trigger=exit_trigger,
        skip_execution=skip_execution,
        skip_reason=skip_reason,
        entry_block_reason=entry_block_reason,
        exit_reason_primary=exit_reason_primary,
        exit_reasons=tuple(exit_reasons),
        exit_armed=exit_armed,
        exit_blocked_by_hold=exit_blocked_by_hold,
        exit_blocked_by_pnl=exit_blocked_by_pnl,
        exit_context=exit_context,
    )
