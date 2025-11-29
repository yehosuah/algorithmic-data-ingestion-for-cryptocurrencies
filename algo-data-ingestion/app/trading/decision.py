from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from app.trading.state import PositionState


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
    spread_bps: Optional[float] = None,
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

    # Execution guard (spread) – informational for offline; live executor enforces too.
    if _is_finite(spread_bps) and cfg.max_spread_bps is not None:
        try:
            if float(spread_bps) > float(cfg.max_spread_bps):
                skip_execution = True
                skip_reason = "spread_threshold"
        except Exception:
            pass

    min_hold_seconds = max(1, int(cfg.min_hold_bars)) * max(1, int(cfg.bar_seconds))
    max_hold_seconds = cfg.max_hold_seconds if cfg.max_hold_seconds and cfg.max_hold_seconds > 0 else None

    # Entry conditions
    if (
        gate_pass
        and probability >= cfg.entry_threshold
        and cfg.long_only
        and state.ready_for_entry(ts)
    ):
        should_enter = True

    # Exit conditions
    exit_due_to_prob_floor = probability < cfg.exit_threshold
    exit_due_to_gate = not gate_pass or exit_due_to_prob_floor
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

    if state.in_position and ready_for_exit:
        if force_time_exit:
            exit_trigger = "time_limit"
        elif exit_due_to_stop:
            exit_trigger = "stop_loss"
        elif exit_due_to_take_profit:
            exit_trigger = "take_profit"
        elif exit_due_to_prob_floor:
            exit_trigger = "prob_floor"
        elif not gate_pass:
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

    return DecisionOutcome(
        should_enter=should_enter,
        should_exit=should_exit,
        exit_trigger=exit_trigger,
        skip_execution=skip_execution,
        skip_reason=skip_reason,
    )
