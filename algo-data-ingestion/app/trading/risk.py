from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _get_symbol_cfg(risk_cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbols = (risk_cfg or {}).get("symbols") or {}
    return symbols.get(symbol, {}) if isinstance(symbols, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_down(value: float, step: Optional[float]) -> float:
    if step is None:
        return value
    try:
        step_f = float(step)
    except (TypeError, ValueError):
        return value
    if step_f <= 0:
        return value
    return math.floor(float(value) / step_f) * step_f


def _choose_cap(candidates: List[float]) -> Optional[float]:
    caps = [c for c in candidates if c is not None and c > 0 and math.isfinite(c)]
    if not caps:
        return None
    return min(caps)


def _blocked_response(
    *,
    reason: str,
    notional: Optional[float],
    qty: Optional[float],
    clip_reasons: List[str],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "allowed": False,
        "final_notional": notional,
        "final_qty": qty,
        "block_reason": reason,
        "clip_reasons": clip_reasons,
        "risk_snapshot": snapshot,
    }


def assess_and_adjust_order(
    *,
    symbol: str,
    action: str,  # "ENTER_LONG" / "EXIT_LONG"
    desired_notional: float,
    desired_qty: float | None,
    price: float | None,
    spread_bps: float | None,
    now_ts: int,
    portfolio_state: dict,
    symbol_state: dict,
    risk_cfg: dict,
) -> dict:
    """
    Apply runtime risk limits to a prospective order. Returns the adjusted size and decision.
    """
    is_entry = str(action or "").upper().startswith("ENTER")
    symbol_cfg = _get_symbol_cfg(risk_cfg, symbol)
    clip_reasons: List[str] = []
    snapshot: Dict[str, Any] = {}

    capital = _safe_float(risk_cfg.get("capital"), 0.0)
    max_gross_leverage = _safe_float(risk_cfg.get("max_gross_leverage"), 0.0)
    max_net_exposure = _safe_float(risk_cfg.get("max_net_exposure"), 0.0)
    max_turnover_per_day = _safe_float(risk_cfg.get("max_turnover_per_day"), 0.0)
    max_orders_per_hour = _safe_float(risk_cfg.get("max_orders_per_hour"), 0.0)
    max_concurrent_positions = int(_safe_float(risk_cfg.get("max_concurrent_positions"), 0))
    daily_loss_limit_pct = _safe_float(risk_cfg.get("daily_loss_limit_pct"), 0.0)
    max_drawdown_pct = _safe_float(risk_cfg.get("max_drawdown_pct"), 0.0)
    halt_on_safe_mode = bool(risk_cfg.get("halt_on_safe_mode", True))
    allow_exits_during_halt = bool(risk_cfg.get("allow_exits_during_halt", True))
    cooldown_after_exit_min = _safe_float(risk_cfg.get("cooldown_minutes_after_exit"), 0.0)
    cooldown_after_loss_min = _safe_float(risk_cfg.get("cooldown_minutes_after_loss"), 0.0)
    halt_if_data_stale_seconds = _safe_float(risk_cfg.get("halt_if_data_stale_seconds"), 0.0)
    halt_if_spread_bps_gt = _safe_float(
        symbol_cfg.get("max_spread_bps", risk_cfg.get("halt_if_spread_bps_gt")), 0.0
    )
    halt_if_vol_zscore_gt = _safe_float(
        symbol_cfg.get("max_vol_zscore", risk_cfg.get("halt_if_vol_zscore_gt")), 0.0
    )
    min_trade_notional = _safe_float(symbol_cfg.get("min_trade_notional", risk_cfg.get("min_trade_notional")), 0.0)
    qty_step = symbol_cfg.get("qty_step") or symbol_state.get("qty_step")
    price_tick = symbol_cfg.get("price_tick") or symbol_state.get("price_tick")

    requested_notional = _safe_float(desired_notional, 0.0)
    if requested_notional <= 0 and price and desired_qty:
        requested_notional = _safe_float(desired_qty) * _safe_float(price)

    current_symbol_notional = _safe_float(symbol_state.get("open_notional"))
    current_symbol_qty = _safe_float(symbol_state.get("open_qty"))
    gross_exposure = _safe_float(portfolio_state.get("gross_exposure"))
    net_exposure = _safe_float(portfolio_state.get("net_exposure"))
    turnover_1d = _safe_float(portfolio_state.get("turnover_1d"))
    orders_last_hour = int(_safe_float(portfolio_state.get("orders_last_hour")))
    open_positions_count = int(
        portfolio_state.get("open_positions")
        or len(portfolio_state.get("open_symbols") or [])
        or 0
    )

    snapshot.update(
        {
            "requested_notional": requested_notional,
            "requested_qty": desired_qty,
            "price": price,
            "spread_bps": spread_bps,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "turnover_1d": turnover_1d,
            "orders_last_hour": orders_last_hour,
            "open_positions": open_positions_count,
            "current_symbol_notional": current_symbol_notional,
            "current_symbol_qty": current_symbol_qty,
        }
    )

    # Hard blocks (entries)
    if is_entry:
        if portfolio_state.get("kill_switch"):
            return _blocked_response(
                reason="kill_switch",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if halt_on_safe_mode and (portfolio_state.get("safe_mode") or portfolio_state.get("reconciliation_latched")):
            return _blocked_response(
                reason="safe_mode",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        daily_pnl_pct = portfolio_state.get("daily_pnl_pct")
        if daily_loss_limit_pct > 0 and daily_pnl_pct is not None and float(daily_pnl_pct) <= -daily_loss_limit_pct:
            snapshot["daily_pnl_pct"] = daily_pnl_pct
            return _blocked_response(
                reason="daily_loss_limit",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        drawdown_pct = portfolio_state.get("drawdown_pct")
        if max_drawdown_pct > 0 and drawdown_pct is not None and float(drawdown_pct) >= max_drawdown_pct:
            snapshot["drawdown_pct"] = drawdown_pct
            return _blocked_response(
                reason="max_drawdown",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if max_turnover_per_day > 0 and capital > 0:
            max_turnover_notional = max_turnover_per_day * capital
            projected_turnover = turnover_1d + abs(requested_notional)
            snapshot["projected_turnover"] = projected_turnover
            snapshot["max_turnover_notional"] = max_turnover_notional
            if projected_turnover > max_turnover_notional:
                return _blocked_response(
                    reason="turnover_limit",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
        if max_orders_per_hour > 0 and orders_last_hour >= max_orders_per_hour:
            return _blocked_response(
                reason="max_orders_per_hour",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if (
            max_concurrent_positions > 0
            and not symbol_state.get("in_position")
            and open_positions_count >= max_concurrent_positions
        ):
            return _blocked_response(
                reason="max_concurrent_positions",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if halt_if_spread_bps_gt > 0 and spread_bps is not None and float(spread_bps) > halt_if_spread_bps_gt:
            return _blocked_response(
                reason="spread_too_wide",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if risk_cfg.get("halt_if_missing_price_bars") and symbol_state.get("missing_price_bars"):
            return _blocked_response(
                reason="missing_price",
                notional=requested_notional,
                qty=desired_qty,
                clip_reasons=clip_reasons,
                snapshot=snapshot,
            )
        if halt_if_data_stale_seconds > 0 and symbol_state.get("last_bar_ts"):
            last_bar_ts = int(symbol_state["last_bar_ts"])
            if now_ts - last_bar_ts > halt_if_data_stale_seconds:
                snapshot["last_bar_ts"] = last_bar_ts
                return _blocked_response(
                    reason="data_stale",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
        if halt_if_vol_zscore_gt > 0 and symbol_state.get("vol_zscore") is not None:
            vol_val = float(symbol_state["vol_zscore"])
            snapshot["vol_zscore"] = vol_val
            if abs(vol_val) > halt_if_vol_zscore_gt:
                return _blocked_response(
                    reason="volatility_halt",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
        if cooldown_after_loss_min > 0 and portfolio_state.get("last_loss_ts"):
            last_loss_ts = int(portfolio_state["last_loss_ts"])
            snapshot["last_loss_ts"] = last_loss_ts
            if now_ts - last_loss_ts < cooldown_after_loss_min * 60:
                return _blocked_response(
                    reason="cooldown_after_loss",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
        if cooldown_after_exit_min > 0 and symbol_state.get("last_exit_ts"):
            last_exit_ts = int(symbol_state["last_exit_ts"])
            snapshot["last_exit_ts"] = last_exit_ts
            if now_ts - last_exit_ts < cooldown_after_exit_min * 60:
                return _blocked_response(
                    reason="cooldown_after_exit",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )

    # Exits are allowed while halted unless explicitly disabled.
    if not is_entry and not allow_exits_during_halt and (portfolio_state.get("safe_mode") or portfolio_state.get("kill_switch")):
        return _blocked_response(
            reason="halted_exit_blocked",
            notional=requested_notional,
            qty=desired_qty,
            clip_reasons=clip_reasons,
            snapshot=snapshot,
        )

    final_notional = requested_notional
    final_qty = desired_qty

    # Apply exposure caps only for entries.
    if is_entry:
        symbol_cap = _choose_cap(
            [
                _safe_float(symbol_cfg.get("max_symbol_notional")),
                _safe_float(risk_cfg.get("max_symbol_notional")),
                _safe_float(risk_cfg.get("max_notional_per_symbol")),
            ]
        )
        symbol_weight_cap = None
        weight = symbol_cfg.get("max_symbol_weight", risk_cfg.get("max_symbol_weight"))
        if capital > 0 and weight is not None:
            try:
                weight_f = float(weight)
                if weight_f > 0:
                    symbol_weight_cap = capital * weight_f
            except (TypeError, ValueError):
                symbol_weight_cap = None
        if symbol_cap is not None:
            available_symbol = symbol_cap - current_symbol_notional
            snapshot["symbol_notional_cap"] = symbol_cap
            snapshot["available_symbol"] = available_symbol
            if available_symbol <= 0:
                return _blocked_response(
                    reason="max_symbol_notional",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
            final_notional = min(final_notional, available_symbol)
            if final_notional < requested_notional:
                clip_reasons.append("max_symbol_notional")
        if symbol_weight_cap is not None:
            available_weight = symbol_weight_cap - current_symbol_notional
            snapshot["symbol_weight_cap"] = symbol_weight_cap
            if available_weight <= 0:
                return _blocked_response(
                    reason="max_symbol_weight",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
            if final_notional > available_weight:
                final_notional = available_weight
                clip_reasons.append("max_symbol_weight")

        gross_cap = capital * max_gross_leverage if capital and max_gross_leverage else None
        net_cap = capital * max_net_exposure if capital and max_net_exposure else None
        if gross_cap is not None:
            snapshot["gross_cap"] = gross_cap
            available_gross = gross_cap - gross_exposure
            if available_gross <= 0:
                return _blocked_response(
                    reason="max_gross_exposure",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
            if final_notional > available_gross:
                final_notional = available_gross
                clip_reasons.append("max_gross_exposure")
        if net_cap is not None:
            snapshot["net_cap"] = net_cap
            available_net = net_cap - net_exposure
            if available_net <= 0:
                return _blocked_response(
                    reason="max_net_exposure",
                    notional=requested_notional,
                    qty=desired_qty,
                    clip_reasons=clip_reasons,
                    snapshot=snapshot,
                )
            if final_notional > available_net:
                final_notional = available_net
                clip_reasons.append("max_net_exposure")

    # Translate notional to quantity when price is available.
    if price and (final_qty is None or final_qty <= 0):
        final_qty = final_notional / float(price) if final_notional and float(price) > 0 else final_qty

    if qty_step:
        rounded_qty = _round_down(final_qty or 0.0, qty_step)
        if rounded_qty != final_qty:
            clip_reasons.append("exchange_qty_step")
        final_qty = rounded_qty

    # If exchange step rounding would clip an entry to zero, attempt a small bump to the minimum
    # quantity step (subject to risk caps). This prevents deadlocks where loss-scaling pushes
    # orders below exchange precision.
    if is_entry and (final_qty is None or final_qty <= 0) and qty_step and price and float(price) > 0 and requested_notional > 0:
        try:
            step_qty = float(qty_step)
        except (TypeError, ValueError):
            step_qty = 0.0
        if step_qty > 0:
            min_notional_for_step = step_qty * float(price)
            required_notional = max(min_notional_for_step, float(min_trade_notional or 0.0))
            bump_ratio = required_notional / float(requested_notional) if requested_notional else float("inf")
            if bump_ratio <= 2.0:
                # Enforce the strictest cap across symbol + portfolio exposure limits.
                max_allowed = float("inf")
                if symbol_cap is not None:
                    max_allowed = min(max_allowed, max(0.0, symbol_cap - current_symbol_notional))
                weight = symbol_cfg.get("max_symbol_weight", risk_cfg.get("max_symbol_weight"))
                if capital > 0 and weight is not None:
                    try:
                        weight_f = float(weight)
                    except (TypeError, ValueError):
                        weight_f = 0.0
                    if weight_f > 0:
                        max_allowed = min(max_allowed, max(0.0, capital * weight_f - current_symbol_notional))
                gross_cap = capital * max_gross_leverage if capital and max_gross_leverage else None
                if gross_cap is not None:
                    max_allowed = min(max_allowed, max(0.0, gross_cap - gross_exposure))
                net_cap = capital * max_net_exposure if capital and max_net_exposure else None
                if net_cap is not None:
                    max_allowed = min(max_allowed, max(0.0, net_cap - net_exposure))

                if required_notional > 0 and required_notional <= max_allowed:
                    required_qty = required_notional / float(price)
                    steps = math.ceil(required_qty / step_qty)
                    bumped_qty = steps * step_qty
                    bumped_notional = bumped_qty * float(price)
                    if bumped_qty > 0 and bumped_notional <= max_allowed:
                        final_qty = bumped_qty
                        final_notional = bumped_notional
                        clip_reasons.append("exchange_qty_step_floor")
    if price_tick and price and price_tick > 0:
        rounded_price = _round_down(float(price), price_tick)
        if rounded_price > 0:
            price = rounded_price
            if final_notional and (desired_notional or desired_qty):
                final_notional = final_qty * price if final_qty is not None else final_notional

    if final_notional is None and final_qty and price:
        final_notional = final_qty * float(price)

    if min_trade_notional and final_notional is not None and final_notional > 0 and final_notional < min_trade_notional:
        return _blocked_response(
            reason="risk_clip_to_zero",
            notional=final_notional,
            qty=final_qty,
            clip_reasons=clip_reasons or ["min_trade_notional"],
            snapshot=snapshot,
        )

    if final_qty is not None and final_qty <= 0:
        return _blocked_response(
            reason="risk_clip_to_zero",
            notional=final_notional,
            qty=final_qty,
            clip_reasons=clip_reasons or ["zero_qty"],
            snapshot=snapshot,
        )

    snapshot.update(
        {
            "final_notional": final_notional,
            "final_qty": final_qty,
            "min_trade_notional": min_trade_notional,
            "clip_reasons": list(clip_reasons),
        }
    )

    return {
        "allowed": True,
        "final_notional": final_notional,
        "final_qty": final_qty,
        "block_reason": None,
        "clip_reasons": clip_reasons,
        "risk_snapshot": snapshot,
    }
