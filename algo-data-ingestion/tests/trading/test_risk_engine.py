from datetime import datetime, timezone

import pytest

from app.trading.risk import assess_and_adjust_order


def _risk_cfg() -> dict:
    return {
        "capital": 1_000.0,
        "max_gross_leverage": 2.0,
        "max_net_exposure": 1.0,
        "max_turnover_per_day": 1.0,
        "max_orders_per_hour": 10,
        "max_concurrent_positions": 5,
        "daily_loss_limit_pct": 0.03,
        "max_drawdown_pct": 0.2,
        "cooldown_minutes_after_exit": 5,
        "cooldown_minutes_after_loss": 10,
        "halt_on_safe_mode": True,
        "allow_exits_during_halt": True,
        "halt_if_spread_bps_gt": 20.0,
        "halt_if_vol_zscore_gt": 5.0,
        "halt_if_missing_price_bars": False,
        "halt_if_data_stale_seconds": 120,
        "symbols": {
            "BTC/USDT": {
                "max_symbol_notional": 100.0,
                "max_symbol_weight": 0.5,
                "max_spread_bps": 15.0,
                "min_trade_notional": 5.0,
                "qty_step": 0.001,
                "price_tick": 0.01,
            }
        },
    }


def _portfolio_state(**overrides) -> dict:
    base = {
        "capital": 1_000.0,
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "open_symbols": [],
        "open_positions": 0,
        "turnover_1d": 0.0,
        "orders_last_hour": 0,
        "daily_pnl_pct": 0.0,
        "drawdown_pct": 0.0,
        "last_loss_ts": None,
        "safe_mode": False,
        "kill_switch": False,
        "reconciliation_latched": False,
    }
    base.update(overrides)
    return base


def _symbol_state(**overrides) -> dict:
    base = {
        "symbol": "BTC/USDT",
        "in_position": False,
        "open_notional": 0.0,
        "open_qty": 0.0,
        "last_exit_ts": None,
        "last_entry_ts": None,
        "last_bar_ts": None,
        "missing_price_bars": False,
        "vol_zscore": None,
        "qty_step": 0.001,
        "price_tick": 0.01,
        "min_trade_notional": 5.0,
    }
    base.update(overrides)
    return base


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def test_entry_blocked_when_daily_loss_limit_breached():
    risk_cfg = _risk_cfg()
    portfolio_state = _portfolio_state(daily_pnl_pct=-0.05)
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=50.0,
        desired_qty=0.01,
        price=10_000.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=portfolio_state,
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "daily_loss_limit"


def test_entry_clipped_to_per_symbol_max_notional():
    risk_cfg = _risk_cfg()
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=150.0,
        desired_qty=None,
        price=10.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=_portfolio_state(),
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is True
    assert result["final_notional"] == pytest.approx(100.0)
    assert "max_symbol_notional" in result["clip_reasons"]


def test_entry_blocked_when_clipped_below_min_notional():
    risk_cfg = _risk_cfg()
    risk_cfg["symbols"]["BTC/USDT"]["max_symbol_notional"] = 4.0
    risk_cfg["symbols"]["BTC/USDT"]["min_trade_notional"] = 5.0
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=20.0,
        desired_qty=None,
        price=1.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=_portfolio_state(),
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "risk_clip_to_zero"


def test_entry_qty_step_floor_bump_prevents_clip_to_zero():
    risk_cfg = _risk_cfg()
    risk_cfg["symbols"]["BTC/USDT"]["qty_step"] = 0.0001
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=7.5,
        desired_qty=None,
        price=100_000.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=_portfolio_state(),
        symbol_state=_symbol_state(qty_step=0.0001),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is True
    assert result["final_qty"] == pytest.approx(0.0001)
    assert result["final_notional"] == pytest.approx(10.0)
    assert "exchange_qty_step_floor" in result["clip_reasons"]


def test_exit_allowed_during_safe_mode():
    risk_cfg = _risk_cfg()
    portfolio_state = _portfolio_state(safe_mode=True)
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="EXIT_LONG",
        desired_notional=20.0,
        desired_qty=2.0,
        price=10.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=portfolio_state,
        symbol_state=_symbol_state(in_position=True, open_notional=20.0, open_qty=2.0),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is True
    assert result["block_reason"] is None


def test_spread_halt_blocks_entry():
    risk_cfg = _risk_cfg()
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=10.0,
        desired_qty=None,
        price=1.0,
        spread_bps=50.0,
        now_ts=_now_ts(),
        portfolio_state=_portfolio_state(),
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "spread_too_wide"


def test_cooldown_blocks_reentry_after_exit():
    risk_cfg = _risk_cfg()
    risk_cfg["cooldown_minutes_after_exit"] = 5
    now_ts = _now_ts()
    symbol_state = _symbol_state(last_exit_ts=now_ts - 60)
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=10.0,
        desired_qty=1.0,
        price=10.0,
        spread_bps=1.0,
        now_ts=now_ts,
        portfolio_state=_portfolio_state(),
        symbol_state=symbol_state,
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "cooldown_after_exit"


def test_portfolio_caps_block_second_symbol_when_at_limit():
    risk_cfg = _risk_cfg()
    risk_cfg["max_gross_leverage"] = 1.0
    risk_cfg["max_net_exposure"] = 1.0
    portfolio_state = _portfolio_state(
        capital=1_000.0,
        gross_exposure=1_000.0,
        net_exposure=1_000.0,
        open_symbols=["ETH/USDT"],
        open_positions=1,
    )
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=100.0,
        desired_qty=0.01,
        price=10_000.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=portfolio_state,
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "max_gross_exposure"


def test_turnover_limit_blocks_combined_symbols():
    risk_cfg = _risk_cfg()
    risk_cfg["max_turnover_per_day"] = 0.5
    risk_cfg["symbols"]["BTC/USDT"]["max_symbol_notional"] = 2_000.0
    risk_cfg["symbols"]["BTC/USDT"]["max_symbol_weight"] = 1.0
    portfolio_state = _portfolio_state(
        capital=2_000.0,
        gross_exposure=500.0,
        net_exposure=500.0,
        turnover_1d=1_100.0,
    )
    result = assess_and_adjust_order(
        symbol="BTC/USDT",
        action="ENTER_LONG",
        desired_notional=900.0,
        desired_qty=None,
        price=9_000.0,
        spread_bps=1.0,
        now_ts=_now_ts(),
        portfolio_state=portfolio_state,
        symbol_state=_symbol_state(),
        risk_cfg=risk_cfg,
    )
    assert result["allowed"] is False
    assert result["block_reason"] == "turnover_limit"
