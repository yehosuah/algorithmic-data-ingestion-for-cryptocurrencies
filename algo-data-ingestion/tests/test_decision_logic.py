from datetime import datetime, timedelta, timezone

from app.trading.decision import TriggerConfig, decide_bar
from app.trading.state import PositionState


def test_decide_bar_entry_and_exit_prob_floor():
    cfg = TriggerConfig(
        entry_threshold=0.6,
        exit_threshold=0.5,
        exit_prob_drop=0.15,
        min_hold_bars=1,
        bar_seconds=60,
    )
    state = PositionState()
    ts = datetime.now(timezone.utc)

    outcome = decide_bar(
        ts=ts,
        probability=0.61,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=100.0,
    )
    assert outcome.should_enter is True

    # simulate entry metadata
    state.metadata["open_price"] = "100.0"
    state.metadata["open_entry_prob"] = "0.61"
    state.in_position = True

    exit_outcome = decide_bar(
        ts=ts,
        probability=0.49,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=99.0,
        entry_price=100.0,
        entry_amount=1.0,
    )
    assert exit_outcome.should_exit is True
    assert exit_outcome.exit_armed is True
    assert exit_outcome.exit_blocked_by_pnl is False
    assert exit_outcome.exit_reason_primary in {"prob_floor", "prob_trailing", "stop_loss"}
    assert "prob_floor" in set(exit_outcome.exit_reasons)
    assert exit_outcome.exit_context.get("pnl_gross") is not None


def test_decide_bar_positive_pnl_direction():
    cfg = TriggerConfig(
        entry_threshold=0.5,
        exit_threshold=0.49,
        exit_prob_drop=0.1,
        min_hold_bars=1,
        bar_seconds=60,
    )
    state = PositionState(in_position=True, entry_ts=datetime.now(timezone.utc))
    state.metadata["open_price"] = "100"
    state.metadata["open_amount"] = "2"
    state.metadata["open_entry_prob"] = "0.7"
    outcome = decide_bar(
        ts=datetime.now(timezone.utc),
        probability=0.45,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=101.0,
        entry_price=100.0,
        entry_amount=2.0,
    )
    assert outcome.exit_armed is True
    assert outcome.exit_context.get("pnl_gross") and outcome.exit_context["pnl_gross"] > 0


def test_decide_bar_spread_guard_defers_non_risk_exits_but_allows_stop_loss():
    cfg = TriggerConfig(
        entry_threshold=0.6,
        exit_threshold=0.5,
        exit_prob_drop=0.15,
        min_hold_bars=1,
        bar_seconds=60,
        max_spread_bps=5.0,
        stop_loss_pct=0.005,
    )
    ts = datetime.now(timezone.utc)

    state = PositionState()
    entry_outcome = decide_bar(
        ts=ts,
        probability=0.61,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=100.0,
        spread_bps=50.0,
    )
    assert entry_outcome.should_enter is True
    assert entry_outcome.skip_execution is True
    assert entry_outcome.skip_reason == "spread_threshold"

    state.in_position = True
    state.entry_ts = ts
    state.hold_until = None
    state.metadata["open_price"] = "100.0"
    state.metadata["open_amount"] = "1.0"
    state.metadata["open_entry_prob"] = "0.61"
    exit_outcome = decide_bar(
        ts=ts,
        probability=0.49,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=99.9,
        entry_price=100.0,
        entry_amount=1.0,
        spread_bps=50.0,
    )
    assert exit_outcome.should_exit is False
    assert exit_outcome.exit_armed is True
    assert exit_outcome.skip_execution is True
    assert exit_outcome.skip_reason == "spread_threshold"

    stop_outcome = decide_bar(
        ts=ts,
        probability=0.49,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=99.0,
        entry_price=100.0,
        entry_amount=1.0,
        spread_bps=50.0,
    )
    assert stop_outcome.should_exit is True
    assert stop_outcome.exit_trigger == "stop_loss"
    assert stop_outcome.skip_execution is False


def test_decide_bar_take_profit_allowed_during_hold():
    cfg = TriggerConfig(
        entry_threshold=0.6,
        exit_threshold=0.5,
        exit_prob_drop=0.15,
        min_hold_bars=10,
        bar_seconds=60,
        stop_loss_pct=0.005,
        take_profit_pct=0.002,
    )
    ts = datetime.now(timezone.utc)
    state = PositionState(in_position=True, entry_ts=ts)
    state.hold_until = ts + timedelta(seconds=cfg.min_hold_bars * cfg.bar_seconds)
    state.metadata["open_price"] = "100.0"
    state.metadata["open_amount"] = "1.0"
    state.metadata["open_entry_prob"] = "0.61"

    outcome = decide_bar(
        ts=ts,
        probability=0.7,
        gate_pass=True,
        state=state,
        cfg=cfg,
        current_price=100.25,
        entry_price=100.0,
        entry_amount=1.0,
    )
    assert outcome.should_exit is True
    assert outcome.exit_trigger == "take_profit"
    assert outcome.exit_blocked_by_hold is False
    assert "take_profit" in set(outcome.exit_reasons)
