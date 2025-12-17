from datetime import datetime, timezone

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
    assert exit_outcome.should_exit is False
    assert exit_outcome.exit_armed is True
    assert exit_outcome.exit_blocked_by_pnl is True
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
