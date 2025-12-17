from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _now(ts: Optional[datetime] = None) -> datetime:
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass
class DeadlockPolicy:
    enabled: bool = False
    window_minutes: int = 60
    min_trades_window: int = 1
    min_coverage_ratio_window: float = 0.01
    cooldown_minutes: int = 30
    max_actions_per_day: int = 3
    actions: List[Mapping[str, object]] = field(default_factory=list)
    adjust_floor: Optional[float] = None
    adjust_step: Optional[float] = None
    audit_every_action: bool = True

    @classmethod
    def from_payload(cls, payload: Optional[Mapping[str, object]]) -> "DeadlockPolicy":
        if not isinstance(payload, Mapping):
            return cls()
        actions = list(payload.get("actions") or [])
        adjust_cfg = payload.get("adjust_prob_gate_min") or {}
        floor = adjust_cfg.get("floor")
        step = adjust_cfg.get("step") or adjust_cfg.get("delta") or adjust_cfg.get("adjust")
        return cls(
            enabled=bool(payload.get("enabled", False)),
            window_minutes=int(payload.get("window_minutes") or 60),
            min_trades_window=int(payload.get("min_trades_window") or 1),
            min_coverage_ratio_window=float(payload.get("min_coverage_ratio_window") or 0.0),
            cooldown_minutes=int(payload.get("cooldown_minutes") or 30),
            max_actions_per_day=int(payload.get("max_actions_per_day") or 3),
            actions=actions,
            adjust_floor=None if floor is None else float(floor),
            adjust_step=None if step is None else float(step),
            audit_every_action=bool(payload.get("audit_every_action", True)),
        )

    def config_hash(self) -> str:
        payload = {
            "enabled": self.enabled,
            "window_minutes": self.window_minutes,
            "min_trades_window": self.min_trades_window,
            "min_coverage_ratio_window": self.min_coverage_ratio_window,
            "cooldown_minutes": self.cooldown_minutes,
            "max_actions_per_day": self.max_actions_per_day,
            "actions": list(self.actions),
            "adjust_floor": self.adjust_floor,
            "adjust_step": self.adjust_step,
            "audit_every_action": self.audit_every_action,
        }
        encoded = repr(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class DeadlockStatus:
    model: str
    symbol: str
    window_label: str
    timeframe: Optional[str]
    coverage_ratio: float
    prob_gate_pass_ratio: float
    trade_count: int
    samples: int
    prob_quantiles: Dict[str, float]
    block_reasons: Dict[str, int]
    prob_gate_min_used: Optional[float] = None

    @property
    def is_deadlocked(self) -> bool:
        return self.trade_count <= 0 or self.coverage_ratio <= 0.0 or self.prob_gate_pass_ratio <= 0.0


class DeadlockMonitor:
    """
    Track rolling coverage/trade metrics to surface deadlocks.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        self.window = timedelta(minutes=max(1, int(window_minutes)))
        self.window_label = f"{int(self.window.total_seconds() // 60)}m"
        self._gate_events: Dict[Tuple[str, str], Deque[Mapping[str, object]]] = defaultdict(deque)
        self._trade_events: Dict[Tuple[str, str], Deque[Mapping[str, object]]] = defaultdict(deque)

    def record_gate(
        self,
        *,
        model: str,
        symbol: str,
        ts: datetime,
        gate_pass: bool,
        probability: float,
        prob_gate_min: Optional[float],
        timeframe: Optional[str] = None,
    ) -> None:
        key = (model, symbol)
        event = {
            "ts": _now(ts),
            "gate_pass": bool(gate_pass),
            "prob": float(probability),
            "prob_gate_min": None if prob_gate_min is None else float(prob_gate_min),
            "timeframe": timeframe or "",
        }
        if event["prob_gate_min"] is not None:
            event["prob_gate_pass"] = float(probability) >= float(event["prob_gate_min"])
        self._gate_events[key].append(event)
        self._prune(key)

    def record_trade(
        self,
        *,
        model: str,
        symbol: str,
        ts: datetime,
        executed: bool,
        blocked_reason: Optional[str] = None,
    ) -> None:
        key = (model, symbol)
        event = {
            "ts": _now(ts),
            "executed": bool(executed),
        }
        if blocked_reason:
            event["reason"] = str(blocked_reason)
        self._trade_events[key].append(event)
        self._prune(key)

    def _prune(self, key: Tuple[str, str]) -> None:
        cutoff = _now() - self.window
        gates = self._gate_events.get(key, deque())
        trades = self._trade_events.get(key, deque())
        while gates and gates[0]["ts"] < cutoff:
            gates.popleft()
        while trades and trades[0]["ts"] < cutoff:
            trades.popleft()

    def snapshot(self) -> Tuple[List[DeadlockStatus], Dict[str, float]]:
        statuses: List[DeadlockStatus] = []
        portfolio_coverage: List[float] = []
        total_trades = 0

        for key in set(self._gate_events.keys()) | set(self._trade_events.keys()):
            model, symbol = key
            gates = self._gate_events.get(key, deque())
            trades = self._trade_events.get(key, deque())
            samples = len(gates)
            coverage_ratio = float(sum(1 for g in gates if g.get("gate_pass"))) / float(samples or 1)
            prob_pass_samples = [g for g in gates if g.get("prob_gate_pass") is not None]
            prob_gate_pass_ratio = (
                float(sum(1 for g in prob_pass_samples if g.get("prob_gate_pass"))) / float(len(prob_pass_samples))
                if prob_pass_samples
                else 0.0
            )
            probs = np.array([float(g.get("prob", 0.0)) for g in gates], dtype=float)
            prob_quantiles = {
                "p50": float(np.nanquantile(probs, 0.5)) if len(probs) else 0.0,
                "p90": float(np.nanquantile(probs, 0.9)) if len(probs) else 0.0,
                "p95": float(np.nanquantile(probs, 0.95)) if len(probs) else 0.0,
                "p99": float(np.nanquantile(probs, 0.99)) if len(probs) else 0.0,
            }
            block_counts: Counter[str] = Counter()
            executed_trades = 0
            for trade in trades:
                if trade.get("executed"):
                    executed_trades += 1
                reason = trade.get("reason")
                if reason and not trade.get("executed"):
                    block_counts[str(reason)] += 1
            prob_gate_min_used = None
            mins = [g.get("prob_gate_min") for g in gates if g.get("prob_gate_min") is not None]
            if mins:
                prob_gate_min_used = float(np.nanmedian(np.array(mins, dtype=float)))
            trade_count = executed_trades
            total_trades += executed_trades
            portfolio_coverage.append(coverage_ratio)
            timeframe = None
            if gates:
                timeframe = str(gates[-1].get("timeframe") or "")
            statuses.append(
                DeadlockStatus(
                    model=model,
                    symbol=symbol,
                    window_label=self.window_label,
                    timeframe=timeframe,
                    coverage_ratio=coverage_ratio,
                    prob_gate_pass_ratio=prob_gate_pass_ratio,
                    trade_count=trade_count,
                    samples=samples,
                    prob_quantiles=prob_quantiles,
                    block_reasons=dict(block_counts),
                    prob_gate_min_used=prob_gate_min_used,
                )
            )

        portfolio_metrics = {
            "trade_count": total_trades,
            "coverage_ratio": float(sum(portfolio_coverage) / float(len(portfolio_coverage))) if portfolio_coverage else 0.0,
        }
        return statuses, portfolio_metrics
