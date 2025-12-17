from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

from redis import Redis

try:  # Optional dependency; only required when Postgres-backed logging is enabled.
    import psycopg  # type: ignore
    from psycopg import sql  # type: ignore
except Exception:  # pragma: no cover - psycopg may be absent in certain deployments
    psycopg = None  # type: ignore
    sql = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from app.trading.executor import OrderDecision

logger = logging.getLogger("app.trading.audit")


async def _run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


def _ensure_utc(ts: Optional[datetime]) -> datetime:
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _json_ready(value: Any) -> Any:
    """
    Ensure payload fragments are JSON serializable. Non-serializable types are coerced to string.
    """
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {k: _json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_ready(v) for v in list(value)]
        if isinstance(value, datetime):
            return _ensure_utc(value).isoformat()
        return str(value)


def _hashable_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _next_seq(seq: int) -> int:
    return seq + 1


def _compute_hmac_entry(entry: Dict[str, Any], key: bytes) -> str:
    body = {
        "audit_run_id": entry.get("audit_run_id"),
        "audit_seq": entry.get("audit_seq"),
        "occurred_at": entry.get("occurred_at"),
        "event_type": entry.get("event_type"),
        "model": entry.get("model"),
        "symbol": entry.get("symbol"),
        "payload": entry.get("payload"),
    }
    digest = hmac.new(key, _hashable_payload(_json_ready(body)).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


class TradingAuditLogger:
    """
    Persist audit events (gate toggles, trade submissions) to Redis streams, Postgres, or flat files.
    """

    def __init__(
        self,
        *,
        backend: str = "redis",
        redis_url: Optional[str] = None,
        redis_stream: str = "trading:audit",
        redis_maxlen: int = 10000,
        postgres_dsn: Optional[str] = None,
        postgres_table: str = "trading_audit_events",
        file_path: Optional[Path] = None,
        audit_source: str = "runtime",
        audit_run_id: Optional[str] = None,
        audit_host_id: Optional[str] = None,
        hmac_key: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.redis_url = redis_url
        self.redis_stream = redis_stream
        self.redis_maxlen = max(0, int(redis_maxlen))
        self.postgres_dsn = postgres_dsn
        self.postgres_table = postgres_table
        self.file_path = file_path
        self.audit_source = audit_source or "runtime"
        self.audit_run_id = audit_run_id or os.getenv("TRADING_AUDIT_RUN_ID") or str(uuid.uuid4())
        self.audit_host_id = audit_host_id or os.getenv("TRADING_AUDIT_HOST_ID") or socket.gethostname()
        env_hmac = hmac_key or os.getenv("TRADING_AUDIT_HMAC_KEY")
        self._hmac_key: Optional[bytes] = env_hmac.encode("utf-8") if env_hmac else None
        self._seq: int = 0

        if self.backend not in {"redis", "postgres", "file"}:
            raise ValueError(f"Unsupported trading audit backend '{self.backend}'")
        if self.backend == "redis" and not self.redis_url:
            raise ValueError("Redis URL must be provided when using redis backend for audit logging")
        if self.backend == "postgres" and not self.postgres_dsn:
            raise ValueError("Postgres DSN must be provided when using postgres backend for audit logging")
        if self.backend == "file":
            if self.file_path is None:
                raise ValueError("Audit file path must be provided when using file backend for audit logging")
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self._redis: Optional[Redis] = None
        self._pg_conn = None

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #
    def _ensure_redis(self) -> Redis:
        if self._redis is None:
            assert self.redis_url is not None
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _ensure_pg_conn(self):
        if psycopg is None or sql is None:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg is required for postgres-backed audit logging")
        if self._pg_conn is None:
            assert self.postgres_dsn is not None
            self._pg_conn = psycopg.connect(self.postgres_dsn)
            self._pg_conn.autocommit = True
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            id BIGSERIAL PRIMARY KEY,
                            occurred_at TIMESTAMPTZ NOT NULL,
                            model TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            payload JSONB NOT NULL
                        )
                        """
                    ).format(table=sql.Identifier(self.postgres_table))
                )
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {table} (occurred_at DESC, model, symbol)"
                    ).format(
                        idx=sql.Identifier(f"{self.postgres_table}_occurred_idx"),
                        table=sql.Identifier(self.postgres_table),
                    )
                )
        return self._pg_conn

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def log_gate_toggle(
        self,
        *,
        model: str,
        symbol: str,
        timestamp: datetime,
        gate_pass: bool,
        probability: float,
        threshold: float,
        previous_gate: Optional[bool],
    ) -> None:
        payload = {
            "gate_pass": bool(gate_pass),
            "probability": float(probability),
            "threshold": float(threshold),
            "previous_gate": previous_gate if previous_gate is None else bool(previous_gate),
        }
        await self._log_event("gate_toggle", model, symbol, timestamp, payload)

    async def log_trade(
        self,
        *,
        model: str,
        symbol: str,
        timeframe: Optional[str] = None,
        timestamp: datetime,
        side: str,
        gate_pass: bool,
        probability: float,
        threshold: float,
        decision: "OrderDecision",
        policy_id: Optional[str] = None,
        risk_payload: Optional[Dict[str, Any]] = None,
        decision_namespace: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = _ensure_utc(timestamp)
        payload: Dict[str, Any] = {
            "side": side,
            "gate_pass": bool(gate_pass),
            "probability": float(probability),
            "prob": float(probability),
            "threshold": float(threshold),
            "executed": bool(decision.executed),
            "price_used": decision.price_used,
            "amount": decision.amount,
            "spread_bps": decision.spread_bps,
            "reason": decision.reason or "",
            "decision": decision.reason or "",
            "skip_reason": decision.reason or "",
            "shadow_mode": bool(getattr(decision, "shadow_mode", False)),
            "blocked_reason": getattr(decision, "blocked_reason", None) or decision.reason or "",
            "notional": getattr(decision, "notional", None),
            "intended_notional": getattr(decision, "notional", None),
            "policy_id": policy_id or getattr(decision, "policy_id", None) or "",
            "order_intent_id": getattr(decision, "order_intent_id", None) or "",
            "intent_status": getattr(decision, "intent_status", None) or "",
            "dedup_blocked": bool(getattr(decision, "dedup_blocked", False)),
            "exchange_order_id": getattr(decision, "exchange_order_id", None) or "",
            "ts": ts.isoformat(),
            "timeframe": timeframe or "",
        }
        if decision_namespace:
            payload["decision_namespace"] = decision_namespace
        payload.update(
            {
                "risk_allowed": None if risk_payload is None else bool(risk_payload.get("allowed")),
                "risk_block_reason": None if risk_payload is None else risk_payload.get("block_reason"),
                "risk_clip_reasons": [] if risk_payload is None else list(risk_payload.get("clip_reasons") or []),
                "risk_final_notional": None if risk_payload is None else risk_payload.get("final_notional"),
                "risk_final_qty": None if risk_payload is None else risk_payload.get("final_qty"),
                "risk_snapshot": _json_ready(risk_payload.get("risk_snapshot") or {}) if risk_payload else {},
            }
        )
        if decision.order_payload is not None:
            payload["order_payload"] = _json_ready(decision.order_payload)
        if extra:
            payload.update(_json_ready(extra))
        await self._log_event("trade", model, symbol, ts, payload)

    async def log_safe_mode(self, *, reason: str, active: bool, timestamp: Optional[datetime] = None) -> None:
        ts = _ensure_utc(timestamp)
        payload = {"reason": reason, "active": bool(active)}
        await self._log_event("safe_mode", "system", "*", ts, payload)

    async def log_reconciliation(self, *, payload: Dict[str, Any], timestamp: Optional[datetime] = None) -> None:
        ts = _ensure_utc(timestamp)
        await self._log_event("reconciliation", "system", "*", ts, payload)

    async def log_deadlock_status(
        self,
        *,
        triggers: Sequence[object],
        portfolio: Mapping[str, object],
        timestamp: Optional[datetime] = None,
        policy_hash: Optional[str] = None,
    ) -> None:
        ts = _ensure_utc(timestamp)
        symbols = []
        for status in triggers:
            symbols.append(
                {
                    "symbol": getattr(status, "symbol", ""),
                    "model": getattr(status, "model", ""),
                    "coverage_ratio": getattr(status, "coverage_ratio", 0.0),
                    "prob_gate_pass_ratio": getattr(status, "prob_gate_pass_ratio", 0.0),
                    "trade_count": getattr(status, "trade_count", 0),
                    "samples": getattr(status, "samples", 0),
                    "prob_quantiles": getattr(status, "prob_quantiles", {}),
                    "block_reasons": getattr(status, "block_reasons", {}),
                    "prob_gate_min_used": getattr(status, "prob_gate_min_used", None),
                    "window": getattr(status, "window_label", ""),
                    "timeframe": getattr(status, "timeframe", ""),
                }
            )
        payload = {
            "policy_hash": policy_hash,
            "window": symbols[0]["window"] if symbols else "",
            "portfolio": dict(portfolio),
            "symbols": symbols,
        }
        await self._log_event("deadlock_status", "system", "*", ts, payload)

    async def log_deadlock_action(
        self,
        *,
        action: str,
        timestamp: Optional[datetime] = None,
        payload: Mapping[str, object],
        policy_hash: Optional[str] = None,
    ) -> None:
        ts = _ensure_utc(timestamp)
        entry_payload = {"action": action, "payload": _json_ready(payload), "policy_hash": policy_hash}
        await self._log_event("deadlock_action", "system", "*", ts, entry_payload)

    async def close(self) -> None:
        if self._redis is not None:
            await _run_in_executor(self._redis.close)
            self._redis = None
        if self._pg_conn is not None:
            await _run_in_executor(self._pg_conn.close)
            self._pg_conn = None

    # ------------------------------------------------------------------ #
    # Backend writers
    # ------------------------------------------------------------------ #
    async def _log_event(
        self,
        event_type: str,
        model: str,
        symbol: str,
        timestamp: Optional[datetime],
        payload: Dict[str, Any],
    ) -> None:
        ts = _ensure_utc(timestamp)
        self._seq = _next_seq(self._seq)
        audit_meta = {
            "audit_source": self.audit_source,
            "audit_run_id": self.audit_run_id,
            "audit_host_id": self.audit_host_id,
            "audit_seq": self._seq,
        }
        payload_with_meta = _json_ready(payload)
        if isinstance(payload_with_meta, dict):
            payload_with_meta = dict(payload_with_meta)
        else:
            payload_with_meta = {"value": payload_with_meta}
        payload_with_meta.setdefault("audit_meta", audit_meta)
        entry = {
            "occurred_at": ts.isoformat(),
            "event_type": event_type,
            "model": model,
            "symbol": symbol,
            "payload": payload_with_meta,
            **audit_meta,
        }
        if self._hmac_key:
            entry["audit_hmac"] = _compute_hmac_entry(entry, self._hmac_key)
        try:
            if self.backend == "redis":
                await _run_in_executor(self._write_redis, entry)
            elif self.backend == "postgres":
                await _run_in_executor(self._write_postgres, ts, model, symbol, event_type, entry["payload"])
            elif self.backend == "file":
                await _run_in_executor(self._write_file, entry)
        except Exception as exc:
            logger.exception("Failed to persist trading audit event (%s backend): %s", self.backend, exc)

    def _write_redis(self, entry: Dict[str, Any]) -> None:
        client = self._ensure_redis()
        body = json.dumps(entry, sort_keys=True)
        kwargs: Dict[str, Any] = {}
        if self.redis_maxlen:
            kwargs["maxlen"] = self.redis_maxlen
            kwargs["approximate"] = True
        client.xadd(self.redis_stream, {"event": body}, **kwargs)

    def _write_postgres(
        self,
        ts: datetime,
        model: str,
        symbol: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        conn = self._ensure_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (occurred_at, model, symbol, event_type, payload)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """
                ).format(table=sql.Identifier(self.postgres_table)),
                (ts, model, symbol, event_type, json.dumps(_json_ready(payload))),
            )

    def _write_file(self, entry: Dict[str, Any]) -> None:
        assert self.file_path is not None
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True))
            fh.write("\n")
