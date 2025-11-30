from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from redis import Redis

try:  # Optional dependency; only required when Postgres persistence is enabled.
    import psycopg  # type: ignore
    from psycopg import sql  # type: ignore
except Exception:  # pragma: no cover - psycopg may be absent in some deployments
    psycopg = None  # type: ignore
    sql = None  # type: ignore

logger = logging.getLogger("app.trading.state")


async def _run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _format_ts(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass
class PositionState:
    in_position: bool = False
    entry_ts: Optional[datetime] = None
    hold_until: Optional[datetime] = None
    last_exit_ts: Optional[datetime] = None
    last_gate: Optional[bool] = None
    last_probability: Optional[float] = None
    last_timestamp: Optional[datetime] = None
    bars_in_position: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def register_bar(self, ts: datetime) -> bool:
        """
        Returns True when the timestamp is newer than the last processed bar.
        """
        if self.last_timestamp and ts <= self.last_timestamp:
            return False
        self.last_timestamp = ts
        if self.in_position:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0
        return True

    def ready_for_entry(self, ts: datetime) -> bool:
        if self.in_position:
            return False
        if self.metadata.get("pending_entry_intent_id"):
            return False
        if self.hold_until is None:
            return True
        return ts >= self.hold_until

    def ready_for_exit(self, ts: datetime) -> bool:
        if not self.in_position:
            return False
        if self.metadata.get("pending_exit_intent_id"):
            return False
        if self.hold_until is None:
            return True
        return ts >= self.hold_until

    def mark_entry(self, ts: datetime, min_hold_seconds: int) -> None:
        self.in_position = True
        self.entry_ts = ts
        self.last_exit_ts = None
        self.bars_in_position = 0
        self.hold_until = ts + timedelta(seconds=min_hold_seconds)

    def mark_exit(self, ts: datetime) -> None:
        self.in_position = False
        self.last_exit_ts = ts
        self.entry_ts = None
        self.bars_in_position = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PositionState":
        return cls(
            in_position=bool(payload.get("in_position", False)),
            entry_ts=_parse_ts(payload.get("entry_ts")),
            hold_until=_parse_ts(payload.get("hold_until")),
            last_exit_ts=_parse_ts(payload.get("last_exit_ts")),
            last_gate=payload.get("last_gate"),
            last_probability=payload.get("last_probability"),
            last_timestamp=_parse_ts(payload.get("last_timestamp")),
            bars_in_position=int(payload.get("bars_in_position") or 0),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, object]:
        raw = asdict(self)
        raw["entry_ts"] = _format_ts(self.entry_ts)
        raw["hold_until"] = _format_ts(self.hold_until)
        raw["last_exit_ts"] = _format_ts(self.last_exit_ts)
        raw["last_timestamp"] = _format_ts(self.last_timestamp)
        return raw


class TradingStateStore:
    """
    Persistence layer for tracking per-symbol trading state across restarts.

    Supports `file` (legacy JSON), `redis` (hash map), and `postgres` (JSONB table) backends.
    """

    def __init__(
        self,
        path: Path,
        *,
        backend: str = "redis",
        redis_url: Optional[str] = None,
        redis_hash: str = "trading:positions",
        postgres_dsn: Optional[str] = None,
        postgres_table: str = "trading_positions",
    ) -> None:
        self.path = path
        self.backend = backend
        self.redis_url = redis_url
        self.redis_hash = redis_hash
        self.postgres_dsn = postgres_dsn
        self.postgres_table = postgres_table

        if self.backend not in {"file", "redis", "postgres"}:
            raise ValueError(f"Unsupported trading state backend '{self.backend}'")
        if self.backend == "redis" and not self.redis_url:
            raise ValueError("Redis URL must be provided when using redis backend for trading state")
        if self.backend == "postgres" and not self.postgres_dsn:
            raise ValueError("Postgres DSN must be provided when using postgres backend for trading state")
        if self.backend == "file":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._state: Dict[str, PositionState] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._redis: Optional[Redis] = None
        self._pg_conn = None
        self._load()

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
            raise RuntimeError("psycopg is required for postgres-backed trading state persistence")
        if self._pg_conn is None:
            assert self.postgres_dsn is not None
            self._pg_conn = psycopg.connect(self.postgres_dsn)
            self._pg_conn.autocommit = True
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            symbol TEXT PRIMARY KEY,
                            state_json JSONB NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    ).format(table=sql.Identifier(self.postgres_table))
                )
        return self._pg_conn

    # ------------------------------------------------------------------ #
    # Load & serialization helpers
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            if self.backend == "file":
                self._load_file()
            elif self.backend == "redis":
                self._load_redis()
            elif self.backend == "postgres":
                self._load_postgres()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed loading trading state from %s backend: %s", self.backend, exc)

    def _load_file(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text())
        positions = payload.get("positions")
        if isinstance(positions, dict):
            for key, data in positions.items():
                try:
                    self._state[key] = PositionState.from_dict(data or {})
                except Exception:
                    continue

    def _load_redis(self) -> None:
        client = self._ensure_redis()
        raw = client.hgetall(self.redis_hash)
        for key, encoded in raw.items():
            try:
                data = json.loads(encoded)
                self._state[key] = PositionState.from_dict(data or {})
            except Exception:
                continue

    def _load_postgres(self) -> None:
        conn = self._ensure_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT symbol, state_json FROM {table}").format(table=sql.Identifier(self.postgres_table))
            )
            rows = cur.fetchall()
        for symbol, state_json in rows:
            try:
                payload = state_json if isinstance(state_json, dict) else json.loads(state_json)
                self._state[str(symbol)] = PositionState.from_dict(payload or {})
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> PositionState:
        state = self._state.get(key)
        if state is None:
            state = PositionState()
            self._state[key] = state
            self._dirty = True
        return state

    def update(self, key: str, state: PositionState) -> None:
        self._state[key] = state
        self._dirty = True

    async def flush(self) -> None:
        if not self._dirty:
            return
        async with self._lock:
            if not self._dirty:
                return
            try:
                if self.backend == "file":
                    payload = {"positions": {key: state.to_dict() for key, state in self._state.items()}}
                    await _run_in_executor(self._write_file, payload)
                elif self.backend == "redis":
                    await _run_in_executor(self._write_redis)
                elif self.backend == "postgres":
                    await _run_in_executor(self._write_postgres)
            except Exception as exc:
                logger.exception("Failed persisting trading state via %s backend: %s", self.backend, exc)
                raise
            else:
                self._dirty = False

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
    def _write_file(self, payload: Dict[str, object]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    def _write_redis(self) -> None:
        client = self._ensure_redis()
        mapping = {key: json.dumps(state.to_dict()) for key, state in self._state.items()}
        pipe = client.pipeline()
        pipe.delete(self.redis_hash)
        if mapping:
            pipe.hset(self.redis_hash, mapping=mapping)
        pipe.execute()

    def _write_postgres(self) -> None:
        conn = self._ensure_pg_conn()
        rows = [(key, json.dumps(state.to_dict())) for key, state in self._state.items()]
        symbols = [key for key, _ in rows]
        with conn.cursor() as cur:
            if not rows:
                cur.execute(sql.SQL("TRUNCATE {table}").format(table=sql.Identifier(self.postgres_table)))
                return
            cur.execute(
                sql.SQL("DELETE FROM {table} WHERE symbol <> ALL(%s)").format(
                    table=sql.Identifier(self.postgres_table)
                ),
                (symbols,),
            )
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (symbol, state_json, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (symbol) DO UPDATE
                    SET state_json = EXCLUDED.state_json,
                        updated_at = EXCLUDED.updated_at
                    """
                ).format(table=sql.Identifier(self.postgres_table)),
                rows,
            )
