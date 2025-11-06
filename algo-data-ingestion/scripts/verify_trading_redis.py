#!/usr/bin/env python3
"""
Utility script to peek at the trading service state/audit data stored in Redis.

Usage:
    python scripts/verify_trading_redis.py

The script honours the same environment variables consumed by the trading service:
    - TRADING_STATE_REDIS_URL (falls back to DECISION_QUEUE_URL or redis://localhost:6379/0)
    - TRADING_STATE_REDIS_HASH (defaults to trading:positions)
    - TRADING_AUDIT_REDIS_URL (falls back to TRADING_STATE_REDIS_URL)
    - TRADING_AUDIT_STREAM (defaults to trading:audit)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Tuple

from redis import Redis


def _connect(url: str) -> Redis:
    return Redis.from_url(url, decode_responses=True)


def _pretty_json(payload: str) -> str:
    try:
        return json.dumps(json.loads(payload), indent=2, sort_keys=True)
    except Exception:
        return payload


def inspect_state(client: Redis, hash_key: str) -> None:
    print(f"[state] inspecting hash '{hash_key}'")
    if not client.exists(hash_key):
        print("  hash does not exist yet (no positions persisted)\n")
        return
    entries: Dict[str, str] = client.hgetall(hash_key)  # type: ignore[assignment]
    print(f"  entries: {len(entries)}")
    for symbol, raw in entries.items():
        print(f"  ├─ {symbol}")
        print(_pretty_json(raw))
    print()


def inspect_audit(client: Redis, stream_key: str, count: int = 5) -> None:
    print(f"[audit] inspecting stream '{stream_key}' (last {count} entries)")
    if not client.exists(stream_key):
        print("  stream does not exist yet (no audit events emitted)\n")
        return
    messages = client.xrevrange(stream_key, count=count)  # type: ignore[assignment]
    if not messages:
        print("  stream is empty\n")
        return
    for message_id, data in messages:
        payload = data.get("event")
        print(f"  ├─ id={message_id}")
        if payload is None:
            print(f"    raw={data}")
        else:
            print(_pretty_json(payload))
    print()


def main() -> int:
    state_url = os.getenv("TRADING_STATE_REDIS_URL") or os.getenv("DECISION_QUEUE_URL") or "redis://localhost:6379/0"
    audit_url = os.getenv("TRADING_AUDIT_REDIS_URL") or state_url
    hash_key = os.getenv("TRADING_STATE_REDIS_HASH", "trading:positions")
    stream_key = os.getenv("TRADING_AUDIT_STREAM", "trading:audit")

    try:
        state_client = _connect(state_url)
        audit_client = _connect(audit_url)
    except Exception as exc:  # pragma: no cover - simple CLI helper
        print(f"Failed to create Redis clients: {exc}", file=sys.stderr)
        return 1

    try:
        inspect_state(state_client, hash_key)
        inspect_audit(audit_client, stream_key)
    except Exception as exc:  # pragma: no cover - simple CLI helper

        print(f"Failed reading from Redis: {exc}", file=sys.stderr)
        return 1
    finally:
        for client in {state_client, audit_client}:
            try:
                client.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
