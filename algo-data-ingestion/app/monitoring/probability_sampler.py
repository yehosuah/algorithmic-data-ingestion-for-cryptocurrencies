from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # redis is optional during offline tooling/tests
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis not installed
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


@dataclass
class ProbabilitySampleConfig:
    enabled: bool
    file_root: Optional[Path]
    max_rows: int
    redis_url: Optional[str]
    redis_stream: str
    redis_maxlen: int


class ProbabilitySampler:
    """
    Captures a bounded sample of live probability streams for diagnostics.

    Samples are appended to `<file_root>/<model>_<prob>.jsonl` and optionally pushed
    to a Redis stream so downstream tooling can aggregate histograms in near-real-time.
    """

    def __init__(self, config: ProbabilitySampleConfig) -> None:
        self.config = config
        self._redis_client = None
        self._redis_failed = False
        root = config.file_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> ProbabilitySampler:
        enabled = _env_bool("PROB_SAMPLE_ENABLED", True)
        root_env = os.getenv("PROB_SAMPLE_ROOT", "logs/probability_samples")
        file_root = Path(root_env).expanduser().resolve()
        if not enabled:
            file_root = None
        config = ProbabilitySampleConfig(
            enabled=enabled,
            file_root=file_root if enabled else None,
            max_rows=max(10, _env_int("PROB_SAMPLE_MAX_ROWS", 512)),
            redis_url=os.getenv("PROB_SAMPLE_REDIS_URL"),
            redis_stream=os.getenv("PROB_SAMPLE_REDIS_STREAM", "probability:samples"),
            redis_maxlen=max(0, _env_int("PROB_SAMPLE_REDIS_MAXLEN", 100_000)),
        )
        return cls(config)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _safe_label(self, label: str) -> str:
        return "".join(c for c in str(label) if c.isalnum() or c in ("-", "_", ".")).strip() or "unknown"

    def _ensure_redis(self):
        if self._redis_client is not None or self._redis_failed:
            return self._redis_client
        url = self.config.redis_url
        if not url or redis is None:
            self._redis_failed = True
            return None
        try:
            self._redis_client = redis.Redis.from_url(url, decode_responses=True)
        except Exception as exc:  # pragma: no cover - network failures
            self._redis_failed = True
            logger.warning("ProbabilitySampler unable to connect to Redis (%s): %s", url, exc)
            return None
        return self._redis_client

    def _prepare_records(
        self,
        df: pd.DataFrame,
        prob_series: pd.Series,
        *,
        model_label: str,
        prob_column: str,
        source: str,
        symbol: Optional[str],
        timeframe: Optional[str],
        job_id: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enabled or prob_series is None or df is None:
            return []
        if not isinstance(prob_series, pd.Series):
            prob_series = pd.Series(prob_series, index=df.index)
        numeric = pd.to_numeric(prob_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if numeric.empty:
            return []
        aligned = numeric.reindex(df.index).dropna()
        if aligned.empty:
            return []
        data = pd.DataFrame(index=aligned.index)
        data["probability"] = aligned.astype(float)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df.loc[data.index, "timestamp"], utc=True, errors="coerce")
        else:
            ts = pd.Series(pd.NaT, index=data.index)
        data["timestamp"] = ts
        if "symbol" in df.columns:
            data["symbol"] = df.loc[data.index, "symbol"].astype(str)
        else:
            data["symbol"] = symbol
        if "timeframe" in df.columns:
            data["timeframe"] = df.loc[data.index, "timeframe"].astype(str)
        else:
            data["timeframe"] = timeframe
        if len(data) > self.config.max_rows:
            data = data.iloc[-self.config.max_rows :]
        rows: List[Dict[str, Any]] = []
        for row in data.itertuples():
            ts_val = None
            if isinstance(row.timestamp, pd.Timestamp) and not pd.isna(row.timestamp):
                ts_val = row.timestamp.isoformat()
            payload: Dict[str, Any] = {
                "timestamp": ts_val,
                "probability": float(row.probability),
                "model": model_label,
                "prob_column": prob_column,
                "source": source,
                "symbol": row.symbol if row.symbol is not None else symbol,
                "timeframe": row.timeframe if row.timeframe is not None else timeframe,
                "job_id": job_id,
            }
            if extra:
                payload.update(extra)
            rows.append(payload)
        return rows

    def record(
        self,
        *,
        model_label: str,
        prob_column: str,
        df: pd.DataFrame,
        prob_series: pd.Series,
        source: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        if df is None or prob_series is None:
            return
        records = self._prepare_records(
            df,
            prob_series,
            model_label=model_label,
            prob_column=prob_column,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            job_id=job_id,
            extra=extra,
        )
        if not records:
            return
        if self.config.file_root is not None:
            label = self._safe_label(model_label)
            prob = self._safe_label(prob_column)
            path = self.config.file_root / f"{label}_{prob}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, default=str))
                    handle.write("\n")
        client = self._ensure_redis()
        if client is not None:
            stream = self.config.redis_stream or "probability:samples"
            maxlen = self.config.redis_maxlen or None
            for row in records:
                fields = {k: "" if v is None else v for k, v in row.items()}
                try:
                    client.xadd(stream, fields, maxlen=maxlen, approximate=True if maxlen else False)
                except Exception as exc:  # pragma: no cover - redis failures
                    logger.warning("ProbabilitySampler failed to push to Redis stream %s: %s", stream, exc)
                    break


_SAMPLER: Optional[ProbabilitySampler] = None


def get_probability_sampler() -> ProbabilitySampler:
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = ProbabilitySampler.from_env()
    return _SAMPLER


def set_probability_sampler_for_tests(sampler: Optional[ProbabilitySampler]) -> None:
    global _SAMPLER
    _SAMPLER = sampler


def record_probability_samples(
    *,
    model_label: str,
    prob_column: str,
    df: pd.DataFrame,
    prob_series: pd.Series,
    source: str,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    job_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    sampler = get_probability_sampler()
    if not sampler.enabled:
        return
    try:
        sampler.record(
            model_label=model_label,
            prob_column=prob_column,
            df=df,
            prob_series=prob_series,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            job_id=job_id,
            extra=extra,
        )
    except Exception as exc:
        logger.debug("ProbabilitySampler failed to record sample for %s: %s", model_label, exc)
