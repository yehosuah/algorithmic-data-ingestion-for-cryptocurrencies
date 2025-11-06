from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert a CCXT timeframe string (e.g. "1m", "5m", "1h") to seconds.
    """
    tf = (timeframe or "").strip().lower()
    if not tf:
        raise ValueError("Timeframe must be a non-empty string")
    unit = tf[-1]
    try:
        value = float(tf[:-1] or 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid timeframe value: {timeframe}") from exc
    if unit == "s":
        return int(value)
    if unit == "m":
        return int(value * 60)
    if unit == "h":
        return int(value * 3600)
    if unit == "d":
        return int(value * 86400)
    raise ValueError(f"Unsupported timeframe unit: {timeframe}")


class TradingModelConfig(BaseModel):
    model: str
    symbol: str
    exchange: str = "binance"
    timeframe: str = "1m"
    order_amount: Optional[float] = Field(default=None, ge=0.0)
    order_notional: Optional[float] = Field(default=None, ge=0.0)
    max_spread_bps: float = Field(default=10.0, gt=0.0)
    min_hold_bars_override: Optional[int] = Field(default=None, ge=1)
    max_hold_minutes: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def ensure_order_sizing(self) -> "TradingModelConfig":
        if self.order_amount is None and self.order_notional is None:
            raise ValueError(
                f"Trading config for model '{self.model}' must provide either order_amount or order_notional"
            )
        return self

    @property
    def bar_seconds(self) -> int:
        return timeframe_to_seconds(self.timeframe)

    @property
    def state_key(self) -> str:
        return f"{self.exchange}:{self.symbol}"


class TradingConfig(BaseSettings):
    decision_queue_url: str = Field("redis://localhost:6379/0", alias="DECISION_QUEUE_URL")
    decision_queue_key: str = Field("trading:decisions", alias="DECISION_QUEUE_KEY")
    redis_poll_timeout: int = Field(5, alias="TRADING_QUEUE_POLL_TIMEOUT")
    dry_run: bool = Field(True, alias="TRADING_DRY_RUN")
    models_root: Path = Field(Path("models"), alias="MODELS_ROOT")
    state_path: Path = Field(Path("data_lake/trading/state.json"), alias="TRADING_STATE_PATH")
    state_backend: Literal["file", "redis", "postgres"] = Field("file", alias="TRADING_STATE_BACKEND")
    state_redis_url: Optional[str] = Field(None, alias="TRADING_STATE_REDIS_URL")
    state_redis_hash: str = Field("trading:positions", alias="TRADING_STATE_REDIS_HASH")
    state_postgres_dsn: Optional[str] = Field(None, alias="TRADING_STATE_PG_DSN")
    state_postgres_table: str = Field("trading_positions", alias="TRADING_STATE_PG_TABLE")
    audit_backend: Optional[Literal["file", "redis", "postgres"]] = Field(None, alias="TRADING_AUDIT_BACKEND")
    audit_redis_url: Optional[str] = Field(None, alias="TRADING_AUDIT_REDIS_URL")
    audit_redis_stream: str = Field("trading:audit", alias="TRADING_AUDIT_STREAM")
    audit_maxlen: int = Field(10000, alias="TRADING_AUDIT_MAXLEN")
    state_stale_minutes: Optional[int] = Field(120, alias="TRADING_STATE_STALE_MINUTES", ge=1)
    audit_postgres_dsn: Optional[str] = Field(None, alias="TRADING_AUDIT_PG_DSN")
    audit_postgres_table: str = Field("trading_audit_events", alias="TRADING_AUDIT_TABLE")
    audit_log_path: Path = Field(Path("data_lake/trading/audit.log"), alias="TRADING_AUDIT_LOG_PATH")
    metrics_port: int = Field(9010, alias="TRADING_METRICS_PORT")
    trading_models: List[TradingModelConfig] = Field(default_factory=list, alias="TRADING_MODELS")
    log_level: str = Field("INFO", alias="TRADING_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_trading_models(cls, data: Dict[str, object]) -> Dict[str, object]:
        raw = data.get("trading_models")
        if isinstance(raw, str):
            try:
                data["trading_models"] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid TRADING_MODELS payload: {raw}") from exc
        return data

    @model_validator(mode="after")
    def _default_models_when_empty(self) -> "TradingConfig":
        if not self.trading_models:
            # Default to the deployable TCN manifest, assuming ETH/USDT minute bars.
            self.trading_models = [
                TradingModelConfig(
                    model="tcn_h120_calmon_relaxed",
                    symbol="ETH/USDT",
                    exchange="binance",
                    timeframe="1m",
                    order_notional=100.0,
                    max_spread_bps=10.0,
                )
            ]
        # Normalize paths
        self.models_root = Path(self.models_root).expanduser().resolve()
        self.state_path = Path(self.state_path).expanduser().resolve()
        if self.state_backend == "file":
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_backend == "redis" and not self.state_redis_url:
            self.state_redis_url = self.decision_queue_url
        if self.state_backend == "postgres" and not self.state_postgres_dsn:
            raise ValueError("TRADING_STATE_PG_DSN must be provided when TRADING_STATE_BACKEND=postgres")
        if self.audit_backend is None:
            self.audit_backend = self.state_backend
        if self.audit_backend == "redis" and not self.audit_redis_url:
            self.audit_redis_url = self.state_redis_url or self.decision_queue_url
        if self.audit_backend == "postgres" and not self.audit_postgres_dsn:
            self.audit_postgres_dsn = self.state_postgres_dsn
        self.audit_log_path = Path(self.audit_log_path).expanduser().resolve()
        if self.audit_backend == "file":
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        return self
