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
    policy_id: Optional[str] = None
    order_amount: Optional[float] = Field(default=None, ge=0.0)
    order_notional: Optional[float] = Field(default=None, ge=0.0)
    max_spread_bps: float = Field(default=10.0, gt=0.0)
    min_hold_bars_override: Optional[int] = Field(default=None, ge=1)
    max_hold_minutes: Optional[int] = Field(default=None, ge=1)
    stop_loss_pct: Optional[float] = Field(default=0.005, ge=0.0)
    take_profit_pct: Optional[float] = Field(default=None, ge=0.0)
    profit_trailing_start_pct: Optional[float] = Field(default=None, ge=0.0)
    profit_trailing_stop_pct: Optional[float] = Field(default=None, ge=0.0)
    disable_prob_exits: bool = Field(
        default=False,
        description="When true, disable probability/gate-driven exits; rely on stop/take-profit/trailing/time exits.",
    )
    entry_rsi_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Optional entry filter requiring features.rsi_14 to exceed this threshold.",
    )
    entry_macd_min: Optional[float] = Field(
        default=None,
        description="Optional entry filter requiring features.macd to exceed this threshold.",
    )
    shadow_mode: Optional[bool] = Field(default=None)

    @model_validator(mode="after")
    def ensure_order_sizing(self) -> "TradingModelConfig":
        if self.order_amount is None and self.order_notional is None:
            raise ValueError(
                f"Trading config for model '{self.model}' must provide either order_amount or order_notional"
            )
        if not self.policy_id:
            self.policy_id = self.model
        return self

    @property
    def bar_seconds(self) -> int:
        return timeframe_to_seconds(self.timeframe)

    @property
    def state_key(self) -> str:
        return f"{self.exchange}:{self.model}:{self.symbol}"


class TradingConfig(BaseSettings):
    decision_queue_url: str = Field("redis://localhost:6379/0", alias="DECISION_QUEUE_URL")
    decision_queue_key: str = Field("trading:decisions", alias="DECISION_QUEUE_KEY")
    decision_hmac_secret: Optional[str] = Field(default=None, alias="TRADING_DECISION_HMAC_SECRET")
    require_signed_decisions: Optional[bool] = Field(default=None, alias="TRADING_REQUIRE_SIGNED_DECISIONS")
    last_timestamp_hash: str = Field("trading:last_processed_ts", alias="TRADING_LAST_TS_HASH")
    redis_poll_timeout: int = Field(1, alias="TRADING_QUEUE_POLL_TIMEOUT")
    price_monitor_interval_seconds: int = Field(
        0,
        alias="TRADING_PRICE_MONITOR_INTERVAL_SECONDS",
        ge=0,
        description="Optional interval (seconds) to check price-based exits (stop/take-profit/trailing) even when no new decision arrives; 0 disables.",
    )
    decision_max_age_seconds: Optional[int] = Field(
        None,
        alias="TRADING_DECISION_MAX_AGE_SECONDS",
        description="Optional age cutoff (seconds) for dropping old decision payloads; unset to accept all.",
    )
    dry_run: bool = Field(True, alias="TRADING_DRY_RUN")
    models_root: Path = Field(Path("models"), alias="MODELS_ROOT")
    risk_limits_path: Path = Field(Path("configs/portfolio_risk_limits.yaml"), alias="TRADING_RISK_LIMITS_PATH")
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
    shadow_mode_default: Optional[bool] = Field(default=None, alias="TRADING_SHADOW_MODE_DEFAULT")
    shadow_symbols: Optional[List[str]] = Field(default=None, alias="TRADING_SHADOW_SYMBOLS")
    log_level: str = Field("INFO", alias="TRADING_LOG_LEVEL")
    last_timestamp_grace_bars: int = Field(3, alias="TRADING_LAST_TS_GRACE_BARS", ge=0)
    intent_ledger_backend: Literal["memory", "redis"] = Field("memory", alias="TRADING_INTENT_LEDGER_BACKEND")
    intent_ledger_redis_url: Optional[str] = Field(None, alias="TRADING_INTENT_LEDGER_REDIS_URL")
    intent_ledger_prefix: str = Field("trading:intent", alias="TRADING_INTENT_LEDGER_PREFIX")
    intent_lock_ttl_seconds: int = Field(6 * 3600, alias="TRADING_INTENT_LOCK_TTL_SECONDS", ge=60)
    reconcile_interval_seconds: int = Field(300, alias="TRADING_RECONCILE_INTERVAL_SECONDS", ge=10)
    reconcile_healthy_streak: int = Field(3, alias="TRADING_RECONCILE_HEALTHY_STREAK", ge=1)
    reconcile_dust_notional: float = Field(1.0, alias="TRADING_RECONCILE_DUST_NOTIONAL", ge=0.0)
    safe_mode_allow_exits: bool = Field(True, alias="TRADING_SAFE_MODE_ALLOW_EXITS")
    order_monitor_max_seconds: int = Field(300, alias="TRADING_ORDER_MONITOR_MAX_SECONDS", ge=30)
    order_monitor_backoff_seconds: int = Field(5, alias="TRADING_ORDER_MONITOR_BACKOFF_SECONDS", ge=1)
    deployment_contract_path: Optional[Path] = Field(
        default=Path("configs/deployment_portfolio_contract.yaml"),
        alias="TRADING_DEPLOYMENT_CONTRACT",
    )
    deadlock_policy_path: Optional[Path] = Field(default=None, alias="TRADING_DEADLOCK_POLICY_PATH")
    deadlock_policy_payload: Optional[str] = Field(default=None, alias="TRADING_DEADLOCK_POLICY")

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
        shadow_raw = data.get("shadow_symbols")
        if isinstance(shadow_raw, str):
            data["shadow_symbols"] = [sym.strip() for sym in shadow_raw.split(",") if sym.strip()]
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
        if self.decision_hmac_secret is not None:
            self.decision_hmac_secret = self.decision_hmac_secret.strip() or None
        if not self.dry_run and not self.decision_hmac_secret:
            raise ValueError("TRADING_DECISION_HMAC_SECRET must be provided when TRADING_DRY_RUN=false")
        self._apply_shadow_overrides()
        # Normalize paths
        self.models_root = Path(self.models_root).expanduser().resolve()
        self.risk_limits_path = Path(self.risk_limits_path).expanduser().resolve()
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
        if self.intent_ledger_backend == "redis" and not self.intent_ledger_redis_url:
            self.intent_ledger_redis_url = self.state_redis_url or self.decision_queue_url
        if self.intent_ledger_backend not in {"redis", "memory"}:
            self.intent_ledger_backend = "memory"
        self.intent_ledger_prefix = self.intent_ledger_prefix.rstrip(":")
        self.intent_lock_ttl_seconds = max(60, int(self.intent_lock_ttl_seconds))
        self.order_monitor_max_seconds = max(self.order_monitor_max_seconds, self.order_monitor_backoff_seconds)
        if self.deployment_contract_path is not None:
            self.deployment_contract_path = Path(self.deployment_contract_path).expanduser().resolve()
        if self.deadlock_policy_path is not None:
            self.deadlock_policy_path = Path(self.deadlock_policy_path).expanduser().resolve()
        return self

    def _apply_shadow_overrides(self) -> None:
        """
        Apply per-symbol shadow routing overrides from env vars after configs are loaded.
        """
        default_shadow = None
        if self.shadow_mode_default is not None:
            try:
                default_shadow = bool(self.shadow_mode_default)
            except Exception:
                default_shadow = None
        overrides = set()
        for sym in self.shadow_symbols or []:
            sym_str = str(sym).strip()
            if sym_str:
                overrides.add(sym_str)
        for cfg in self.trading_models:
            shadow_value = cfg.shadow_mode
            if shadow_value is None:
                shadow_value = bool(default_shadow) if default_shadow is not None else False
            if cfg.symbol in overrides:
                shadow_value = True
            cfg.shadow_mode = bool(shadow_value)
