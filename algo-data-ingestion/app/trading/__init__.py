from __future__ import annotations

from .config import TradingConfig, TradingModelConfig  # noqa: F401

__all__ = ["TradingConfig", "TradingModelConfig", "TradingService"]


def __getattr__(name: str):
    if name == "TradingService":
        from .service import TradingService

        return TradingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
