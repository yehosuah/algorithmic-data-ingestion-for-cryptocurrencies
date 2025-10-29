from __future__ import annotations

from typing import Optional

from prometheus_client import REGISTRY, CollectorRegistry, Gauge

try:
    from app.ingestion_service.utils import _METRICS_REGISTRY as _INGEST_REGISTRY  # type: ignore
except Exception:  # pragma: no cover - fallback when ingestion service not loaded
    _INGEST_REGISTRY = None  # type: ignore


def _resolve_registry() -> CollectorRegistry:
    """
    Prefer the ingestion service registry so metrics surface on /metrics.
    Fallback to the global registry when the service layer is not imported
    (e.g. offline training scripts or unit tests).
    """
    return _INGEST_REGISTRY or REGISTRY


_REGISTRY: CollectorRegistry = _resolve_registry()


def _gauge(name: str, documentation: str, *, labelnames: tuple[str, ...]) -> Gauge:
    """
    Helper to construct gauges bound to the resolved registry.
    """
    return Gauge(
        name,
        documentation,
        labelnames=labelnames,
        registry=_REGISTRY,
    )


MODEL_GATE_COVERAGE_RATIO: Gauge = _gauge(
    "model_gate_coverage_ratio",
    "Fraction of rows in the latest batch that passed the manifest gate.",
    labelnames=("model", "mode"),
)

MODEL_RSS_MINUTE_SPIKE_SHARE: Gauge = _gauge(
    "model_rss_minute_spike_share",
    "Share of rows with RSS minute spike indicator > 0 in the latest batch.",
    labelnames=("model",),
)

MODEL_RSS_MINUTE_SPIKE_THRESHOLD: Gauge = _gauge(
    "model_rss_minute_spike_threshold",
    "Configured minimum RSS minute spike share for the model before triggering fallbacks.",
    labelnames=("model",),
)

MODEL_PROBABILITY_SIGMA: Gauge = _gauge(
    "model_probability_sigma",
    "Minimum monthly standard deviation of inference probabilities across the latest batch.",
    labelnames=("model",),
)

MODEL_PROBABILITY_SIGMA_THRESHOLD: Gauge = _gauge(
    "model_probability_sigma_threshold",
    "Configured probability sigma guardrail for the model.",
    labelnames=("model",),
)


def set_rss_threshold(model: str, value: Optional[float]) -> None:
    """
    Publish the configured RSS coverage floor for the given model.
    """
    if value is None:
        return
    MODEL_RSS_MINUTE_SPIKE_THRESHOLD.labels(model=model).set(float(value))


def set_probability_sigma_threshold(model: str, value: Optional[float]) -> None:
    """
    Publish the configured probability sigma guardrail for the given model.
    """
    if value is None:
        return
    MODEL_PROBABILITY_SIGMA_THRESHOLD.labels(model=model).set(float(value))


def observe_gate_coverage(model: str, mode: str, coverage: float) -> None:
    """
    Record gate coverage (fraction of rows passing the manifest predicates) for the batch.
    """
    MODEL_GATE_COVERAGE_RATIO.labels(model=model, mode=mode).set(float(coverage))


def observe_rss_share(model: str, share: Optional[float]) -> None:
    """
    Record the observed RSS minute spike share for the batch.
    """
    if share is None:
        return
    MODEL_RSS_MINUTE_SPIKE_SHARE.labels(model=model).set(float(share))


def observe_probability_sigma(model: str, sigma: Optional[float]) -> None:
    """
    Record the minimum monthly probability sigma observed in the batch.
    """
    if sigma is None:
        return
    MODEL_PROBABILITY_SIGMA.labels(model=model).set(float(sigma))
