from __future__ import annotations

import os
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional, Tuple

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
    use_ingest = os.getenv("USE_INGEST_METRICS_REGISTRY", "1").strip().lower()
    if use_ingest in {"0", "false", "no"}:
        return REGISTRY
    return _INGEST_REGISTRY or REGISTRY


_REGISTRY: CollectorRegistry = _resolve_registry()


def get_metrics_registry() -> CollectorRegistry:
    """
    Expose the registry used by the inference metrics so external services
    (e.g., scheduler) can serve it via Prometheus endpoints.
    """
    return _REGISTRY


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


_GATE_COVERAGE_ROLLING_WINDOW_SECONDS = max(
    60,
    int(os.getenv("MODEL_GATE_COVERAGE_ROLLING_WINDOW_SECONDS", str(24 * 3600))),
)


class _RollingCoverageBuffer:
    def __init__(self, window_seconds: int) -> None:
        self.window = timedelta(seconds=max(60, window_seconds))
        self.samples: Dict[Tuple[str, str], Deque[Tuple[datetime, int, int]]] = defaultdict(deque)
        self.pass_totals: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def observe(
        self,
        model: str,
        mode: str,
        passed: int,
        total: int,
        timestamp: Optional[datetime],
    ) -> float:
        if total <= 0:
            return 0.0
        ts = timestamp or datetime.now(timezone.utc)
        key = (model, mode)
        dq = self.samples[key]
        dq.append((ts, int(passed), int(total)))
        self.pass_totals[key] += int(passed)
        self.total_counts[key] += int(total)
        cutoff = ts - self.window
        while dq and dq[0][0] < cutoff:
            old_ts, old_pass, old_total = dq.popleft()
            _ = old_ts  # explicit discard for readability
            self.pass_totals[key] -= int(old_pass)
            self.total_counts[key] -= int(old_total)
        denom = max(0, self.total_counts[key])
        if denom <= 0:
            return 0.0
        return float(self.pass_totals[key]) / float(denom)


_ROLLING_COVERAGE = _RollingCoverageBuffer(_GATE_COVERAGE_ROLLING_WINDOW_SECONDS)


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


def record_gate_coverage_sample(
    model: str,
    mode: str,
    passed: int,
    total: int,
    *,
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Track gate coverage using raw counts so rolling windows can be computed.
    """
    ratio = _ROLLING_COVERAGE.observe(model, mode, passed, total, timestamp)
    MODEL_GATE_COVERAGE_RATIO.labels(model=model, mode=f"{mode}_rolling24h").set(float(ratio))


def set_gate_coverage_reference(
    model: str,
    reference: Optional[float],
    tolerance: Optional[float] = None,
) -> None:
    """
    Publish the reference OOS coverage band so Prometheus alerts can compare against live data.
    """
    if reference is None:
        return
    ref = float(reference)
    tol = float(tolerance) if tolerance is not None else 0.0
    lower = max(0.0, ref - tol)
    upper = min(1.0, ref + tol)
    MODEL_GATE_COVERAGE_RATIO.labels(model=model, mode="oos_reference").set(ref)
    MODEL_GATE_COVERAGE_RATIO.labels(model=model, mode="oos_reference_lower").set(lower)
    MODEL_GATE_COVERAGE_RATIO.labels(model=model, mode="oos_reference_upper").set(upper)
