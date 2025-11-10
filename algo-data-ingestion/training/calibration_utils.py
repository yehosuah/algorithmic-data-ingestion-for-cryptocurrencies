from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ReliabilityCurve = Sequence[Dict[str, float]]
HistogramBins = Sequence[Dict[str, float]]


@dataclass
class MetricsBundle:
    brier: float
    ece: float
    roc_auc: float
    pr_auc: float
    reliability: ReliabilityCurve
    histogram: HistogramBins

    def to_dict(self) -> Dict[str, float | Sequence[Dict[str, float]]]:
        payload: Dict[str, float | Sequence[Dict[str, float]]] = {
            "brier": float(self.brier),
            "ece": float(self.ece),
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "reliability": list(self.reliability),
            "histogram": list(self.histogram),
        }
        return payload


@dataclass
class CalibratorResult:
    method: str
    estimator: object
    metadata: Dict[str, object]
    metrics_before: MetricsBundle
    metrics_after: MetricsBundle
    val_predictions_raw: np.ndarray
    val_predictions_cal: np.ndarray


class PowerCalibrator:
    def __init__(self, gamma: float) -> None:
        self.gamma = float(max(gamma, 1e-3))

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return _power_transform(probs, self.gamma)


class IsotonicBlendCalibrator:
    def __init__(self, iso: IsotonicRegression, weight: float) -> None:
        self.iso = iso
        self.weight = float(min(max(weight, 0.0), 1.0))

    def transform(self, probs: np.ndarray) -> np.ndarray:
        iso_vals = np.asarray(self.iso.predict(probs), dtype=float)
        return self.weight * iso_vals + (1.0 - self.weight) * probs


def _to_numpy(series: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(series, pd.Series):
        return series.to_numpy(dtype=float)
    return np.asarray(series, dtype=float)


def _safe_probs(probs: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    return np.clip(probs, eps, 1.0 - eps)


def _logit_transform(probs: np.ndarray) -> np.ndarray:
    safe = _safe_probs(probs)
    return np.log(safe / (1.0 - safe))


def _power_transform(probs: np.ndarray, gamma: float) -> np.ndarray:
    safe = _safe_probs(probs)
    g = max(float(gamma), 1e-3)
    pos = np.power(safe, g)
    neg = np.power(1.0 - safe, g)
    denom = pos + neg
    denom[denom == 0.0] = 1.0
    return pos / denom


def _compute_reliability(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 20,
) -> Tuple[ReliabilityCurve, HistogramBins, float]:
    probs = _safe_probs(probs)
    labels = labels.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    total = float(len(probs))
    reliability: list[Dict[str, float]] = []
    hist: list[Dict[str, float]] = []
    ece = 0.0

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = float(np.sum(mask))
        if count == 0:
            rel = {
                "bin_lower": float(bins[idx]),
                "bin_upper": float(bins[idx + 1]),
                "mean_pred": float("nan"),
                "empirical_rate": float("nan"),
                "count": 0.0,
            }
            reliability.append(rel)
            hist.append(
                {
                    "bin_lower": float(bins[idx]),
                    "bin_upper": float(bins[idx + 1]),
                    "fraction": 0.0,
                    "count": 0.0,
                }
            )
            continue

        preds_bin = probs[mask]
        labels_bin = labels[mask]
        mean_pred = float(np.mean(preds_bin))
        empirical = float(np.mean(labels_bin)) if len(labels_bin) else float("nan")
        reliability.append(
            {
                "bin_lower": float(bins[idx]),
                "bin_upper": float(bins[idx + 1]),
                "mean_pred": mean_pred,
                "empirical_rate": empirical,
                "count": count,
            }
        )
        hist.append(
            {
                "bin_lower": float(bins[idx]),
                "bin_upper": float(bins[idx + 1]),
                "fraction": float(count / max(total, 1.0)),
                "count": count,
            }
        )
        if np.isfinite(empirical) and np.isfinite(mean_pred):
            ece += abs(empirical - mean_pred) * (count / max(total, 1.0))

    return reliability, hist, float(ece)


def compute_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 20,
) -> MetricsBundle:
    y = _to_numpy(np.asarray(labels, dtype=float))
    p = _safe_probs(_to_numpy(np.asarray(probs, dtype=float)))

    brier = float(brier_score_loss(y, p))
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = float(average_precision_score(y, p))
    except ValueError:
        pr_auc = float("nan")
    reliability, hist, ece = _compute_reliability(p, y, n_bins=n_bins)
    return MetricsBundle(
        brier=brier,
        ece=float(ece),
        roc_auc=roc,
        pr_auc=pr_auc,
        reliability=reliability,
        histogram=hist,
    )


def _fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(probs, labels)
    return iso


def _fit_isotonic_blend(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    max_folds: int = 5,
) -> IsotonicBlendCalibrator:
    n = len(probs)
    folds = max(2, min(max_folds, n // 2000 if n >= 4000 else 3))
    candidate_weights = np.linspace(0.0, 0.6, 7)
    scores = {float(w): 0.0 for w in candidate_weights}
    counts = {float(w): 0 for w in candidate_weights}
    indices = np.arange(n)
    for fold in range(folds):
        mask = indices % folds == fold
        train_mask = ~mask
        if not train_mask.any() or not mask.any():
            continue
        iso = _fit_isotonic(probs[train_mask], labels[train_mask])
        iso_val = iso.predict(probs[mask])
        base_val = probs[mask]
        for weight in candidate_weights:
            blended = weight * iso_val + (1.0 - weight) * base_val
            metric = compute_metrics(blended, labels[mask])
            scores[float(weight)] += metric.brier
            counts[float(weight)] += 1
    best_weight = 0.0
    best_score = float("inf")
    for weight, total in scores.items():
        if counts[weight] == 0:
            continue
        avg = total / counts[weight]
        if avg < best_score:
            best_score = avg
            best_weight = weight
    iso_full = _fit_isotonic(probs, labels)
    return IsotonicBlendCalibrator(iso_full, best_weight)


def _fit_platt(probs: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    logit = _logit_transform(probs).reshape(-1, 1)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
    )
    clf.fit(logit, labels)
    return clf


def _fit_power_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    gamma_grid: Optional[Sequence[float]] = None,
) -> PowerCalibrator:
    if gamma_grid is None:
        gamma_grid = tuple(np.linspace(0.6, 2.4, 36))
    best_gamma = 1.0
    best_score = compute_metrics(_power_transform(probs, best_gamma), labels).brier
    for gamma in gamma_grid:
        preds = _power_transform(probs, gamma)
        score = compute_metrics(preds, labels).brier
        if score < best_score - 1e-6:
            best_score = score
            best_gamma = float(gamma)
    refined = np.linspace(max(0.5, best_gamma - 0.3), min(3.0, best_gamma + 0.3), 20)
    for gamma in refined:
        preds = _power_transform(probs, gamma)
        score = compute_metrics(preds, labels).brier
        if score < best_score - 1e-6:
            best_score = score
            best_gamma = float(gamma)
    return PowerCalibrator(best_gamma)


def apply_posthoc_calibration(
    probs: np.ndarray,
    *,
    method: str,
    estimator: object | None,
) -> np.ndarray:
    probs = _safe_probs(np.asarray(probs, dtype=float))
    method = (method or "").lower()
    if method in {"identity", "none", ""} or estimator is None:
        return probs
    if method == "isotonic":
        return np.asarray(estimator.predict(probs), dtype=float)
    if method == "platt":
        logit = _logit_transform(probs).reshape(-1, 1)
        return estimator.predict_proba(logit)[:, 1]
    if method == "power":
        if not isinstance(estimator, PowerCalibrator):
            raise ValueError("Power calibration requires PowerCalibrator estimator.")
        return estimator.transform(probs)
    if method == "isotonic_blend":
        if not isinstance(estimator, IsotonicBlendCalibrator):
            raise ValueError("Isotonic blend calibration requires IsotonicBlendCalibrator estimator.")
        return estimator.transform(probs)
    raise ValueError(f"Unsupported calibration method '{method}'")


def fit_best_calibrator(
    train_probs: Sequence[float],
    train_labels: Sequence[int],
    val_probs: Sequence[float],
    val_labels: Sequence[int],
    *,
    methods: Sequence[str] = ("identity", "isotonic", "platt", "power", "isotonic_blend"),
    min_train_rows: int = 1000,
    n_bins: int = 20,
) -> CalibratorResult:
    train_p = _safe_probs(_to_numpy(train_probs))
    train_y = _to_numpy(train_labels).astype(int)
    val_p = _safe_probs(_to_numpy(val_probs))
    val_y = _to_numpy(val_labels).astype(int)

    base_metrics = compute_metrics(val_p, val_y, n_bins=n_bins)
    best_method = "identity"
    best_estimator: object | None = None
    best_metrics = base_metrics
    calibrated_vals = val_p.copy()

    for method in methods:
        method = method.lower().strip()
        if method in {"identity", "none"}:
            continue
        if len(train_p) < min_train_rows:
            continue
        try:
            if method == "isotonic":
                estimator = _fit_isotonic(train_p, train_y)
            elif method == "platt":
                estimator = _fit_platt(train_p, train_y)
            elif method == "power":
                estimator = _fit_power_calibrator(train_p, train_y)
            elif method == "isotonic_blend":
                estimator = _fit_isotonic_blend(train_p, train_y)
            else:
                continue
            val_cal = apply_posthoc_calibration(val_p, method=method, estimator=estimator)
            metrics = compute_metrics(val_cal, val_y, n_bins=n_bins)
        except Exception:
            continue

        if metrics.brier < best_metrics.brier:
            best_method = method
            best_estimator = estimator
            best_metrics = metrics
            calibrated_vals = val_cal

    extra_meta: Dict[str, float] = {}
    if isinstance(best_estimator, PowerCalibrator):
        extra_meta["power_gamma"] = float(best_estimator.gamma)
    if isinstance(best_estimator, IsotonicBlendCalibrator):
        extra_meta["isotonic_blend_weight"] = float(best_estimator.weight)
    metadata = {
        "train_rows": int(len(train_p)),
        "val_rows": int(len(val_p)),
        "methods_evaluated": list(methods),
        **extra_meta,
    }

    return CalibratorResult(
        method=best_method,
        estimator=best_estimator,
        metadata=metadata,
        metrics_before=base_metrics,
        metrics_after=best_metrics,
        val_predictions_raw=val_p,
        val_predictions_cal=calibrated_vals,
    )
