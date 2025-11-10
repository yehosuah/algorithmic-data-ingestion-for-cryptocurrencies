from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from training.calibration_utils import (
    CalibratorResult,
    compute_metrics,
    fit_best_calibrator,
    apply_posthoc_calibration,
)
from training.calibration_store import save_calibrator, load_calibrator


def test_compute_metrics_basic():
    probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    labels = np.array([0, 1, 0, 1], dtype=int)

    metrics = compute_metrics(probs, labels, n_bins=2)

    assert metrics.brier == pytest.approx(0.025)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert len(metrics.reliability) == 2
    assert metrics.reliability[0]["count"] == pytest.approx(2.0)


def test_fit_best_calibrator_prefers_isotonic():
    rng = np.random.default_rng(42)
    train_probs = np.linspace(0.05, 0.95, 2000)
    train_labels = (train_probs + rng.normal(0, 0.05, len(train_probs)) > 0.5).astype(int)
    val_probs = np.linspace(0.05, 0.95, 800)
    val_labels = (val_probs + rng.normal(0, 0.05, len(val_probs)) > 0.55).astype(int)

    result = fit_best_calibrator(
        train_probs,
        train_labels,
        val_probs,
        val_labels,
        methods=("isotonic", "platt"),
        min_train_rows=500,
    )

    assert result.method in {"isotonic", "platt"}
    assert result.metrics_after.brier <= result.metrics_before.brier


def test_calibration_store_roundtrip(tmp_path: Path):
    estimator = IsotonicRegression(out_of_bounds="clip")
    x = np.array([0.1, 0.4, 0.8])
    y = np.array([0, 0, 1])
    estimator.fit(x, y)

    from training.calibration_utils import MetricsBundle

    dummy_metrics = compute_metrics(np.array([0.2, 0.7]), np.array([0, 1]))
    from training.calibration_utils import CalibratorResult

    result = CalibratorResult(
        method="isotonic",
        estimator=estimator,
        metadata={"train_rows": 3, "val_rows": 2, "methods_evaluated": ["isotonic"]},
        metrics_before=dummy_metrics,
        metrics_after=dummy_metrics,
        val_predictions_raw=np.array([0.2, 0.7]),
        val_predictions_cal=np.array([0.2, 0.7]),
    )
    dataset_info = {"source": "synthetic", "train_rows": 3, "val_rows": 2}

    meta_path = save_calibrator(tmp_path, "blender_prob", result=result, dataset_info=dataset_info)
    assert meta_path is not None and meta_path.exists()

    loaded = load_calibrator(tmp_path, "blender_prob")
    assert loaded is not None
    assert loaded.method == "isotonic"
    preds = apply_posthoc_calibration(np.array([0.3, 0.6]), method=loaded.method, estimator=loaded.estimator)
    assert preds.shape == (2,)
