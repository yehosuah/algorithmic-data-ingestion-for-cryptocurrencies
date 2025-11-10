from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib

from .calibration_utils import CalibratorResult


CALIBRATION_DIRNAME = "calibration"


@dataclass
class LoadedCalibrator:
    method: str
    target_column: str
    estimator: object
    metadata: Dict[str, object]


def _calibration_dir(model_dir: Path) -> Path:
    path = Path(model_dir) / CALIBRATION_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path(model_dir: Path, target_column: str) -> Path:
    safe = target_column.replace("/", "_")
    return _calibration_dir(model_dir) / f"{safe}.json"


def _estimator_path(model_dir: Path, target_column: str) -> Path:
    safe = target_column.replace("/", "_")
    return _calibration_dir(model_dir) / f"{safe}.joblib"


def save_calibrator(
    model_dir: Path,
    target_column: str,
    *,
    result: CalibratorResult,
    dataset_info: Dict[str, object],
) -> Optional[Path]:
    """
    Persist the fitted calibrator (if any) plus metadata. Returns the metadata path
    or None when calibration is effectively identity/disabled.
    """
    meta_path = _metadata_path(model_dir, target_column)
    est_path = _estimator_path(model_dir, target_column)

    if result.method in {"identity", "none", ""} or result.estimator is None:
        if meta_path.exists():
            meta_path.unlink()
        if est_path.exists():
            est_path.unlink()
        return None

    joblib.dump(result.estimator, est_path)
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "method": result.method,
        "target_column": target_column,
        "estimator_path": str(est_path.relative_to(model_dir)),
        "dataset": dataset_info,
        "metrics": {
            "before": result.metrics_before.to_dict(),
            "after": result.metrics_after.to_dict(),
        },
        "metadata": result.metadata,
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return meta_path


def load_calibrator(model_dir: Path, target_column: str) -> Optional[LoadedCalibrator]:
    meta_path = _metadata_path(model_dir, target_column)
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    method = meta.get("method")
    rel_path = meta.get("estimator_path")
    if not method or not rel_path:
        return None
    est_path = Path(model_dir) / rel_path
    if not est_path.exists():
        return None
    estimator = joblib.load(est_path)
    return LoadedCalibrator(
        method=str(method),
        target_column=str(target_column),
        estimator=estimator,
        metadata=meta,
    )
