from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import joblib

from app.ingestion_service.manifests import (
    get_manifest_registry,
    prepare_decision_payload,
)
from training.blender import build_blender_features
from training.calibration_store import load_calibrator, LoadedCalibrator
from training.calibration_utils import apply_posthoc_calibration
from training.infer import (
    ManifestArtifacts,
    load_base_predictor,
    load_tcn_predictor,
    predict_base,
    predict_tcn,
)
from app.monitoring.probability_sampler import record_probability_samples

logger = logging.getLogger(__name__)


class UnsupportedModelError(RuntimeError):
    """Raised when a manifest directory does not expose a recognised model artifact."""


def _detect_model_kind(model_dir: Path) -> str:
    if (model_dir / "model.json").exists():
        return "base_xgb"
    if (model_dir / "blender.joblib").exists():
        return "blender"
    if (model_dir / "tcn.pt").exists():
        return "tcn"
    raise UnsupportedModelError(f"Unable to infer model type from artifacts in {model_dir}")


@dataclass
class BaseRunner:
    label: str
    artifacts: ManifestArtifacts

    def score(
        self,
        df: pd.DataFrame,
        *,
        include_features: bool,
        update_metrics: bool,
    ) -> Dict[str, object]:
        raise NotImplementedError


@dataclass
class BaseXGBRunner(BaseRunner):
    label: str
    artifacts: ManifestArtifacts
    calibrator: object
    feature_columns: list[str]

    def score(
        self,
        df: pd.DataFrame,
        *,
        include_features: bool = False,
        update_metrics: bool = True,
    ) -> Dict[str, object]:
        """
        Score the provided batch using the cached calibrator and emit a decision payload.
        """
        if df is None or df.empty:
            raise ValueError("Cannot score an empty batch")

        # Predict probabilities for the feature frame.
        prob_series = predict_base(df, self.calibrator, self.feature_columns)

        frame = df.copy()
        prob_col = self.artifacts.prob_column or "base_prob"
        frame[prob_col] = prob_series
        try:
            record_probability_samples(
                model_label=self.label,
                prob_column=prob_col,
                df=frame,
                prob_series=prob_series,
                source="ingestion_service",
                extra={"runner": "base_xgb"},
            )
        except Exception:
            logger.debug("Probability sampling failed for base runner %s", self.label, exc_info=True)

        # Prepare payload with manifest gate decision embedded.
        payload = prepare_decision_payload(
            self.label,
            frame,
            prob_series=prob_series,
            gate_column="gate_pass",
            include_features=include_features,
            update_metrics=update_metrics,
        )
        return payload


@dataclass
class TCNRunner(BaseRunner):
    label: str
    artifacts: ManifestArtifacts
    model: object
    calibrator: object
    series_columns: List[str]
    scaler: object
    window: int
    stride: int = 1

    def score(
        self,
        df: pd.DataFrame,
        *,
        include_features: bool,
        update_metrics: bool,
    ) -> Dict[str, object]:
        if df is None or df.empty:
            raise ValueError("Cannot score an empty batch")

        prob_col = self.artifacts.prob_column or "tcn_prob"
        stride = max(1, int(self.stride))

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        prob_col = self.artifacts.prob_column or "tcn_prob"
        if prob_col in df_sorted.columns:
            df_sorted = df_sorted.drop(columns=[prob_col])
        missing = [c for c in self.series_columns if c not in df_sorted.columns]
        if missing:
            raise KeyError(f"Missing required TCN feature columns: {missing}")

        prob_df = predict_tcn(
            df_sorted,
            self.model,
            self.calibrator,
            self.series_columns,
            self.scaler,
            self.window,
            stride=stride,
        )
        if prob_df.empty:
            raise ValueError("TCN predictor returned no probabilities; ensure sufficient history is provided")

        merged = pd.merge(df_sorted, prob_df, on="timestamp", how="inner")
        if merged.empty:
            raise ValueError("No overlap between feature rows and TCN probabilities")

        prob_series = merged[prob_col].astype(float)
        try:
            record_probability_samples(
                model_label=self.label,
                prob_column=prob_col,
                df=merged,
                prob_series=prob_series,
                source="ingestion_service",
                extra={"runner": "tcn"},
            )
        except Exception:
            logger.debug("Probability sampling failed for TCN runner %s", self.label, exc_info=True)
        payload = prepare_decision_payload(
            self.label,
            merged,
            prob_series=prob_series,
            gate_column="gate_pass",
            include_features=include_features,
            update_metrics=update_metrics,
        )
        return payload


@dataclass
class BlenderRunner(BaseRunner):
    label: str
    artifacts: ManifestArtifacts
    model: object
    candidate_columns: Sequence[str]
    prob_calibrator: Optional[LoadedCalibrator]

    def score(
        self,
        df: pd.DataFrame,
        *,
        include_features: bool,
        update_metrics: bool,
    ) -> Dict[str, object]:
        if df is None or df.empty:
            raise ValueError("Cannot score an empty batch")

        prob_col = self.artifacts.prob_column or "blender_prob"
        df_proc = df.copy()
        X, _ = build_blender_features(
            df_proc,
            candidate_cols=self.candidate_columns,
            use_rss_features=True,
        )

        if X.empty:
            raise ValueError("Blender feature matrix is empty; verify input columns match training expectations")

        proba = self.model.predict_proba(X.values)
        if proba.shape[1] < 2:
            raise ValueError("Blender model must output binary class probabilities")
        prob_series = pd.Series(proba[:, 1], index=X.index, name=prob_col).astype(float)
        if self.prob_calibrator is not None:
            calibrated = apply_posthoc_calibration(
                prob_series.to_numpy(),
                method=self.prob_calibrator.method,
                estimator=self.prob_calibrator.estimator,
            )
            prob_series = pd.Series(calibrated, index=prob_series.index, name=prob_col)

        scored = df_proc.loc[X.index].copy()
        scored[prob_col] = prob_series

        payload = prepare_decision_payload(
            self.label,
            scored,
            prob_series=prob_series,
            gate_column="gate_pass",
            include_features=include_features,
            update_metrics=update_metrics,
        )
        return payload


class ModelScoringService:
    """
    Lazy-loading scoring façade for deployable models backed by manifest artifacts.
    """

    def __init__(self) -> None:
        self._registry = get_manifest_registry()
        self._lock = Lock()
        self._runners: Dict[str, BaseRunner] = {}

    def _build_runner(self, label: str) -> BaseRunner:
        artifacts = self._registry.ensure_loaded(label)
        model_dir = self._registry.get_path(label)
        model_kind = _detect_model_kind(model_dir)

        if model_kind != "base_xgb":
            if model_kind == "tcn":
                model, calibrator, series_cols, scaler, window = load_tcn_predictor(model_dir)
                metadata = artifacts.manifest.get("metadata") or {}
                stride = int(metadata.get("stride", 1))
                logger.info(
                    "Initialised TCN scorer for '%s' (series_cols=%d, window=%d, stride=%d)",
                    label,
                    len(series_cols),
                    window,
                    stride,
                )
                return TCNRunner(
                    label=label,
                    artifacts=artifacts,
                    model=model,
                    calibrator=calibrator,
                    series_columns=list(series_cols),
                    scaler=scaler,
                    window=int(window),
                    stride=stride,
                )
            if model_kind == "blender":
                feature_path = model_dir / "blender_features.txt"
                if not feature_path.exists():
                    raise FileNotFoundError(f"Blender feature list missing: {feature_path}")
                candidate_columns = [
                    line.strip() for line in feature_path.read_text().splitlines() if line.strip()
                ]
                if not candidate_columns:
                    raise ValueError(f"No feature columns declared in {feature_path}")
                model = joblib.load(model_dir / "blender.joblib")
                prob_col = artifacts.prob_column or "blender_prob"
                prob_calibrator = load_calibrator(model_dir, prob_col)
                logger.info(
                    "Initialised blender scorer for '%s' (feature_cols=%d)",
                    label,
                    len(candidate_columns),
                )
                return BlenderRunner(
                    label=label,
                    artifacts=artifacts,
                    model=model,
                    candidate_columns=candidate_columns,
                    prob_calibrator=prob_calibrator,
                )

        prob_col = artifacts.prob_column or "base_prob"
        calibrator, feature_columns = load_base_predictor(
            model_dir,
            prob_column=prob_col,
            apply_calibration=getattr(artifacts, "apply_calibration", True),
        )
        logger.info("Initialised base XGB scorer for '%s' (features=%d)", label, len(feature_columns))

        return BaseXGBRunner(
            label=label,
            artifacts=artifacts,
            calibrator=calibrator,
            feature_columns=list(feature_columns),
        )

    def _get_runner(self, label: str) -> BaseRunner:
        with self._lock:
            runner = self._runners.get(label)
            if runner is not None:
                return runner
            runner = self._build_runner(label)
            self._runners[label] = runner
            return runner

    def score_batch(
        self,
        label: str,
        df: pd.DataFrame,
        *,
        include_features: bool = False,
        update_metrics: bool = True,
    ) -> Dict[str, object]:
        """
        Compute model probabilities + manifest gate decision for a batch of rows.
        """
        runner = self._get_runner(label)
        return runner.score(
            df,
            include_features=include_features,
            update_metrics=update_metrics,
        )


_SCORING_SERVICE_SINGLETON: Optional[ModelScoringService] = None


def get_scoring_service() -> ModelScoringService:
    global _SCORING_SERVICE_SINGLETON
    if _SCORING_SERVICE_SINGLETON is None:
        _SCORING_SERVICE_SINGLETON = ModelScoringService()
    return _SCORING_SERVICE_SINGLETON


def _set_scoring_service_for_tests(service: Optional[ModelScoringService]) -> None:
    """
    Test helper: override the module-level singleton so fixtures can control lifecycle.
    """
    global _SCORING_SERVICE_SINGLETON
    _SCORING_SERVICE_SINGLETON = service
