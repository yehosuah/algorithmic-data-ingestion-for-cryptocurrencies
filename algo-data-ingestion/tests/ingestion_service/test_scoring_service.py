from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from app.ingestion_service.manifests import (
    ManifestRegistry,
    _set_manifest_registry_for_tests,
    parse_model_specs,
)
from app.ingestion_service.scoring import (
    ModelScoringService,
    _set_scoring_service_for_tests,
    get_scoring_service,
)


BASE_FEATURE_COLUMNS = json.loads(
    Path("models/base_xgb_h120_calmon_spread0/feature_list.json").read_text()
)


@pytest.fixture(name="loaded_registry")
def fixture_loaded_registry():
    registry = ManifestRegistry()
    models_root = Path("models").resolve()
    specs = parse_model_specs(
        "base_xgb_h120_calmon_spread0, tcn_h120_calmon_relaxed, blender_h120_v6"
    )
    registry.preload(models_root=models_root, specs=specs, clear=True)
    _set_manifest_registry_for_tests(registry)
    try:
        yield registry
    finally:
        _set_manifest_registry_for_tests(None)


@pytest.fixture(name="scoring_service")
def fixture_scoring_service(loaded_registry: ManifestRegistry, monkeypatch: pytest.MonkeyPatch):
    # Ensure the scoring singleton is reset for each test.
    _set_scoring_service_for_tests(None)

    # --- Base XGB stubs -------------------------------------------------
    def fake_load_base_predictor(model_dir: Path, prob_column: str = "base_prob"):
        del model_dir, prob_column
        class Dummy:  # pragma: no cover - representation only
            ...
        return Dummy(), ["feature_a"]

    def fake_predict_base(df: pd.DataFrame, calibrator, feat_cols):
        del calibrator, feat_cols
        # Deterministic probabilities irrespective of input
        return pd.Series([0.85, 0.4], index=df.index, name="base_prob")

    monkeypatch.setattr(
        "app.ingestion_service.scoring.load_base_predictor",
        fake_load_base_predictor,
        raising=True,
    )
    monkeypatch.setattr(
        "app.ingestion_service.scoring.predict_base",
        fake_predict_base,
        raising=True,
    )

    # --- TCN stubs -------------------------------------------------------
    def fake_load_tcn_predictor(model_dir: Path, prob_column: str = "tcn_prob"):
        del model_dir, prob_column
        class DummyModel:  # pragma: no cover
            ...
        class DummyCalib:  # pragma: no cover
            def predict_proba(self, arr):
                arr = np.asarray(arr).reshape(-1, 1)
                pos = np.full(len(arr), 1.0)
                return np.column_stack([1 - pos, pos])
        return DummyModel(), DummyCalib(), ["series_a"], None, 3

    def fake_predict_tcn(df: pd.DataFrame, model, calibrator, series_cols, scaler, window, *, stride=1):
        del model, calibrator, series_cols, scaler, window, stride
        ts = pd.to_datetime(df["timestamp"], utc=True)
        probs = pd.Series([0.9, 0.4], index=ts.index[: len(ts)]).iloc[: len(ts)]
        return pd.DataFrame({"timestamp": ts.reset_index(drop=True), "tcn_prob": probs.values})

    monkeypatch.setattr(
        "app.ingestion_service.scoring.load_tcn_predictor",
        fake_load_tcn_predictor,
        raising=True,
    )
    monkeypatch.setattr(
        "app.ingestion_service.scoring.predict_tcn",
        fake_predict_tcn,
        raising=True,
    )

    # --- Blender stubs ---------------------------------------------------
    real_joblib_load = joblib.load

    def fake_joblib_load(path, *args, **kwargs):
        if str(path).endswith("blender.joblib"):
            class DummyBlender:
                def predict_proba(self_inner, X):
                    n = len(X)
                    pos = np.full(n, 0.7, dtype=float)
                    if n > 1:
                        pos[1:] = 0.4
                    neg = 1.0 - pos
                    return np.column_stack([neg, pos])
            return DummyBlender()
        return real_joblib_load(path, *args, **kwargs)

    def fake_build_blender_features(df: pd.DataFrame, *, candidate_cols=None, use_rss_features=True):
        del use_rss_features
        if candidate_cols is None:
            candidate_cols = ["base_prob", "tcn_prob"]
        X = pd.DataFrame(index=df.index)
        for col in candidate_cols:
            X[col] = df.get(col, 0.0)
        return X.astype(float).fillna(0.0), list(candidate_cols)

    monkeypatch.setattr("app.ingestion_service.scoring.joblib.load", fake_joblib_load, raising=True)
    monkeypatch.setattr(
        "app.ingestion_service.scoring.build_blender_features",
        fake_build_blender_features,
        raising=True,
    )

    return get_scoring_service()


def _build_common_frame(model_dir: Path) -> pd.DataFrame:
    del model_dir  # common synthetic frame independent of artifact dir
    timestamps = pd.to_datetime(
        ["2025-10-01T00:00:00Z", "2025-10-01T00:01:00Z"],
        utc=True,
    )
    data = {
        "timestamp": timestamps,
        "hl_spread": [5e-4, 5e-4],
        "hl_spread_z": [-0.5, -0.1],
        "rvol_20": [5e-5, 6e-4],
        "series_a": [0.1, 0.2],
        "base_prob": [0.5, 0.3],
        "tcn_prob": [0.5, 0.3],
        "rss_spike_presence": [1.0, 0.0],
    }
    for col in BASE_FEATURE_COLUMNS:
        data.setdefault(col, [0.0, 0.0])
    return pd.DataFrame(data)


def test_score_batch_base(scoring_service: ModelScoringService, loaded_registry: ManifestRegistry):
    model_dir = loaded_registry.get_path("base_xgb_h120_calmon_spread0")
    frame = _build_common_frame(model_dir)

    payload = scoring_service.score_batch("base_xgb_h120_calmon_spread0", frame, update_metrics=False)

    assert payload["model"] == "base_xgb_h120_calmon_spread0"
    assert "items" in payload and len(payload["items"]) == 2
    gate_results = [item["gate_pass"] for item in payload["items"]]
    # Base manifest inference gate currently enforces prob >= 0.9, so both rows fail.
    assert gate_results == [False, False]
    probabilities = [item["probability"] for item in payload["items"]]
    assert probabilities == [pytest.approx(0.85), pytest.approx(0.4)]


def test_score_batch_tcn(scoring_service: ModelScoringService, loaded_registry: ManifestRegistry):
    model_dir = loaded_registry.get_path("tcn_h120_calmon_relaxed")
    frame = _build_common_frame(model_dir)

    payload = scoring_service.score_batch("tcn_h120_calmon_relaxed", frame, update_metrics=False)

    assert payload["model"] == "tcn_h120_calmon_relaxed"
    gate_results = [item["gate_pass"] for item in payload["items"]]
    # TCN manifest keeps prob >= 0.6, so only the high-prob row survives.
    assert gate_results == [True, False]
    probabilities = [item["probability"] for item in payload["items"]]
    assert probabilities == [pytest.approx(0.9), pytest.approx(0.4)]


def test_score_batch_blender(scoring_service: ModelScoringService, loaded_registry: ManifestRegistry):
    model_dir = loaded_registry.get_path("blender_h120_v6")
    frame = _build_common_frame(model_dir)

    payload = scoring_service.score_batch("blender_h120_v6", frame, update_metrics=False)

    assert payload["model"] == "blender_h120_v6"
    gate_results = [item["gate_pass"] for item in payload["items"]]
    assert gate_results == [True, False]
    probabilities = [item["probability"] for item in payload["items"]]
    assert probabilities[0] > probabilities[1]
    assert 0.0 < probabilities[0] < 1.0
    assert 0.0 <= probabilities[1] < 1.0
