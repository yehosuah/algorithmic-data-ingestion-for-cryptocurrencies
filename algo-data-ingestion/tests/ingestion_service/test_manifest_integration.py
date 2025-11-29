from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ingestion_service.manifests import (
    ManifestRegistry,
    _set_manifest_registry_for_tests,
    parse_model_specs,
    prepare_decision_payload,
)


def _make_test_frame() -> pd.DataFrame:
    """Construct a minimal frame that exercises manifest gate predicates."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-10-01T00:00:00Z", "2025-10-01T00:01:00Z"],
                utc=True,
            ),
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "hl_spread": [5e-4, 5e-4],
            "hl_spread_z": [-0.5, 0.0],
            "rvol_20": [5e-5, 5e-5],
            "sym_spread_ratio": [0.5, 0.5],
            "sym_rvol_ratio": [0.5, 0.5],
            "sym_liquidity_rank": [1.0, 1.0],
            "base_prob": [0.9, 0.1],
        }
    )


def test_parse_model_specs_basic():
    spec = "alpha=models/base_alpha, beta=models/base_beta"
    parsed = parse_model_specs(spec)
    assert parsed == [
        ("alpha", "models/base_alpha"),
        ("beta", "models/base_beta"),
    ]


@pytest.fixture(name="loaded_registry")
def fixture_loaded_registry():
    registry = ManifestRegistry()
    models_root = Path("models").resolve()
    specs = parse_model_specs("base_xgb_h120_calmon_spread0")
    registry.preload(models_root=models_root, specs=specs, clear=True)
    _set_manifest_registry_for_tests(registry)
    try:
        yield registry
    finally:
        _set_manifest_registry_for_tests(None)


def test_manifest_registry_applies_gate_and_builds_payload(loaded_registry: ManifestRegistry):
    df = _make_test_frame()
    annotated = loaded_registry.annotate_with_gate_pass(
        "base_xgb_h120_calmon_spread0",
        df,
        inplace=False,
        update_metrics=False,
    )
    assert annotated["gate_pass"].tolist() == [True, False]

    payload = loaded_registry.build_decision_payload("base_xgb_h120_calmon_spread0", annotated)
    assert payload["model"] == "base_xgb_h120_calmon_spread0"
    assert [item["gate_pass"] for item in payload["items"]] == [True, False]
    assert payload["items"][0]["probability"] == pytest.approx(0.9)


def test_prepare_decision_payload_pipeline(loaded_registry: ManifestRegistry):
    df = _make_test_frame()
    payload = prepare_decision_payload(
        "base_xgb_h120_calmon_spread0",
        df,
        update_metrics=False,
    )
    assert payload["prob_column"] == "base_prob"
    gate_vector = [item["gate_pass"] for item in payload["items"]]
    assert gate_vector == [True, False]


def test_build_decision_payload_includes_price_fields(loaded_registry: ManifestRegistry):
    df = _make_test_frame()
    df["close"] = [101.0, 102.0]
    annotated = loaded_registry.annotate_with_gate_pass(
        "base_xgb_h120_calmon_spread0",
        df,
        inplace=False,
        update_metrics=False,
    )
    payload = loaded_registry.build_decision_payload("base_xgb_h120_calmon_spread0", annotated)
    first = payload["items"][0]
    assert first["close"] == pytest.approx(101.0)
    assert first.get("price", first["close"]) == pytest.approx(101.0)
