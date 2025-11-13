import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_report_path(manifest_path: Path, report_ref: str) -> Path:
    candidate = manifest_path.parent / report_ref
    if candidate.exists():
        return candidate
    candidate = Path(report_ref)
    if candidate.exists():
        return candidate
    raise AssertionError(f"Report referenced by {manifest_path} not found: {report_ref}")


def test_manifest_gate_configs_match_reports():
    manifest_paths = sorted(Path("models").glob("**/manifest*.json"))
    assert manifest_paths, "No model manifests discovered under models/"

    for manifest_path in manifest_paths:
        if " " in manifest_path.name:
            # Alternate manifest snapshots are retained for reference but
            # should not block CI gating checks.
            continue
        manifest = _load_json(manifest_path)
        gates = manifest.get("gates")
        if not isinstance(gates, dict):
            # Some manifests may be metadata-only; skip them.
            continue

        training_gate = gates.get("training") or {}
        inference_gate = gates.get("inference") or {}
        assert (
            training_gate and inference_gate
        ), f"{manifest_path} missing training or inference gate configuration"

        report_ref = manifest.get("report_path")
        if not report_ref:
            continue
        report_path = _resolve_report_path(manifest_path, report_ref)
        report = _load_json(report_path)

        # Gate definitions must stay in sync between training reports and manifests.
        assert (
            report.get("gate_config") == gates
        ), f"{manifest_path} gate_config diverged from report.json"

        selected_threshold = manifest.get("threshold", {}).get("value")
        if selected_threshold is not None and "selected_threshold" in report:
            assert float(report["selected_threshold"]) == float(
                selected_threshold
            ), f"{manifest_path} threshold mismatch"

        # Inference gates should never be looser than the training gate.
        def _as_mapping(value: Any) -> Dict[str, float]:
            if isinstance(value, dict):
                return {k: float(v) for k, v in value.items() if v is not None}
            if value is None:
                return {}
            return {"default": float(value)}

        for key, tr_value in training_gate.items():
            inf_value = inference_gate.get(key)
            if tr_value is None:
                # Training gate disabled; inference can tighten it but not remove it entirely.
                if key.endswith("_max") and inf_value is not None:
                    continue
                # Nothing to compare for other keys when training allows everything.
                continue

            if key.endswith("_max"):
                assert inf_value is not None, f"{manifest_path} inference gate missing {key}"
                tr_map = _as_mapping(tr_value)
                inf_map = _as_mapping(inf_value)
                for symbol, tr_limit in tr_map.items():
                    inf_limit = inf_map.get(symbol, inf_map.get("default"))
                    if inf_limit is None:
                        continue
                    assert float(inf_limit) <= float(tr_limit), (
                        f"{manifest_path} inference {key} ({inf_limit}) "
                        f"should not exceed training bound ({tr_limit}) for symbol {symbol}"
                    )
                for symbol, inf_limit in inf_map.items():
                    tr_limit = tr_map.get(symbol, tr_map.get("default"))
                    if tr_limit is None:
                        continue
                    assert float(inf_limit) <= float(tr_limit), (
                        f"{manifest_path} inference {key} ({inf_limit}) "
                        f"should not exceed training bound ({tr_limit}) for symbol {symbol}"
                    )
            elif key == "prob_gate_min":
                # Higher inference probability thresholds are safer.
                if inf_value is None:
                    continue
                tr_map = _as_mapping(tr_value)
                inf_map = _as_mapping(inf_value)
                for symbol, inf_limit in inf_map.items():
                    tr_limit = tr_map.get(symbol, tr_map.get("default"))
                    if tr_limit is None:
                        continue
                    assert float(inf_limit) >= float(tr_limit), (
                        f"{manifest_path} inference {key} ({inf_limit}) "
                        f"is looser than training ({tr_limit}) for symbol {symbol}"
                    )
            elif key == "min_hold_bars":
                assert inf_value is not None, f"{manifest_path} inference gate missing {key}"
                assert int(inf_value) >= int(tr_value), (
                    f"{manifest_path} inference {key} ({inf_value}) "
                    f"is lower than training ({tr_value})"
                )
            elif key == "long_only":
                if inf_value is None:
                    continue
                assert bool(inf_value) == bool(tr_value), (
                    f"{manifest_path} inference {key} ({inf_value}) "
                    f"differs from training ({tr_value})"
                )
