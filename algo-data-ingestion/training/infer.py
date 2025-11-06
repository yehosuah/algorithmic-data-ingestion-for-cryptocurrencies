from __future__ import annotations
import errno
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import copy
import shutil
import tempfile
import time
import joblib
import numpy as np
import pandas as pd
import torch

from .tcn_model import TinyTCN

try:
    from app.monitoring.model_metrics import (
        observe_gate_coverage,
        observe_probability_sigma,
        observe_rss_share,
        set_probability_sigma_threshold,
        set_rss_threshold,
    )
except Exception:  # pragma: no cover - allow training-only environments
    def observe_gate_coverage(*args, **kwargs):  # type: ignore[func-returns-value]
        return None

    def observe_probability_sigma(*args, **kwargs):  # type: ignore[func-returns-value]
        return None

    def observe_rss_share(*args, **kwargs):  # type: ignore[func-returns-value]
        return None

    def set_probability_sigma_threshold(*args, **kwargs):  # type: ignore[func-returns-value]
        return None

    def set_rss_threshold(*args, **kwargs):  # type: ignore[func-returns-value]
        return None


logger = logging.getLogger(__name__)


DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "spread_column": "hl_spread",
    "prob_column": "base_prob",
    "training": {
        "hl_spread_max": None,
        "hl_spread_z_max": 0.25,
        "rvol20_max": 2e-4,
        "prob_gate_min": None,
        "min_hold_bars": 10,
        "long_only": True,
    },
    "inference": {
        "hl_spread_max": 7e-4,
        "hl_spread_z_max": -0.25,
        "rvol20_max": 8e-5,
        "prob_gate_min": 0.72,
        "min_hold_bars": 10,
        "long_only": True,
    },
}


@dataclass
class ManifestArtifacts:
    base_dir: Path
    manifest: Dict[str, Any]
    gate_config: Dict[str, Any]
    report: Dict[str, Any]
    model_label: str
    prob_column: str
    rss_indicator_column: Optional[str]
    rss_minute_share_threshold: Optional[float]
    prob_sigma_threshold: Optional[float]


def _read_text_retry(path: Path, attempts: int = 5, delay: float = 0.1) -> str:
    last_exc: Optional[OSError] = None
    for idx in range(max(1, attempts)):
        try:
            return path.read_text()
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EDEADLK and idx + 1 < attempts:
                time.sleep(delay * (idx + 1))
                continue
            last_exc = exc
            break
    if last_exc is not None and getattr(last_exc, "errno", None) == errno.EDEADLK:
        # Fallback: copy to a temp file on the container filesystem before reading.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copyfile(path, tmp_path)
            return tmp_path.read_text()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    if last_exc is not None:
        raise last_exc
    # Should not reach here, but keep mypy happy.
    return path.read_text()


def _load_manifest_payload(base_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(_read_text_retry(manifest_path))
        if isinstance(payload, dict):
            return payload
        logger.warning("Manifest at %s is not a JSON object", manifest_path)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - diagnostics only
        logger.warning("Failed to parse manifest at %s: %s", manifest_path, exc)
    return None


def _resolve_report_path(base_dir: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    report_ref = manifest.get("report_path")
    if isinstance(report_ref, str) and report_ref:
        candidates.append((base_dir / report_ref).resolve())
        candidates.append(Path(report_ref).resolve())
    candidates.append((base_dir / "report.json").resolve())
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_report_payload(base_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    report_path = _resolve_report_path(base_dir, manifest)
    if not report_path:
        return {}
    try:
        payload = json.loads(_read_text_retry(report_path))
        if isinstance(payload, dict):
            return payload
        logger.warning("Report at %s is not a JSON object", report_path)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - diagnostics only
        logger.warning("Failed to parse report at %s: %s", report_path, exc)
    return {}


def load_manifest_artifacts(base_dir: Path, *, model_label: Optional[str] = None) -> ManifestArtifacts:
    """
    Load manifest + report metadata and normalise the gate configuration for downstream inference.
    """
    manifest = _load_manifest_payload(base_dir) or {}
    gate_config = copy.deepcopy(DEFAULT_GATE_CONFIG)
    gates_raw = manifest.get("gates")
    if isinstance(gates_raw, dict):
        for key, value in gates_raw.items():
            if key in ("training", "inference") and isinstance(value, dict):
                gate_config[key].update(value)
            else:
                gate_config[key] = value
    # Defensive copies for nested dicts
    for section in ("training", "inference"):
        gate_config[section] = copy.deepcopy(gate_config.get(section, {}))
    prob_column = str(gate_config.get("prob_column") or DEFAULT_GATE_CONFIG.get("prob_column", "base_prob"))
    gate_config["prob_column"] = prob_column

    report = _load_report_payload(base_dir, manifest)
    metadata = manifest.get("metadata") or {}
    rss_audit = report.get("rss_audit") or {}
    rss_indicator_column = rss_audit.get("minute_indicator_column")
    rss_threshold = rss_audit.get("min_minute_spike_share")
    if rss_threshold is None:
        rss_threshold = rss_audit.get("minute_spike_share")
    prob_guardrail = report.get("prob_sigma_guardrail") or metadata.get("prob_sigma_guardrail") or {}
    prob_sigma_threshold = prob_guardrail.get("threshold")

    label = model_label or manifest.get("model_label") or manifest.get("model_name") or manifest.get("model_id") or base_dir.name

    return ManifestArtifacts(
        base_dir=base_dir,
        manifest=manifest,
        gate_config=gate_config,
        report=report,
        model_label=str(label),
        prob_column=prob_column,
        rss_indicator_column=rss_indicator_column,
        rss_minute_share_threshold=rss_threshold,
        prob_sigma_threshold=prob_sigma_threshold,
    )


def _register_metric_thresholds(artifacts: ManifestArtifacts) -> None:
    set_rss_threshold(artifacts.model_label, artifacts.rss_minute_share_threshold)
    set_probability_sigma_threshold(artifacts.model_label, artifacts.prob_sigma_threshold)


def _compute_min_monthly_sigma(df: pd.DataFrame, prob_series: pd.Series) -> Optional[float]:
    if prob_series is None or prob_series.empty:
        return None
    numeric = pd.to_numeric(prob_series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    sigma_global = float(numeric.std(ddof=0))
    if "timestamp" not in df.columns:
        return sigma_global
    try:
        ts = pd.to_datetime(df.loc[numeric.index, "timestamp"], utc=True, errors="coerce")
    except Exception:
        return sigma_global
    valid = ~ts.isna()
    if not valid.any():
        return sigma_global
    grouped = numeric.loc[valid].groupby(ts.loc[valid].dt.to_period("M")).std(ddof=0).dropna()
    if grouped.empty:
        return sigma_global
    return float(grouped.min())


def _update_inference_metrics(
    df: pd.DataFrame,
    prob_series: pd.Series,
    gate_mask: pd.Series,
    artifacts: ManifestArtifacts,
    *,
    mode: str,
) -> None:
    try:
        coverage = float(gate_mask.astype(bool).mean()) if len(gate_mask) else 0.0
    except Exception:
        coverage = 0.0
    observe_gate_coverage(artifacts.model_label, mode, coverage)

    rss_share: Optional[float] = None
    indicator_col = artifacts.rss_indicator_column
    if indicator_col and indicator_col in df.columns:
        try:
            indicator = pd.Series(df[indicator_col], index=df.index)
            rss_share = float((indicator.fillna(0.0) > 0).mean())
        except Exception:
            rss_share = None
    observe_rss_share(artifacts.model_label, rss_share)

    sigma = _compute_min_monthly_sigma(df, prob_series)
    observe_probability_sigma(artifacts.model_label, sigma)

def load_base_predictor(base_dir: Path):
    feat_cols = json.loads(_read_text_retry(base_dir / "feature_list.json"))
    calib_path = base_dir / "calibrator.joblib"
    if calib_path.exists():
        calib = joblib.load(calib_path)
    else:
        # Load raw booster model if calibrator absent
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(base_dir / "model.json"))
        calib = xgb.XGBClassifier()
        calib._Booster = booster
        calib._le = None
        calib.n_features_in_ = len(feat_cols)
        calib.classes_ = np.array([0, 1])
    return calib, feat_cols


def load_gate_config(base_dir: Path) -> Dict[str, Any]:
    """
    Load gate configuration stored in a manifest; fall back to defaults when absent.
    """
    artifacts = load_manifest_artifacts(base_dir)
    return copy.deepcopy(artifacts.gate_config)


def predict_base(df: pd.DataFrame, calib, feat_cols: List[str]) -> pd.Series:
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c].astype(float) if c in df.columns else 0.0
    p = calib.predict_proba(X.values)[:, 1]
    return pd.Series(p, index=df.index, name="base_prob")


def compute_gate_mask(
    df: pd.DataFrame,
    gate_config: Optional[Dict[str, Any]] = None,
    *,
    prob: Optional[pd.Series] = None,
    mode: str = "inference",
) -> pd.Series:
    """
    Compute a boolean mask indicating which rows satisfy the configured trade gate.

    Parameters
    ----------
    df:
        Feature frame containing at least the spread/volatility fields referenced by the gate.
    gate_config:
        Dictionary shaped like DEFAULT_GATE_CONFIG; when omitted, defaults are used.
    prob:
        Optional probability series to evaluate the probability gate; defaults to df[prob_column].
    mode:
        Either "inference" or "training" to pick the appropriate sub-gate.
    """
    cfg = gate_config or DEFAULT_GATE_CONFIG
    gate = cfg.get(mode) or {}
    mask = pd.Series(True, index=df.index, dtype=bool)

    spread_col = cfg.get("spread_column")
    if spread_col and gate.get("hl_spread_max") is not None and spread_col in df.columns:
        try:
            mask &= df[spread_col].astype(float) <= float(gate["hl_spread_max"])
        except Exception:
            mask &= False

    if gate.get("hl_spread_z_max") is not None and "hl_spread_z" in df.columns:
        try:
            mask &= df["hl_spread_z"].astype(float) <= float(gate["hl_spread_z_max"])
        except Exception:
            mask &= False

    if gate.get("rvol20_max") is not None and "rvol_20" in df.columns:
        try:
            mask &= df["rvol_20"].astype(float) <= float(gate["rvol20_max"])
        except Exception:
            mask &= False

    prob_col = cfg.get("prob_column", "base_prob")
    prob_threshold = gate.get("prob_gate_min")
    if prob_threshold is not None:
        if prob is None:
            if prob_col not in df.columns:
                raise KeyError(f"Probability column '{prob_col}' required for gate evaluation")
            prob_series = df[prob_col]
        else:
            prob_series = prob
        if not isinstance(prob_series, pd.Series):
            prob_series = pd.Series(prob_series, index=df.index)
        else:
            prob_series = prob_series.reindex(df.index)
        mask &= prob_series.astype(float) >= float(prob_threshold)

    return mask.fillna(False)


def apply_manifest_gates(
    df: pd.DataFrame,
    gate_source: Union[Path, ManifestArtifacts],
    *,
    prob_series: Optional[pd.Series] = None,
    mode: str = "inference",
    model_label: Optional[str] = None,
    update_metrics: bool = True,
) -> Tuple[pd.Series, ManifestArtifacts]:
    """
    Apply the manifest-defined gate predicates to a feature frame and optional probability series.

    Parameters
    ----------
    df:
        Feature frame containing the columns referenced by the gate predicates.
    gate_source:
        Either a Path to the model directory (containing manifest.json) or a pre-loaded ManifestArtifacts instance.
    prob_series:
        Optional probability series aligned with df.index; when omitted the manifest's prob_column
        is pulled directly from df.
    mode:
        "inference" or "training" – selects the corresponding section of the manifest gate config.
    model_label:
        Optional override for the model label used in monitoring metrics when loading from Path.
    update_metrics:
        When True, Prometheus gauges are updated with coverage/RSS/probability sigma observations.
    """
    if isinstance(gate_source, ManifestArtifacts):
        artifacts = gate_source
    else:
        artifacts = load_manifest_artifacts(Path(gate_source), model_label=model_label)

    prob = prob_series
    if prob is None:
        prob_col = artifacts.prob_column or DEFAULT_GATE_CONFIG.get("prob_column", "base_prob")
        if prob_col not in df.columns:
            raise KeyError(f"Probability column '{prob_col}' required for gate evaluation")
        prob = df[prob_col]

    if not isinstance(prob, pd.Series):
        prob = pd.Series(prob, index=df.index)
    else:
        prob = prob.reindex(df.index)

    mask = compute_gate_mask(df, artifacts.gate_config, prob=prob, mode=mode)
    mask = mask.reindex(df.index).fillna(False).astype(bool)

    if update_metrics:
        _register_metric_thresholds(artifacts)
        _update_inference_metrics(df, prob, mask, artifacts, mode=mode)

    return mask, artifacts


def score_base_with_manifest(
    df: pd.DataFrame,
    model_dir: Path,
    *,
    mode: str = "inference",
    model_label: Optional[str] = None,
    update_metrics: bool = True,
) -> pd.DataFrame:
    """
    Score a batch with the base XGB model, attach the calibrated probability column, and
    annotate the manifest gate decision (gate_pass).
    """
    artifacts = load_manifest_artifacts(model_dir, model_label=model_label)
    calib, feat_cols = load_base_predictor(model_dir)
    prob_series = predict_base(df, calib, feat_cols)
    scored = df.copy()
    prob_col = artifacts.prob_column or "base_prob"
    scored[prob_col] = prob_series

    gate_mask, _ = apply_manifest_gates(
        scored,
        artifacts,
        prob_series=prob_series,
        mode=mode,
        update_metrics=update_metrics,
    )
    scored["gate_pass"] = gate_mask
    return scored


def load_tcn_predictor(tcn_dir: Path):
    meta = json.loads((tcn_dir / "tcn_meta.json").read_text())
    pre = joblib.load(tcn_dir / "tcn_preproc.joblib")
    calib = joblib.load(tcn_dir / "tcn_calibrator.joblib")
    series_cols = pre["series_cols"]
    scaler = pre["scaler"]
    channels = tuple(int(x) for x in meta.get("channels", [32, 32]))
    kernel_size = int(meta.get("kernel_size", 3))
    window = int(meta.get("window", 32))
    model = TinyTCN(n_inputs=len(series_cols), channels=channels, kernel_size=kernel_size)
    state = torch.load(tcn_dir / "tcn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, calib, series_cols, scaler, window


def predict_tcn(df: pd.DataFrame, model: TinyTCN, calib, series_cols: List[str], scaler, window: int, *, stride: int = 1) -> pd.DataFrame:
    series_df = df[series_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    series_df = series_df.ffill().bfill().fillna(0.0)
    vals = series_df.values
    if scaler is not None:
        try:
            vals = scaler.transform(vals)
        except Exception:
            pass
    n, c = vals.shape
    L = window
    stride = max(1, int(stride))
    starts = list(range(0, max(0, n - L), stride))
    N = len(starts)
    if N == 0:
        return pd.DataFrame(columns=["timestamp", "tcn_prob"])  # not enough data
    batch_size = max(256, min(4096, max(1, 8192 // stride)))
    prob_chunks: List[np.ndarray] = []
    ts_idx: List[int] = []
    for offset in range(0, N, batch_size):
        batch_starts = starts[offset:offset + batch_size]
        batch_len = len(batch_starts)
        if batch_len == 0:
            continue
        X_batch = np.empty((batch_len, c, L), dtype=np.float32)
        ts_batch: List[int] = []
        for i, start in enumerate(batch_starts):
            seg = vals[start:start + L, :].T
            m = seg.mean(axis=1, keepdims=True)
            s = seg.std(axis=1, keepdims=True) + 1e-6
            seg = (seg - m) / s
            X_batch[i] = seg
            ts_batch.append(start + L)
        with torch.no_grad():
            logits_batch = model(torch.from_numpy(X_batch)).view(-1).cpu().numpy()
        prob_batch = calib.predict_proba(logits_batch.reshape(-1, 1))[:, 1]
        prob_chunks.append(prob_batch)
        ts_idx.extend(ts_batch)
    if not prob_chunks:
        return pd.DataFrame(columns=["timestamp", "tcn_prob"])
    prob = np.concatenate(prob_chunks)
    ts = pd.to_datetime(df["timestamp"], utc=True).iloc[ts_idx].reset_index(drop=True)
    out = pd.DataFrame({"timestamp": ts, "tcn_prob": prob})
    return out
