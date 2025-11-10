from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import ensure_labels
from .data import sliding_windows
from .feature_eng import augment_market_features
from .blender import build_blender_features
from .infer import (
    DEFAULT_GATE_CONFIG,
    compute_gate_mask,
    load_gate_config,
    load_base_predictor,
    predict_base,
)
from .calibration_store import load_calibrator
from .calibration_utils import apply_posthoc_calibration
from .metrics import equity_curve, summary_stats
from .model import (
    calibrate,
    extract_features_labels,
    predict_proba,
    train_xgb,
)
from .thresholds import select_prob_threshold
from .reporting import ensure_kpi_schema, social_signal_audit
from .walkforward import time_folds
from .tcn_model import TrainConfig, calibrate_logits, train_tcn


@dataclass
class EvaluationSummary:
    """
    Container for out-of-sample evaluation artifacts.

    Attributes
    ----------
    oos_frame:
        Row-level out-of-sample predictions (one row per validation observation).
    threshold:
        Probability threshold selected under the aligned gate configuration.
    training_report:
        Aggregated metrics computed under the training gate.
    inference_report:
        Aggregated metrics computed under the inference gate.
    gate_config:
        Gate configuration used for both reports (after optional alignment).
    auc:
        ROC AUC computed from the OOF predictions.
    diagnostics:
        Optional extra derived metrics (fold-level stats, etc.).
    """

    oos_frame: pd.DataFrame
    threshold: float
    training_report: Dict[str, Any]
    inference_report: Dict[str, Any]
    gate_config: Dict[str, Any]
    auc: float
    diagnostics: Dict[str, Any]


def _resolve_gate_config(
    gate_config: Optional[Dict[str, Any]],
    *,
    align_training_with_inference: bool,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = copy.deepcopy(gate_config or fallback or DEFAULT_GATE_CONFIG)
    if "training" not in base or not isinstance(base["training"], dict):
        base["training"] = {}
    if "inference" not in base or not isinstance(base["inference"], dict):
        base["inference"] = {}
    if align_training_with_inference:
        # Keep a copy so callers can still inspect the original inference gate
        base["training"] = copy.deepcopy(base["inference"])
    return base


def _compute_sample_weight(
    scheme: str,
    ret_next: pd.Series,
    *,
    cost_bps: float,
) -> Optional[np.ndarray]:
    if scheme == "none":
        return None
    weights: np.ndarray
    if scheme == "abs_return":
        weights = np.abs(ret_next.to_numpy(dtype=float)) * 1e4
    elif scheme == "cost_margin":
        weights = np.abs(ret_next.to_numpy(dtype=float)) * 1e4 - float(cost_bps)
        weights = np.clip(weights, 0.0, None)
    else:
        raise ValueError(f"Unknown sample weight scheme: {scheme}")
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return weights * (len(weights) / total)


def _build_gate_series(
    df_subset: pd.DataFrame,
    prob: pd.Series,
    gate_config: Dict[str, Any],
    *,
    mode: str,
) -> pd.Series:
    gate_series = compute_gate_mask(
        df_subset,
        gate_config,
        mode=mode,
        prob=prob,
    )
    gate_series = gate_series.reindex(prob.index)
    return gate_series.fillna(False).astype(bool)


def evaluate_base_xgb_oos(
    df: pd.DataFrame,
    *,
    gate_config: Optional[Dict[str, Any]] = None,
    align_gates: bool = True,
    n_folds: int = 6,
    embargo_minutes: int = 60,
    fold_scheme: str = "even",
    cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    spread_col: Optional[str] = "hl_spread",
    spread_scale: float = 0.0,
    sample_weight_scheme: str = "none",
    calibration_method: str = "isotonic",
    xgb_params: Optional[Dict[str, Any]] = None,
    threshold_grid: Optional[Iterable[float]] = None,
    min_total_turnover: float = 2.0,
    max_total_turnover: Optional[float] = None,
) -> EvaluationSummary:
    """
    Run walk-forward out-of-sample evaluation for the base XGBoost family.

    Parameters mirror the CLI defaults used in training scripts so the results
    remain reproducible.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty; cannot run OOS evaluation.")

    df_proc = ensure_labels(df).copy()
    if "timestamp" not in df_proc.columns:
        raise KeyError("Dataset must include a timestamp column for leak-proof folds.")
    df_proc = augment_market_features(df_proc)
    df_proc = df_proc.sort_values("timestamp").reset_index(drop=True)
    rss_audit = social_signal_audit(df_proc)

    gate_cfg = _resolve_gate_config(
        gate_config,
        align_training_with_inference=align_gates,
    )
    spread_series_full = None
    if spread_scale != 0.0 and spread_col and spread_col in df_proc.columns:
        spread_series_full = df_proc[spread_col].astype(float)

    X_full, y_full, feat_cols = extract_features_labels(df_proc)
    if len(feat_cols) == 0:
        raise ValueError("No feature columns detected for base XGB evaluation.")

    n_rows = len(df_proc)
    prob_oof = np.zeros(n_rows, dtype=float)
    prob_raw_oof = np.zeros(n_rows, dtype=float)
    fold_assign = np.full(n_rows, -1, dtype=int)
    oof_mask = np.zeros(n_rows, dtype=bool)

    for fold_idx, (tr_idx, va_idx) in enumerate(
        time_folds(
            df_proc,
            n_folds=n_folds,
            embargo_minutes=embargo_minutes,
            scheme=fold_scheme,
        ),
        start=1,
    ):
        X_tr = X_full.iloc[tr_idx]
        y_tr = y_full.iloc[tr_idx]
        X_va = X_full.iloc[va_idx]
        y_va = y_full.iloc[va_idx]

        sample_weight = None
        if sample_weight_scheme != "none":
            sample_weight = _compute_sample_weight(
                sample_weight_scheme,
                df_proc.loc[X_tr.index, "ret_next"],
                cost_bps=cost_bps,
            )

        booster = train_xgb(
            X_tr,
            y_tr,
            params=xgb_params,
            sample_weight=sample_weight,
        )

        if calibration_method == "none":
            calibrator = booster
        else:
            calibrator = calibrate(booster, X_va, y_va, method=calibration_method)

        prob_calibrated = predict_proba(calibrator, X_va)
        prob_uncalibrated = predict_proba(booster, X_va)

        prob_oof[va_idx] = prob_calibrated
        prob_raw_oof[va_idx] = prob_uncalibrated
        fold_assign[va_idx] = fold_idx
        oof_mask[va_idx] = True

    valid_idx = np.where(oof_mask)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No validation indices populated during OOS evaluation.")

    prob_series = pd.Series(prob_oof[valid_idx], index=df_proc.index[valid_idx], name="p_cal")
    prob_raw_series = pd.Series(prob_raw_oof[valid_idx], index=df_proc.index[valid_idx], name="p_raw")
    ret_series = df_proc.loc[valid_idx, "ret_next"].astype(float)
    y_series = df_proc.loc[valid_idx, "y_dir"].astype(int)
    timestamps = pd.to_datetime(df_proc.loc[valid_idx, "timestamp"], utc=True)

    gate_train_series = _build_gate_series(
        df_proc.loc[valid_idx],
        prob_series,
        gate_cfg,
        mode="training",
    )
    gate_infer_series = _build_gate_series(
        df_proc.loc[valid_idx],
        prob_series,
        gate_cfg,
        mode="inference",
    )

    spread_series = None
    if spread_series_full is not None:
        spread_series = spread_series_full.iloc[valid_idx]

    thr, report_train = select_prob_threshold(
        ret_series,
        prob_series,
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("training", {}).get("long_only", True)),
        gate_mask=gate_train_series,
        grid=np.array(list(threshold_grid)) if threshold_grid is not None else None,
        min_hold_bars=int(max(1, gate_cfg.get("training", {}).get("min_hold_bars", 1))),
        min_total_turnover=float(min_total_turnover),
        max_total_turnover=max_total_turnover,
    )
    eq_infer = equity_curve(
        ret_series,
        prob_series,
        threshold=float(thr),
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("inference", {}).get("long_only", True)),
        gate_mask=gate_infer_series,
        min_hold_bars=int(max(1, gate_cfg.get("inference", {}).get("min_hold_bars", 1))),
    )
    report_infer = summary_stats(eq_infer)

    report_train = dict(report_train)
    report_train["gate_fraction"] = float(gate_train_series.mean())
    report_train["oof_count"] = int(len(valid_idx))
    if spread_series is not None:
        report_train["spread_column"] = spread_col
        report_train["spread_scale"] = float(spread_scale)

    report_infer = dict(report_infer)
    report_infer["gate_fraction"] = float(gate_infer_series.mean())
    report_infer["oof_count"] = int(len(valid_idx))
    report_train["rss_audit"] = rss_audit
    report_infer["rss_audit"] = rss_audit
    report_train = ensure_kpi_schema(report_train)
    report_infer = ensure_kpi_schema(report_infer)

    auc = roc_auc_score(y_series, prob_series)

    oos_frame = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "fold": fold_assign[valid_idx].astype(int),
            "prob_calibrated": prob_series.to_numpy(),
            "prob_uncalibrated": prob_raw_series.to_numpy(),
            "label": y_series.to_numpy(),
            "ret_next": ret_series.to_numpy(),
            "gate_training": gate_train_series.to_numpy(dtype=bool),
            "gate_inference": gate_infer_series.to_numpy(dtype=bool),
        }
    )

    diagnostics: Dict[str, Any] = {
        "feature_cols": feat_cols,
    }

    return EvaluationSummary(
        oos_frame=oos_frame,
        threshold=float(thr),
        training_report=report_train,
        inference_report=report_infer,
        gate_config=gate_cfg,
        auc=float(auc),
        diagnostics=diagnostics,
    )


def evaluate_blender_oos(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    gate_config: Optional[Dict[str, Any]] = None,
    align_gates: bool = True,
    cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    spread_col: Optional[str] = "hl_spread",
    spread_scale: float = 0.0,
) -> EvaluationSummary:
    """
    Evaluate a pre-trained blender model against an OOS dataset using the persisted manifest.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty; cannot run OOS evaluation.")

    model_dir = Path(model_dir)
    feat_path = model_dir / "blender_features.txt"
    if not feat_path.exists():
        raise FileNotFoundError(f"Blender feature list missing: {feat_path}")
    feature_list = [line.strip() for line in feat_path.read_text().splitlines() if line.strip()]
    if not feature_list:
        raise ValueError(f"No blender feature columns declared in {feat_path}")

    model_path = model_dir / "blender.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Blender model artifact missing: {model_path}")
    blender_model = joblib.load(model_path)

    threshold_path = model_dir / "threshold.txt"
    try:
        threshold_value = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5
    except ValueError:
        threshold_value = 0.5

    gate_cfg = _resolve_gate_config(
        gate_config,
        align_training_with_inference=align_gates,
    )
    prob_col = str(gate_cfg.get("prob_column") or "blender_prob")

    df_proc = ensure_labels(df).copy()
    if "timestamp" not in df_proc.columns:
        raise KeyError("Dataset must include a timestamp column for leak-proof folds.")
    df_proc = augment_market_features(df_proc)
    df_proc = df_proc.sort_values("timestamp").reset_index(drop=True)
    rss_audit = social_signal_audit(df_proc)

    X, cols = build_blender_features(df_proc, candidate_cols=feature_list)
    if X.empty:
        raise ValueError("Blender feature frame is empty after preprocessing.")
    prob = blender_model.predict_proba(X.values)[:, 1]
    prob_series = pd.Series(prob, index=df_proc.index, name=prob_col)
    prob_calibrator = load_calibrator(model_dir, prob_col)
    if prob_calibrator is not None:
        calibrated = apply_posthoc_calibration(
            prob_series.to_numpy(),
            method=prob_calibrator.method,
            estimator=prob_calibrator.estimator,
        )
        prob_series = pd.Series(calibrated, index=prob_series.index, name=prob_col)
    df_proc[prob_col] = prob_series

    ret_series = df_proc["ret_next"].astype(float)
    y_series = df_proc["y_dir"].astype(int)
    timestamps = pd.to_datetime(df_proc["timestamp"], utc=True, errors="coerce")

    gate_train_series = _build_gate_series(
        df_proc,
        prob_series,
        gate_cfg,
        mode="training",
    )
    gate_infer_series = _build_gate_series(
        df_proc,
        prob_series,
        gate_cfg,
        mode="inference",
    )

    spread_series = None
    if spread_scale != 0.0 and spread_col and spread_col in df_proc.columns:
        spread_series = df_proc[spread_col].astype(float)

    train_min_hold = int(max(1, gate_cfg.get("training", {}).get("min_hold_bars", 1)))
    infer_min_hold = int(max(1, gate_cfg.get("inference", {}).get("min_hold_bars", 1)))

    eq_train = equity_curve(
        ret_series,
        prob_series,
        threshold=float(threshold_value),
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("training", {}).get("long_only", True)),
        gate_mask=gate_train_series,
        min_hold_bars=train_min_hold,
    )
    eq_infer = equity_curve(
        ret_series,
        prob_series,
        threshold=float(threshold_value),
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("inference", {}).get("long_only", True)),
        gate_mask=gate_infer_series,
        min_hold_bars=infer_min_hold,
    )

    report_train = summary_stats(eq_train)
    report_train["gate_fraction"] = float(gate_train_series.mean())
    report_train["rss_audit"] = rss_audit
    report_train["oos_count"] = int(len(df_proc))
    report_train["selected_threshold"] = float(threshold_value)
    report_train["criterion"] = report_train.get("criterion", "final_equity")
    report_train["cost_bps"] = float(cost_bps)
    report_train["spread_scale"] = float(spread_scale)
    report_train["slippage_bps"] = float(slippage_bps)
    report_train["long_only"] = bool(gate_cfg.get("training", {}).get("long_only", True))
    report_train["min_hold_bars"] = train_min_hold
    if spread_series is not None and spread_col:
        report_train["spread_column"] = spread_col
    report_train = ensure_kpi_schema(report_train)

    report_infer = summary_stats(eq_infer)
    report_infer["gate_fraction"] = float(gate_infer_series.mean())
    report_infer["rss_audit"] = rss_audit
    report_infer["oos_count"] = int(len(df_proc))
    report_infer["selected_threshold"] = float(threshold_value)
    report_infer["criterion"] = report_infer.get("criterion", "final_equity")
    report_infer["cost_bps"] = float(cost_bps)
    report_infer["spread_scale"] = float(spread_scale)
    report_infer["slippage_bps"] = float(slippage_bps)
    report_infer["long_only"] = bool(gate_cfg.get("inference", {}).get("long_only", True))
    report_infer["min_hold_bars"] = infer_min_hold
    if spread_series is not None and spread_col:
        report_infer["spread_column"] = spread_col
    report_infer = ensure_kpi_schema(report_infer)

    try:
        auc = float(roc_auc_score(y_series, prob_series))
    except ValueError:
        auc = float("nan")

    oos_frame = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "prob_calibrated": prob_series.to_numpy(),
            "prob_uncalibrated": prob_series.to_numpy(),
            "label": y_series.to_numpy(),
            "ret_next": ret_series.to_numpy(),
            "gate_training": gate_train_series.to_numpy(dtype=bool),
            "gate_inference": gate_infer_series.to_numpy(dtype=bool),
        }
    )

    diagnostics: Dict[str, Any] = {
        "feature_cols": cols,
        "model_dir": str(model_dir),
        "threshold": float(threshold_value),
    }

    return EvaluationSummary(
        oos_frame=oos_frame,
        threshold=float(threshold_value),
        training_report=report_train,
        inference_report=report_infer,
        gate_config=gate_cfg,
        auc=auc,
        diagnostics=diagnostics,
    )


def load_model_gate_config(model_dir: Path, *, fallback_align: bool = True) -> Dict[str, Any]:
    """
    Utility to load a model's persisted gate configuration (manifest preferred).
    """
    return _resolve_gate_config(
        load_gate_config(model_dir),
        align_training_with_inference=fallback_align,
    )


def evaluate_tcn_oos(
    df: pd.DataFrame,
    *,
    gate_config: Optional[Dict[str, Any]] = None,
    align_gates: bool = True,
    base_model_dir: Optional[Path] = None,
    window: int = 32,
    kernel_size: int = 3,
    channels: Sequence[int] = (32, 32),
    dropout: float = 0.05,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight: Optional[float] = None,
    stride: int = 1,
    n_folds: int = 6,
    embargo_minutes: int = 60,
    fold_scheme: str = "even",
    cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    spread_col: Optional[str] = "hl_spread",
    spread_scale: float = 0.0,
    threshold_grid: Optional[Iterable[float]] = None,
    min_total_turnover: float = 2.0,
    max_total_turnover: Optional[float] = None,
    series_cols_override: Optional[Sequence[str]] = None,
) -> EvaluationSummary:
    """
    Walk-forward evaluation for the TCN family using sliding windows.

    The dataset must already contain the feature channels referenced by
    `series_cols_override` (or the defaults selected by `sliding_windows`).
    """
    if df.empty:
        raise ValueError("Input dataframe is empty; cannot run OOS evaluation.")

    df_proc = ensure_labels(df).copy()
    if "timestamp" not in df_proc.columns:
        raise KeyError("Dataset must include timestamp for leak-proof folds.")
    df_proc = augment_market_features(df_proc)
    if base_model_dir is not None:
        base_model_dir = Path(base_model_dir)
        if not base_model_dir.exists():
            raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
        calib, feat_cols = load_base_predictor(base_model_dir)
        try:
            base_prob = predict_base(df_proc, calib, feat_cols)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate base probabilities using {base_model_dir}") from exc
        df_proc["base_prob"] = base_prob.reindex(df_proc.index).fillna(0.0).astype(float)
    df_proc = df_proc.sort_values("timestamp").reset_index(drop=True)
    rss_audit = social_signal_audit(df_proc)

    X, y, ts, series_cols, _ = sliding_windows(
        df_proc,
        window=int(window),
        series_cols=list(series_cols_override) if series_cols_override is not None else None,
        stride=max(1, int(stride)),
    )
    if len(ts) == 0:
        raise ValueError("Sliding window construction returned no samples; check window/stride sizing.")

    # Align auxiliary columns (ret_next, spreads, etc.) to the windowed frame.
    aligned = df_proc.iloc[df_proc.index[-len(ts):]].reset_index(drop=True)
    win_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "y_dir": y.astype(int),
            "ret_next": aligned["ret_next"].astype(float).to_numpy(),
        }
    )
    for col in ("hl_spread_z", "rvol_20"):
        if col in aligned.columns:
            win_df[col] = aligned[col].astype(float).to_numpy()
    if spread_col and spread_col in aligned.columns:
        win_df[spread_col] = aligned[spread_col].astype(float).to_numpy()

    gate_cfg = _resolve_gate_config(
        gate_config,
        align_training_with_inference=align_gates,
    )

    n_rows = len(win_df)
    prob_oof = np.zeros(n_rows, dtype=float)
    prob_raw_oof = np.zeros(n_rows, dtype=float)
    fold_assign = np.full(n_rows, -1, dtype=int)
    mask_oof = np.zeros(n_rows, dtype=bool)

    train_cfg = TrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
        weight_decay=float(weight_decay),
        class_weight=float(class_weight) if class_weight is not None else None,
    )

    fold_records: List[pd.DataFrame] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        time_folds(
            win_df,
            n_folds=n_folds,
            embargo_minutes=embargo_minutes,
            scheme=fold_scheme,
        ),
        start=1,
    ):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model, logits_tr, logits_va = train_tcn(
            X_tr,
            y_tr,
            val=(X_va, y_va),
            kernel_size=int(kernel_size),
            channels=tuple(int(c) for c in channels),
            dropout=float(dropout),
            config=train_cfg,
        )
        if logits_va is None or not np.isfinite(logits_va).all():
            raise RuntimeError("Validation logits contain non-finite values; aborting OOS evaluation.")

        calib = calibrate_logits(logits_va, y_va, method="isotonic")
        prob_cal = calib.predict_proba(logits_va.reshape(-1, 1))[:, 1]
        prob_raw = 1.0 / (1.0 + np.exp(-np.clip(logits_va, -20, 20)))

        prob_oof[va_idx] = prob_cal
        prob_raw_oof[va_idx] = prob_raw
        fold_assign[va_idx] = fold_idx
        mask_oof[va_idx] = True

        fold_records.append(
            pd.DataFrame(
                {
                    "timestamp": win_df.loc[va_idx, "timestamp"].to_numpy(),
                    "fold": fold_idx,
                    "logit": logits_va,
                    "prob_uncalibrated": prob_raw,
                    "prob_calibrated": prob_cal,
                    "label": y_va.astype(int),
                }
            )
        )

    valid_idx = np.where(mask_oof)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No validation indices populated during TCN OOS evaluation.")

    prob_series = pd.Series(prob_oof[valid_idx], index=win_df.index[valid_idx], name="p_cal")
    prob_raw_series = pd.Series(prob_raw_oof[valid_idx], index=win_df.index[valid_idx], name="p_raw")
    ret_series = win_df.loc[valid_idx, "ret_next"].astype(float)
    y_series = win_df.loc[valid_idx, "y_dir"].astype(int)
    timestamps = win_df.loc[valid_idx, "timestamp"]

    gate_train_series = _build_gate_series(
        win_df.loc[valid_idx],
        prob_series,
        gate_cfg,
        mode="training",
    )
    gate_infer_series = _build_gate_series(
        win_df.loc[valid_idx],
        prob_series,
        gate_cfg,
        mode="inference",
    )

    spread_series = None
    if spread_scale != 0.0 and spread_col and spread_col in win_df.columns:
        spread_series = win_df.loc[valid_idx, spread_col].astype(float)

    thr, report_train = select_prob_threshold(
        ret_series,
        prob_series,
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("training", {}).get("long_only", False)),
        gate_mask=gate_train_series,
        grid=np.array(list(threshold_grid)) if threshold_grid is not None else None,
        min_hold_bars=int(max(1, gate_cfg.get("training", {}).get("min_hold_bars", 1))),
        min_total_turnover=float(min_total_turnover),
        max_total_turnover=max_total_turnover,
    )

    eq_infer = equity_curve(
        ret_series,
        prob_series,
        threshold=float(thr),
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
        long_only=bool(gate_cfg.get("inference", {}).get("long_only", False)),
        gate_mask=gate_infer_series,
        min_hold_bars=int(max(1, gate_cfg.get("inference", {}).get("min_hold_bars", 1))),
    )
    report_infer = summary_stats(eq_infer)

    report_train = dict(report_train)
    report_train["gate_fraction"] = float(gate_train_series.mean())
    report_train["oof_count"] = int(len(valid_idx))
    report_train["series_cols"] = list(series_cols)

    report_infer = dict(report_infer)
    report_infer["gate_fraction"] = float(gate_infer_series.mean())
    report_infer["oof_count"] = int(len(valid_idx))
    report_train["rss_audit"] = rss_audit
    report_infer["rss_audit"] = rss_audit
    report_train = ensure_kpi_schema(report_train)
    report_infer = ensure_kpi_schema(report_infer)

    auc = roc_auc_score(y_series, prob_series)

    oos_frame = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "fold": fold_assign[valid_idx].astype(int),
            "prob_calibrated": prob_series.to_numpy(),
            "prob_uncalibrated": prob_raw_series.to_numpy(),
            "label": y_series.to_numpy(),
            "ret_next": ret_series.to_numpy(),
            "gate_training": gate_train_series.to_numpy(dtype=bool),
            "gate_inference": gate_infer_series.to_numpy(dtype=bool),
        }
    )

    diagnostics: Dict[str, Any] = {
        "series_cols": list(series_cols),
        "fold_logits": pd.concat(fold_records, ignore_index=True) if fold_records else pd.DataFrame(),
    }

    return EvaluationSummary(
        oos_frame=oos_frame,
        threshold=float(thr),
        training_report=report_train,
        inference_report=report_infer,
        gate_config=gate_cfg,
        auc=float(auc),
        diagnostics=diagnostics,
    )


def compare_oos_frames(current: pd.DataFrame, baseline: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare two OOS evaluation frames and return summary deltas.
    """
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df_sorted = df.sort_values("timestamp")
        else:
            df_sorted = df.copy()
        return df_sorted.reset_index(drop=True)

    cur = _prepare(current)
    base = _prepare(baseline)
    n_cur = len(cur)
    n_base = len(base)
    overlap = min(n_cur, n_base)

    metrics: Dict[str, Any] = {
        "row_count_current": int(n_cur),
        "row_count_baseline": int(n_base),
        "overlap_rows": int(overlap),
        "overlap_ratio": float(overlap / max(1, max(n_cur, n_base))),
    }
    if overlap == 0:
        return metrics

    columns = {
        "prob_calibrated": "mae",
        "prob_uncalibrated": "mae",
        "ret_next": "mae",
        "label": "mismatch_rate",
        "gate_training": "mismatch_rate",
        "gate_inference": "mismatch_rate",
    }
    for col, mode in columns.items():
        if col not in cur.columns or col not in base.columns:
            continue
        cur_vals = cur[col].to_numpy()[:overlap]
        base_vals = base[col].to_numpy()[:overlap]
        if mode == "mae":
            metrics[f"{col}_mae"] = float(np.mean(np.abs(cur_vals - base_vals)))
            if col == "prob_calibrated" and overlap >= 2:
                try:
                    corr = np.corrcoef(cur_vals, base_vals)[0, 1]
                    if np.isfinite(corr):
                        metrics["prob_calibrated_corr"] = float(corr)
                except Exception:
                    pass
        else:
            metrics[f"{col}_mismatch_rate"] = float(np.mean(cur_vals != base_vals))
    return metrics
