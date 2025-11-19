from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from training.data import load_canonical_contract, load_training_dataset
from training.model import extract_features_labels, train_xgb, predict_proba as predict_xgb_proba
from training.transformer_model import TransformerModel
from training.tcn_model import TrainConfig, train_tcn, TinyTCN
from training.data import sliding_windows
from training.metrics import equity_curve, summary_stats


@dataclass
class DiagnosticsConfig:
    contract_path: str
    best_model_configs_path: str
    model_name: str = "xgb"
    label_key: Optional[str] = None
    regime_col: Optional[str] = "regime_id"
    train_fraction: float = 0.7
    random_state: int = 42
    low_cost_bps: float = 0.25
    entry_long: float = 0.52
    exit_long: float = 0.50


def _split(df: pd.DataFrame, fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = max(1, min(n - 1, int(n * fraction)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 5:
        return float("nan")
    corr = np.corrcoef(y_pred[mask], y_true[mask])[0, 1]
    return float(corr)


def _extract_label_and_return(df: pd.DataFrame, label_key: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    lbl_col = label_key or ("y_dir" if "y_dir" in df.columns else None)
    if lbl_col is None or lbl_col not in df.columns:
        raise KeyError("Unable to resolve label column for diagnostics (expected y_dir or provided label_key).")
    y = df[lbl_col].astype(float)
    # realized return fallback
    ret_col = "ret_next" if "ret_next" in df.columns else lbl_col
    ret = df[ret_col].astype(float)
    return y, ret


def _train_predict_xgb(df_train: pd.DataFrame, df_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    X_train, y_train, feat_cols = extract_features_labels(df_train)
    # Drop any object columns that may have slipped through
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_oos = df_oos[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    model = train_xgb(X_train, y_train, params=params, early_stopping_rounds=0)
    return predict_xgb_proba(model, X_oos)


def _train_predict_transformer(df_train: pd.DataFrame, df_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    window = int(params.get("seq_len", params.get("window", 64)))
    stride = int(params.get("seq_stride", params.get("stride", 10)))
    X_tr, y_tr, ts_tr, _, _ = sliding_windows(df_train, window=window, stride=stride)
    X_va, _, ts_va, _, _ = sliding_windows(df_oos, window=window, stride=stride)
    if len(ts_tr) == 0 or len(ts_va) == 0:
        raise ValueError("Transformer diagnostics require enough rows to build windows.")
    model = TransformerModel(config=params)
    model.fit(X_tr, y_tr)
    prob_va = model.predict_proba(X_va)
    # align back to tail of oos frame
    tail = len(prob_va)
    idx = df_oos.index[-tail:]
    probs = pd.Series(prob_va, index=idx).reindex(df_oos.index).fillna(method="ffill").fillna(method="bfill").to_numpy()
    return probs


def _train_predict_tcn(df_train: pd.DataFrame, df_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    window = int(params.get("seq_len", params.get("window", 48)))
    stride = int(params.get("seq_stride", params.get("stride", 10)))
    X_tr, y_tr, ts_tr, series_cols, _ = sliding_windows(df_train, window=window, stride=stride)
    X_va, _, ts_va, _, _ = sliding_windows(df_oos, window=window, stride=stride)
    if len(ts_tr) == 0 or len(ts_va) == 0:
        raise ValueError("TCN diagnostics require enough rows to build windows.")
    train_cfg = TrainConfig(
        epochs=int(params.get("epochs", 5)),
        lr=float(params.get("lr", 1e-3)),
        batch_size=int(params.get("batch_size", 128)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        class_weight=None,
    )
    model, _, _ = train_tcn(X_tr, y_tr, train_cfg=train_cfg)
    tiny = TinyTCN(n_inputs=X_tr.shape[1], channels=(32, 32), kernel_size=int(params.get("kernel_size", 3)))
    tiny.load_state_dict(model.state_dict())
    logits = tiny(torch.tensor(X_va)).detach().cpu().numpy()
    prob_va = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    tail = len(prob_va)
    idx = df_oos.index[-tail:]
    probs = pd.Series(prob_va, index=idx).reindex(df_oos.index).fillna(method="ffill").fillna(method="bfill").to_numpy()
    return probs


def _compute_signal_metrics(y_true: np.ndarray, y_prob: np.ndarray, ret: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_f = y_true[mask]
    y_prob_f = y_prob[mask]
    if len(y_true_f) >= 5:
        try:
            out["auc"] = float(roc_auc_score(y_true_f, y_prob_f))
        except Exception:
            out["auc"] = float("nan")
        try:
            out["brier"] = float(brier_score_loss(y_true_f, y_prob_f))
        except Exception:
            out["brier"] = float("nan")
        try:
            out["log_loss"] = float(log_loss(y_true_f, y_prob_f))
        except Exception:
            out["log_loss"] = float("nan")
    else:
        out["auc"] = out["brier"] = out["log_loss"] = float("nan")
    out["ic"] = _ic(y_prob, ret)
    out["count"] = int(len(y_true_f))
    return out


def _compute_pnl_metrics(ret_next: np.ndarray, prob: np.ndarray, cfg: DiagnosticsConfig) -> Dict[str, Any]:
    ret_series = pd.Series(ret_next.astype(float))
    prob_series = pd.Series(prob.astype(float))
    eq = equity_curve(
        ret_series,
        prob_series,
        threshold=float(cfg.entry_long),
        cost_bps=float(cfg.low_cost_bps),
        long_only=True,
        gate_mask=None,
        min_hold_bars=1,
    )
    return summary_stats(eq)


def run_edge_diagnostics(
    contract_path: str,
    best_model_configs_path: str,
    model_name: str = "xgb",
    label_key: Optional[str] = None,
    regime_col: Optional[str] = "regime_id",
    train_fraction: float = 0.7,
    random_state: int = 42,
    low_cost_bps: float = 0.25,
) -> Dict[str, Any]:
    """
    Run edge diagnostics for a single model on the canonical dataset.
    """
    cfg = DiagnosticsConfig(
        contract_path=contract_path,
        best_model_configs_path=best_model_configs_path,
        model_name=model_name,
        label_key=label_key,
        regime_col=regime_col,
        train_fraction=train_fraction,
        random_state=random_state,
        low_cost_bps=low_cost_bps,
    )
    contract = load_canonical_contract(contract_path)
    df = load_training_dataset(contract)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Drop obvious non-numeric columns except regime/symbol
    drop_obj = [c for c in df.columns if df[c].dtype == object and c not in {"symbol", regime_col}]
    if drop_obj:
        df = df.drop(columns=drop_obj)
    # Ensure regime_id is numeric friendly for metrics
    if regime_col and regime_col in df.columns and df[regime_col].dtype == object:
        df[regime_col] = df[regime_col].astype("category").cat.codes

    y_all, ret_all = _extract_label_and_return(df, cfg.label_key)
    df_train, df_oos = _split(df, cfg.train_fraction)
    y_train, _ = _extract_label_and_return(df_train, cfg.label_key)
    y_oos, ret_oos = _extract_label_and_return(df_oos, cfg.label_key)

    text = Path(best_model_configs_path).read_text()
    try:
        best_cfg = json.loads(text)
    except Exception:
        best_cfg = {}
        try:
            best_cfg = yaml.safe_load(text) or {}
        except Exception:
            pass
    params = best_cfg.get(model_name, {}).get("params", {})

    if model_name == "xgb":
        prob_oos = _train_predict_xgb(df_train, df_oos, params)
    elif model_name == "transformer":
        prob_oos = _train_predict_transformer(df_train, df_oos, params)
    elif model_name == "tcn":
        prob_oos = _train_predict_tcn(df_train, df_oos, params)
    else:
        raise ValueError(f"Unsupported model for diagnostics: {model_name}")

    metrics_global = _compute_signal_metrics(y_oos.to_numpy(), prob_oos, ret_oos.to_numpy())
    pnl_global = _compute_pnl_metrics(ret_oos.to_numpy(), prob_oos, cfg)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "label_key": cfg.label_key or "y_dir",
        "train_fraction": cfg.train_fraction,
        "low_cost_bps": cfg.low_cost_bps,
        "global": {
            "metrics_signal": metrics_global,
            "metrics_pnl": pnl_global,
        },
        "by_regime": {},
    }

    if regime_col and regime_col in df_oos.columns:
        regimes = df_oos[regime_col]
        try:
            for reg, idx in regimes.groupby(regime_col).indices.items():
                idx_list = list(idx)
                if len(idx_list) < 10:
                    result["by_regime"][str(reg)] = {"insufficient_data": True, "count": len(idx_list)}
                    continue
                y_reg = y_oos.iloc[idx_list].to_numpy()
                ret_reg = ret_oos.iloc[idx_list].to_numpy()
                prob_reg = np.asarray(prob_oos)[idx_list]
                result["by_regime"][str(reg)] = {
                    "metrics_signal": _compute_signal_metrics(y_reg, prob_reg, ret_reg),
                    "metrics_pnl": _compute_pnl_metrics(ret_reg, prob_reg, cfg),
                }
        except KeyError:
            pass

    return result
