from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data import load_canonical_contract, load_training_dataset
from .sequence_builder import build_sequences
from .model_registry import get_model_class
from .time_series_cv import TimeSeriesSplitConfig, make_time_series_splits
from .walkforward import time_folds
from .thresholds import select_prob_threshold
from .metrics import equity_curve, summary_stats


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: List[str]
    label_col: str
    regime_cols: List[str]
    ret_col: str
    seq: Optional[SequenceDataset]
    test_start: int
    contract: Dict[str, Any]


@dataclass
class ModelRun:
    name: str
    model: Any
    oof_preds: pd.Series
    test_preds: pd.Series


def compute_regime_id(df: pd.DataFrame, regime_cols: List[str]) -> pd.Series:
    if not regime_cols:
        return pd.Series(["default"] * len(df), index=df.index)
    reg_df = df[regime_cols].astype(str)
    return reg_df.apply(lambda row: "|".join(row.values), axis=1)


def prepare_canonical_data(
    contract_path: str,
    *,
    seq_len: int = 32,
    horizon: int = 1,
    test_size: float = 0.2,
    seq_stride: int = 1,
) -> DatasetBundle:
    contract = load_canonical_contract(contract_path)
    df = load_training_dataset(contract)
    df = df.sort_values(["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]).reset_index(drop=True)
    label_col = df.attrs.get("label_col") or contract.get("labels", {}).get("primary")
    feature_cols = df.attrs.get("feature_cols") or contract.get("features", {}).get("core", [])
    regime_cols = df.attrs.get("regime_cols") or []
    if label_col is None:
        raise KeyError("Primary label not defined in contract.")
    if feature_cols is None:
        feature_cols = []
    ret_col = "net_return_15m" if "net_return_15m" in df.columns else "ret_next"
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1.0 - float(test_size)))
    seq_data = None
    if seq_len and seq_len > 0:
        X_seq, y_seq, idx = build_sequences(
            df,
            feature_cols,
            label_col,
            seq_len=seq_len,
            horizon=horizon,
            return_index=True,
            stride=seq_stride,
        )
        seq_data = SequenceDataset(X_seq, y_seq, idx)
    return DatasetBundle(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        regime_cols=regime_cols,
        ret_col=ret_col,
        seq=seq_data,
        test_start=test_start,
        contract=contract,
    )


def generate_cv_predictions(
    model_name: str,
    bundle: DatasetBundle,
    *,
    n_folds: int = 4,
    embargo_minutes: int = 60,
    cv_config: TimeSeriesSplitConfig | None = None,
    model_config: Optional[Dict] = None,
    use_sequences: bool = False,
) -> ModelRun:
    ModelCls = get_model_class(model_name)
    df_train = bundle.df.iloc[: bundle.test_start]
    df_test = bundle.df.iloc[bundle.test_start :]
    oof = pd.Series(index=df_train.index, dtype=float)
    test_pred = pd.Series(index=df_test.index, dtype=float)

    if use_sequences:
        if bundle.seq is None:
            raise ValueError(f"Sequences not available for model {model_name}")
        train_mask = bundle.seq.indices < bundle.test_start
        test_mask = ~train_mask
        X_tr_all = bundle.seq.X[train_mask]
        y_tr_all = bundle.seq.y[train_mask]
        idx_tr_all = bundle.seq.indices[train_mask]
        meta_df = bundle.df.loc[idx_tr_all, ["timestamp"]].reset_index(drop=True)
        order = meta_df["timestamp"].argsort().to_numpy()
        meta_df_sorted = meta_df.iloc[order].reset_index(drop=True)
        X_tr_all = X_tr_all[order]
        y_tr_all = y_tr_all[order]
        idx_tr_all = idx_tr_all[order]
        ret_col = bundle.ret_col if bundle.ret_col in bundle.df.columns else None
        if cv_config:
            split_df = meta_df_sorted.assign(idx=idx_tr_all)
            splits = make_time_series_splits(split_df, "timestamp", cv_config)
            split_pairs = [(np.where(np.isin(idx_tr_all, s["train_idx"]))[0], np.where(np.isin(idx_tr_all, s["val_idx"]))[0]) for s in splits]
        else:
            split_pairs = list(time_folds(meta_df_sorted, n_folds=n_folds, embargo_minutes=embargo_minutes))
        for tr_idx, va_idx in split_pairs:
            model = ModelCls(model_config or {})
            val_tuple = (X_tr_all[va_idx], y_tr_all[va_idx])
            fit_kwargs = {"val_data": val_tuple}
            if ret_col:
                fit_kwargs["val_returns"] = bundle.df.loc[idx_tr_all[va_idx], ret_col].to_numpy()
            model.fit(X_tr_all[tr_idx], y_tr_all[tr_idx], **fit_kwargs)
            preds = model.predict_proba(X_tr_all[va_idx])
            oof.loc[idx_tr_all[va_idx]] = preds
        final_model = ModelCls(model_config or {})
        final_model.fit(X_tr_all, y_tr_all)
        test_indices = bundle.seq.indices[test_mask]
        test_order = np.argsort(test_indices)
        test_pred = pd.Series(
            final_model.predict_proba(bundle.seq.X[test_mask][test_order]),
            index=test_indices[test_order],
        )
    else:
        X_all = df_train[bundle.feature_cols]
        y_all = df_train[bundle.label_col].astype(int)
        ret_col = bundle.ret_col if bundle.ret_col in bundle.df.columns else None
        if cv_config:
            splits = make_time_series_splits(df_train, "timestamp", cv_config)
            split_pairs = [(s["train_idx"], s["val_idx"]) for s in splits]
        else:
            split_pairs = list(time_folds(df_train, n_folds=n_folds, embargo_minutes=embargo_minutes))
        for tr_idx, va_idx in split_pairs:
            model = ModelCls(model_config or {})
            val_tuple = (X_all.iloc[va_idx], y_all.iloc[va_idx])
            fit_kwargs = {"val_data": val_tuple}
            if ret_col:
                fit_kwargs["val_returns"] = df_train.iloc[va_idx][ret_col].to_numpy()
            model.fit(X_all.iloc[tr_idx], y_all.iloc[tr_idx], **fit_kwargs)
            preds = model.predict_proba(X_all.iloc[va_idx])
            oof.iloc[va_idx] = preds
        final_model = ModelCls(model_config or {})
        final_model.fit(X_all, y_all)
        test_pred = pd.Series(
            final_model.predict_proba(df_test[bundle.feature_cols]),
            index=df_test.index,
        )

    return ModelRun(
        name=model_name,
        model=final_model,
        oof_preds=oof,
        test_preds=test_pred,
    )


def evaluate_predictions(
    bundle: DatasetBundle,
    prob_series: pd.Series,
    *,
    cost_bps: float = 5.0,
    spread_col: Optional[str] = None,
    long_only: bool = False,
    min_hold_bars: int = 1,
) -> Dict[str, Any]:
    prob_series = prob_series.dropna()
    df_eval = bundle.df.loc[prob_series.index].copy()
    df_eval = df_eval.assign(prob=prob_series)
    ret_series = df_eval[bundle.ret_col] if bundle.ret_col in df_eval.columns else pd.Series(0.0, index=df_eval.index)
    ret_series = ret_series.fillna(0.0)
    spread_series = None
    if spread_col and spread_col in df_eval.columns:
        spread_series = df_eval[spread_col].fillna(0.0)
    thr, thr_report = select_prob_threshold(
        ret_series,
        prob_series,
        cost_bps=cost_bps,
        spread_series=spread_series,
        min_hold_bars=min_hold_bars,
    )
    eq = equity_curve(
        ret_series,
        prob_series,
        threshold=thr,
        cost_bps=cost_bps,
        spread_series=spread_series,
        min_hold_bars=min_hold_bars,
        long_only=long_only,
    )
    stats = summary_stats(eq)
    y_true = df_eval[bundle.label_col].astype(int)
    hit_rate = float(
        np.mean(((prob_series >= 0.5) & (y_true == 1)) | ((prob_series < 0.5) & (y_true == 0)))
    ) if len(y_true) else 0.0
    regime_pnl = {}
    if bundle.regime_cols:
        tmp = df_eval.copy()
        tmp["pnl"] = eq["pnl"].values
        for col in bundle.regime_cols:
            regime_pnl[col] = {str(k): float(v) for k, v in tmp.groupby(col)["pnl"].sum().to_dict().items()}
        tmp["regime_id"] = compute_regime_id(tmp, bundle.regime_cols)
        regime_pnl["composite"] = {str(k): float(v) for k, v in tmp.groupby("regime_id")["pnl"].sum().to_dict().items()}
    stats.update({
        "threshold": thr,
        "hit_rate": hit_rate,
        "pnl_net": float(eq["pnl"].sum()),
        "regime_pnl": regime_pnl,
        "threshold_report": thr_report,
    })
    return stats


def append_leaderboard(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    if "regime_pnl" in payload and isinstance(payload["regime_pnl"], dict):
        payload["regime_pnl_json"] = json.dumps(payload.pop("regime_pnl"))
    df_new = pd.DataFrame([payload])
    if path.exists():
        df_old = pd.read_csv(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)
