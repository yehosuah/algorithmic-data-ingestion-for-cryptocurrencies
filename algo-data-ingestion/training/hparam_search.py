from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Ensure model registry is populated
from . import blender  # noqa: F401
from . import model as base_models  # noqa: F401
from . import regime_blender  # noqa: F401
from . import stacking_meta_model  # noqa: F401
from . import tcn_model  # noqa: F401
from . import transformer_model  # noqa: F401
from . import deeplob_model  # noqa: F401
from .data import load_canonical_contract, load_training_dataset
from .model_registry import get_model_class
from .sequence_builder import build_sequences
from .thresholds import select_prob_threshold
from .metrics import equity_curve, summary_stats
from .time_series_cv import TimeSeriesSplitConfig, evaluate_by_regime, make_time_series_splits

SEQ_MODELS = {"tcn", "transformer", "deeplob"}


def _json_safe(obj: Any) -> Any:
    """Recursively convert keys/values to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _resolve_columns(df: pd.DataFrame, contract: Dict[str, Any]) -> tuple[list[str], str, list[str], str]:
    feature_cols = df.attrs.get("feature_cols") or contract.get("features", {}).get("core", [])
    label_col = df.attrs.get("label_col") or contract.get("labels", {}).get("primary")
    regime_cols = df.attrs.get("regime_cols") or []
    ret_col = "net_return_15m" if "net_return_15m" in df.columns else "ret_next"
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in {"timestamp", "symbol", "y_dir", "dt", "exchange", "timeframe", "feature_version", "close", "ret_next"}]
    if label_col is None:
        label_col = "y_dir"
    return feature_cols, label_col, regime_cols, ret_col


def _evaluate_split(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    prob: np.ndarray,
    *,
    label_col: str,
    ret_col: str,
    regime_cols: Sequence[str],
    cost_bps: float,
    long_only: bool,
    min_hold_bars: int,
) -> Dict[str, Any]:
    subset = df.loc[val_idx]
    prob_series = pd.Series(prob, index=subset.index)
    y_true = subset[label_col].astype(int)
    ret_series = subset[ret_col] if ret_col in subset.columns else pd.Series(0.0, index=subset.index)
    thr, thr_report = select_prob_threshold(
        ret_series,
        prob_series,
        cost_bps=cost_bps,
        long_only=long_only,
        min_hold_bars=min_hold_bars,
    )
    eq = equity_curve(
        ret_series,
        prob_series,
        threshold=thr,
        cost_bps=cost_bps,
        long_only=long_only,
        min_hold_bars=min_hold_bars,
    )
    stats = summary_stats(eq)
    hit_rate = float(
        np.mean(((prob_series >= 0.5) & (y_true == 1)) | ((prob_series < 0.5) & (y_true == 0)))
    ) if len(y_true) else 0.0
    regime_pnl = {}
    regime_metrics = {}
    if regime_cols:
        tmp = subset.copy()
        tmp["pnl"] = eq["pnl"].values
        # Skip high-cardinality regimes to avoid excessive compute
        filtered_regimes = [c for c in regime_cols if c in tmp.columns and tmp[c].nunique(dropna=True) <= 50]
        for col in filtered_regimes:
            regime_pnl[col] = tmp.groupby(col)["pnl"].sum().to_dict()
            try:
                regime_metrics[col] = evaluate_by_regime(
                    tmp,
                    y_true=y_true.to_numpy(),
                    y_proba=prob_series.to_numpy(),
                    regime_col=col,
                    ret_col=ret_col,
                    cost_bps=cost_bps,
                    long_only=long_only,
                    min_hold_bars=min_hold_bars,
                )
            except Exception:
                # Keep robustness; regime analysis is diagnostic
                pass
    return {
        "pnl_net": float(eq["pnl"].sum()),
        "sharpe": stats.get("sharpe"),
        "hit_rate": hit_rate,
        "threshold": thr,
        "threshold_report": thr_report,
        "regime_pnl": regime_pnl,
        "regime_metrics": regime_metrics,
        "final_equity": stats.get("final_equity"),
        "avg_turnover": stats.get("avg_turnover"),
        "total_turnover": stats.get("total_turnover"),
    }


def objective_single_model(
    model_name: str,
    hparams: Dict[str, Any],
    contract_path: str,
    cv_config: TimeSeriesSplitConfig,
    *,
    preloaded_df: Optional[pd.DataFrame] = None,
    seq_len: int = 32,
    horizon: int = 1,
    seq_stride: int = 1,
    max_rows: Optional[int] = None,
    cost_bps: float = 5.0,
    long_only: bool = False,
    min_hold_bars: int = 1,
) -> Dict[str, Any]:
    """
    - Load canonical dataset using contract_path.
    - Build time-series splits.
    - For each CV split:
      * Train model with given hparams on train.
      * Evaluate on val using P&L net, Sharpe, hit-rate, etc.
    - Aggregate CV metrics (mean P&L, mean Sharpe).
    - Return a dict with metrics and any artifacts needed.
    """
    contract = load_canonical_contract(contract_path)
    df = preloaded_df if preloaded_df is not None else load_training_dataset(contract)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[-int(max_rows) :]
    df = df.sort_values(["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]).reset_index(drop=True)
    feature_cols, label_col, regime_cols, ret_col = _resolve_columns(df, contract)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    splits = make_time_series_splits(df, "timestamp", cv_config)
    ModelCls = get_model_class(model_name)
    params_for_model = {k: v for k, v in hparams.items() if k not in {"seq_len", "horizon"}}
    fold_metrics: List[Dict[str, Any]] = []

    if model_name in SEQ_MODELS:
        seq_len_use = int(hparams.get("seq_len", seq_len))
        horizon_use = int(hparams.get("horizon", horizon))
        stride_use = int(hparams.get("seq_stride", seq_stride))
        X_seq, y_seq, idx_seq = build_sequences(
            df,
            feature_cols,
            label_col,
            seq_len=seq_len_use,
            horizon=horizon_use,
            stride=stride_use,
            return_index=True,
        )
        for split in splits:
            train_mask = np.isin(idx_seq, split["train_idx"])
            val_mask = np.isin(idx_seq, split["val_idx"])
            if not val_mask.any() or not train_mask.any():
                continue
            X_train, y_train = X_seq[train_mask], y_seq[train_mask]
            X_val, y_val = X_seq[val_mask], y_seq[val_mask]
            mdl = ModelCls(params_for_model)
            val_returns = df.loc[idx_seq[val_mask], ret_col].to_numpy() if ret_col in df.columns else None
            mdl.fit(X_train, y_train, val_data=(X_val, y_val), val_returns=val_returns)
            preds = mdl.predict_proba(X_val)
            metrics = _evaluate_split(
                df,
                idx_seq[val_mask],
                preds,
                label_col=label_col,
                ret_col=ret_col,
                regime_cols=regime_cols,
                cost_bps=cost_bps,
                long_only=long_only,
                min_hold_bars=min_hold_bars,
            )
            fold_metrics.append(metrics)
    else:
        for split in splits:
            train_idx = split["train_idx"]
            val_idx = split["val_idx"]
            X_train = df.loc[train_idx, feature_cols]
            y_train = df.loc[train_idx, label_col].astype(int)
            X_val = df.loc[val_idx, feature_cols]
            y_val = df.loc[val_idx, label_col].astype(int)
            mdl = ModelCls(params_for_model)
            fit_kwargs: Dict[str, Any] = {"val_data": (X_val, y_val)}
            if ret_col in df.columns:
                fit_kwargs["val_returns"] = df.loc[val_idx, ret_col].to_numpy()
            mdl.fit(X_train, y_train, **fit_kwargs)
            preds = mdl.predict_proba(X_val)
            metrics = _evaluate_split(
                df,
                val_idx,
                preds,
                label_col=label_col,
                ret_col=ret_col,
                regime_cols=regime_cols,
                cost_bps=cost_bps,
                long_only=long_only,
                min_hold_bars=min_hold_bars,
            )
            fold_metrics.append(metrics)

    if not fold_metrics:
        return {"model": model_name, "hparams": hparams, "mean_pnl_net_cv": -np.inf, "mean_sharpe_cv": -np.inf}

    pnl_vals = [m.get("pnl_net", 0.0) for m in fold_metrics]
    sharpe_vals = [m.get("sharpe", 0.0) for m in fold_metrics]
    hit_rates = [m.get("hit_rate", 0.0) for m in fold_metrics]
    mean_pnl = float(np.mean(pnl_vals))
    mean_sharpe = float(np.mean(sharpe_vals))
    mean_hit = float(np.mean(hit_rates))
    regime_vars = []
    for fm in fold_metrics:
        reg = fm.get("regime_pnl") or {}
        for _, pnl_map in reg.items():
            vals = list(pnl_map.values())
            if len(vals) > 1:
                regime_vars.append(float(np.var(vals)))
    regime_var = float(np.mean(regime_vars)) if regime_vars else 0.0
    return {
        "model": model_name,
        "hparams": hparams,
        "fold_metrics": fold_metrics,
        "mean_pnl_net_cv": mean_pnl,
        "mean_sharpe_cv": mean_sharpe,
        "mean_hit_rate_cv": mean_hit,
        "regime_pnl_variance_cv": regime_var,
    }


def random_search(
    model_name: str,
    search_space: Dict[str, List[Any]],
    n_trials: int,
    contract_path: str,
    cv_config: TimeSeriesSplitConfig,
    *,
    output_dir: str,
    seq_len: int = 32,
    horizon: int = 1,
    seq_stride: int = 1,
    max_rows: Optional[int] = None,
    cost_bps: float = 5.0,
    long_only: bool = False,
    min_hold_bars: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    start_idx = 0
    existing_path = out_dir / "results.csv"
    if existing_path.exists():
        try:
            df_existing = pd.read_csv(existing_path)
            records = df_existing.to_dict(orient="records")
            results_count = len(records)
            start_idx = results_count
        except Exception:
            records = []
            start_idx = 0
    # Load dataset once per randomized search run to avoid repeated parquet reads
    contract = load_canonical_contract(contract_path)
    df_full = load_training_dataset(contract)
    for i in range(start_idx, start_idx + n_trials):
        sampled = {k: rng.choice(v) for k, v in search_space.items()}
        # Allow search space to tune trading/eval knobs too
        cost_bps_use = float(sampled.pop("cost_bps", cost_bps))
        long_only_use = bool(sampled.pop("long_only", long_only))
        min_hold_use = int(sampled.pop("min_hold_bars", min_hold_bars))
        seq_stride_use = int(sampled.get("seq_stride", seq_stride))
        trial_id = f"{model_name}_trial_{i:03d}"
        res = objective_single_model(
            model_name,
            sampled,
            contract_path,
            cv_config,
            preloaded_df=df_full,
            seq_len=seq_len,
            horizon=horizon,
            seq_stride=seq_stride_use,
            max_rows=max_rows,
            cost_bps=cost_bps_use,
            long_only=long_only_use,
            min_hold_bars=min_hold_use,
        )
        res["trial_id"] = trial_id
        results.append(res)
        record = {
            "trial_id": trial_id,
            "model": model_name,
            "mean_pnl_net_cv": res.get("mean_pnl_net_cv"),
            "mean_sharpe_cv": res.get("mean_sharpe_cv"),
            "mean_hit_rate_cv": res.get("mean_hit_rate_cv"),
            "regime_pnl_variance_cv": res.get("regime_pnl_variance_cv"),
            "cost_bps": cost_bps_use,
            "long_only": long_only_use,
            "min_hold_bars": min_hold_use,
        }
        for k, v in sampled.items():
            record[f"param_{k}"] = v
        records.append(record)
        safe_res = _json_safe(res)
        (out_dir / f"{trial_id}.json").write_text(json.dumps(safe_res, indent=2))
        pd.DataFrame(records).to_csv(out_dir / "results.csv", index=False)
    # Save summary sorted by Sharpe then PnL
    if records:
        df_rec = pd.DataFrame(records)
        df_rec.sort_values(["mean_sharpe_cv", "mean_pnl_net_cv"], ascending=False).to_csv(out_dir / "summary_sorted.csv", index=False)
    return results
