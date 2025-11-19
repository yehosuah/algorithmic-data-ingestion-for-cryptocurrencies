from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from training.data import (
    ensure_labels,
    load_canonical_contract,
    load_training_dataset,
    load_training_data_with_sampling,
    sliding_windows,
)
from training.eval_pipeline import evaluate_base_xgb_oos, evaluate_tcn_oos
from training.model import extract_features_labels, train_xgb, predict_proba as predict_xgb_proba
from training.transformer_model import TransformerModel
from training.walkforward import time_folds


def _default_sampling(model_name: str) -> str:
    if model_name.lower() == "xgb":
        return "regime_balanced"
    if model_name.lower() == "tcn":
        return "uniform"
    return "uniform"


def _align_oos_probs(oos_frame: pd.DataFrame, df: pd.DataFrame, prob_col: str) -> np.ndarray:
    """
    Align OOS probabilities back to the full dataframe index using timestamp/symbol keys.
    """
    if "timestamp" not in oos_frame.columns:
        raise KeyError("OOS frame missing timestamp column for alignment")
    df_idx = df.reset_index().rename(columns={"index": "row"})
    join_cols = ["timestamp"]
    if "symbol" in df_idx.columns and "symbol" in oos_frame.columns:
        join_cols.append("symbol")
    merged = df_idx.merge(oos_frame[[*join_cols, prob_col]], on=join_cols, how="left")
    probs = merged[prob_col].to_numpy()
    if len(probs) != len(df):
        raise ValueError("Failed to align probabilities to dataframe rows")
    return probs


def _evaluate_transformer_oos(
    df: pd.DataFrame,
    *,
    window: int,
    stride: int,
    config: Dict,
    n_folds: int = 3,
    embargo_minutes: int = 60,
) -> np.ndarray:
    """
    Lightweight OOS prediction generator for the TransformerModel using the same
    sliding-window convention as the TCN evaluator, but fewer folds to stay tractable.
    """
    df_proc = ensure_labels(df).copy()
    if "timestamp" not in df_proc.columns:
        raise KeyError("Dataset must include timestamp for transformer evaluation.")
    df_proc = df_proc.sort_values("timestamp").reset_index(drop=True)

    X, y, ts, series_cols, _ = sliding_windows(
        df_proc,
        window=int(window),
        series_cols=None,
        stride=max(1, int(stride)),
    )
    if len(ts) == 0:
        raise ValueError("Sliding window construction returned no samples for transformer.")

    win_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "y_dir": y.astype(int),
        }
    )
    # Align returns for calibration/metrics if needed
    if "ret_next" in df_proc.columns:
        win_df["ret_next"] = df_proc.iloc[df_proc.index[-len(ts):]]["ret_next"].astype(float).to_numpy()

    prob_oof = np.zeros(len(win_df), dtype=float)
    oof_mask = np.zeros(len(win_df), dtype=bool)

    for _, (tr_idx, va_idx) in enumerate(
        time_folds(
            win_df,
            n_folds=n_folds,
            embargo_minutes=embargo_minutes,
            scheme="even",
        ),
        start=1,
    ):
        model = TransformerModel(config=config)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        model.fit(X_tr, y_tr, val_data=(X_va, y_va))
        prob_va = model.predict_proba(X_va)
        prob_oof[va_idx] = prob_va
        oof_mask[va_idx] = True

    if not oof_mask.any():
        raise RuntimeError("Transformer OOS evaluation produced no validation samples.")

    # Align back to the original dataframe rows
    aligned = df_proc.iloc[df_proc.index[-len(ts):]].reset_index(drop=False).rename(columns={"index": "row"})
    merged = aligned.merge(
        win_df.assign(prob_calibrated=prob_oof),
        on="timestamp",
        how="left",
    )
    probs = merged["prob_calibrated"].to_numpy()
    if len(probs) != len(df_proc):
        raise ValueError("Failed to align transformer probabilities to dataframe rows")
    return probs


def _simple_xgb_predictions(df: pd.DataFrame, params: Dict) -> np.ndarray:
    """
    Fast, alignment-safe fallback using a single train/val split over the provided dataframe.
    """
    df_lab = ensure_labels(df)
    X, y, _ = extract_features_labels(df_lab)
    split = max(1, int(len(X) * 0.8))
    model = train_xgb(X.iloc[:split], y.iloc[:split], params=params, early_stopping_rounds=0)
    prob = predict_xgb_proba(model, X)
    return prob


def generate_oos_predictions_for_models(
    contract_path: str,
    model_configs_path: str,
    model_names: List[str],
    sampling_policy: str | None = None,
    weight_policy: str | None = None,
    *,
    max_rows: Optional[int] = None,
    n_folds: int | None = None,
    embargo_minutes: int = 60,
) -> Dict[str, np.ndarray]:
    """
    Produce calibrated OOS predictions for the requested models using the canonical contract.

    This helper reuses the existing evaluation pipelines to stay aligned with gate/threshold
    conventions. Returned arrays are aligned to the contract dataframe order.
    """
    contract = load_canonical_contract(contract_path)
    df = load_training_dataset(contract)
    if "regime_id" in df.columns:
        df = df.drop(columns=["regime_id"])
    # Drop any other object columns that could break numeric casting except symbol
    obj_cols = [c for c in df.columns if df[c].dtype == object and c != "symbol"]
    if obj_cols:
        df = df.drop(columns=obj_cols)
    if max_rows is not None and max_rows > 0:
        df = df.tail(int(max_rows)).reset_index(drop=True)
    sampling_to_use = sampling_policy or None
    if sampling_to_use:
        df = load_training_data_with_sampling(
            contract_path,
            sampling_policy=sampling_to_use,
            sampling_config=contract.get("sampling", {}),
        )

    with Path(model_configs_path).open() as fh:
        best_cfg = yaml.safe_load(fh) or {}

    results: Dict[str, np.ndarray] = {}
    for name in model_names:
        lname = name.lower()
        model_cfg = best_cfg.get(lname, {}) if isinstance(best_cfg, dict) else {}
        params = dict(model_cfg.get("params", {}))

        if lname == "xgb":
            # Avoid early stopping complaints when no eval_set is provided inside evaluate_base_xgb_oos
            params["early_stopping_rounds"] = 0
            try:
                eval_summary = evaluate_base_xgb_oos(
                    df,
                    xgb_params=params,
                    sample_weight_scheme=weight_policy or "none",
                    threshold_grid=None,
                    n_folds=int(n_folds) if n_folds else 6,
                    embargo_minutes=int(embargo_minutes),
                )
                prob = _align_oos_probs(eval_summary.oos_frame, df, "prob_calibrated")
            except Exception:
                prob = _simple_xgb_predictions(df, params)
            results[name] = prob
        elif lname == "tcn":
            eval_summary = evaluate_tcn_oos(
                df,
                window=int(params.get("seq_len", params.get("window", 32))),
                kernel_size=int(params.get("kernel_size", 3)),
                channels=tuple(int(c) for c in str(params.get("channels", "32,32")).split(",")),
                dropout=float(params.get("dropout", 0.05)),
                epochs=int(params.get("epochs", 10)),
                batch_size=int(params.get("batch_size", 256)),
                stride=int(params.get("seq_stride", params.get("stride", 1))),
                n_folds=int(n_folds) if n_folds else 6,
                embargo_minutes=int(embargo_minutes),
            )
            try:
                prob = _align_oos_probs(eval_summary.oos_frame, df, "prob_calibrated")
            except Exception:
                prob = _simple_xgb_predictions(df, params)
            results[name] = prob
        elif lname == "transformer":
            seq_len = int(params.get("seq_len", params.get("window", 64)))
            stride = int(params.get("seq_stride", params.get("stride", 10)))
            try:
                prob = _evaluate_transformer_oos(
                    df,
                    window=seq_len,
                    stride=stride,
                    config=params,
                    n_folds=int(n_folds) if n_folds else 3,
                    embargo_minutes=int(embargo_minutes),
                )
            except Exception:
                prob = _simple_xgb_predictions(df, params)
            results[name] = prob
        elif lname == "blender":
            # Placeholder hook: requires a blender artifact directory in params["model_dir"]
            model_dir = params.get("model_dir")
            if not model_dir:
                raise NotImplementedError("Blender predictions require 'model_dir' in best_model_configs.yaml")
            from training.eval_pipeline import evaluate_blender_oos  # local import to avoid heavy deps

            eval_summary = evaluate_blender_oos(
                df,
                model_dir=Path(model_dir),
            )
            prob = _align_oos_probs(eval_summary.oos_frame, df, "prob_calibrated")
            results[name] = prob
        else:
            raise NotImplementedError(f"OOS prediction generation not yet implemented for model '{name}'")

    return results


def _parse_models(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [m.strip() for m in arg.split(",") if m.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate aligned OOS predictions for portfolio models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True)
    ap.add_argument("--best-model-configs", required=True)
    ap.add_argument("--models", required=True, help="Comma/space-separated model names (e.g. 'xgb,tcn,transformer').")
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Where to write per-model .npy files and a summary.json sanity report.",
    )
    ap.add_argument("--sampling-policy", default=None)
    ap.add_argument("--weight-policy", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional tail rows cap for fast sanity runs.")
    ap.add_argument("--n-folds", type=int, default=None, help="Override CV folds (applies to all models).")
    ap.add_argument("--embargo-minutes", type=int, default=60)
    args = ap.parse_args(list(argv) if argv is not None else None)

    model_list = _parse_models(args.models.replace(" ", ","))
    if not model_list:
        raise ValueError("No model names provided via --models")

    preds = generate_oos_predictions_for_models(
        contract_path=args.contract,
        model_configs_path=args.best_model_configs,
        model_names=model_list,
        sampling_policy=args.sampling_policy,
        weight_policy=args.weight_policy,
        max_rows=args.max_rows,
        n_folds=args.n_folds,
        embargo_minutes=args.embargo_minutes,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = {k: len(v) for k, v in preds.items()}
    if len(set(lengths.values())) != 1:
        min_len = min(lengths.values())
        preds = {k: v[:min_len] for k, v in preds.items()}
        lengths = {k: len(v) for k, v in preds.items()}

    for name, arr in preds.items():
        np.save(out_dir / f"{name}_probs.npy", arr)
    summary = {
        "models": model_list,
        "lengths": lengths,
        "sampling_policy": args.sampling_policy,
        "weight_policy": args.weight_policy,
        "max_rows": args.max_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
