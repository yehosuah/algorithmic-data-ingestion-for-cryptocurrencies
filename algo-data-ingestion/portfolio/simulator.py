from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from training.infer import compute_gate_mask, DEFAULT_GATE_CONFIG

from .ensemble import combine_model_signals
from .gating import apply_thresholds_to_probs
from .metrics import compute_portfolio_metrics


def _load_gate_config(risk_config: Mapping[str, object]) -> Dict[str, object]:
    """
    Fetch a gate configuration compatible with training.infer.compute_gate_mask.
    """
    cfg = risk_config.get("gate_config")
    if isinstance(cfg, Mapping):
        return dict(cfg)
    path = risk_config.get("gate_config_path")
    if path:
        p = Path(str(path)).expanduser()
        if p.exists():
            text = p.read_text()
            try:
                if p.suffix.lower() in {".yaml", ".yml"}:
                    return yaml.safe_load(text) or {}
                return json.loads(text)
            except Exception:
                pass
    return dict(DEFAULT_GATE_CONFIG)


def _resolve_label_column(df: pd.DataFrame) -> str:
    label_col = df.attrs.get("label_col")
    if label_col and label_col in df.columns:
        return str(label_col)
    for candidate in ("cost_adjusted_15m", "ret_next", "net_return_15m"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No suitable label/return column found in dataframe")


def run_portfolio_simulation(
    df: pd.DataFrame,
    model_signals: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, float]],
    risk_config: dict[str, object],
    sampling_policy: str | None = None,
    weight_policy: str | None = None,
) -> dict[str, object]:
    """
    Map model probabilities into constrained portfolio trades and compute metrics.

    - Respects manifest-style gates via compute_gate_mask.
    - Applies dual thresholds + optional regime overrides to probability streams.
    - Aggregates signals with portfolio-layer weights and clamps exposures using
      leverage/turnover caps from `risk_config`.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty; cannot simulate portfolio.")
    if not model_signals:
        raise ValueError("model_signals is empty; provide at least one model stream")
    n = len(df)
    min_len = min([len(df), *[len(arr) for arr in model_signals.values()]])
    if min_len <= 0:
        raise ValueError("No usable rows for simulation after alignment.")
    if min_len != n:
        df = df.tail(min_len).reset_index(drop=True)
        model_signals = {k: v[-min_len:] for k, v in model_signals.items()}
        n = min_len

    risk_cfg = risk_config or {}
    capital = float(risk_cfg.get("capital", 1_000_000))
    max_gross = float(risk_cfg.get("max_gross_leverage", 1.0))
    max_net = float(risk_cfg.get("max_net_exposure", 1.0))
    max_symbol_weight = float(risk_cfg.get("max_symbol_weight", 1.0))
    max_turnover_per_day = float(risk_cfg.get("max_turnover_per_day", 5.0))
    transaction_cost_bps = float(risk_cfg.get("transaction_cost_bps", 0.0))
    slippage_bps = float(risk_cfg.get("slippage_bps", 0.0))
    spread_scale = float(risk_cfg.get("spread_scale", 0.0))
    spread_col = str(risk_cfg.get("spread_column", "hl_spread"))
    gate_mode = str(risk_cfg.get("gate_mode", "inference"))
    long_only = bool(risk_cfg.get("long_only", False))
    model_weights = risk_cfg.get("model_weights") or {}
    ensemble_mode = risk_cfg.get("ensemble_mode", "weighted_sum")
    regime_col = None
    regimes = df.attrs.get("regime_cols") or []
    if regimes:
        regime_col = regimes[0]
    if risk_cfg.get("regime_col"):
        regime_col = risk_cfg.get("regime_col")

    gate_cfg = _load_gate_config(risk_cfg)

    df_sorted = df.sort_values("timestamp").reset_index(drop=False).rename(columns={"index": "row"})
    label_col = _resolve_label_column(df_sorted)

    gate_masks: Dict[str, pd.Series] = {}
    per_model_positions: Dict[str, np.ndarray] = {}
    for model_name, probs in model_signals.items():
        prob_series = pd.Series(probs, index=df.index, name=model_name)
        gate_mask = compute_gate_mask(df, gate_cfg, prob=prob_series, mode=gate_mode)
        gate_masks[model_name] = gate_mask
        thr_cfg = thresholds.get(model_name, thresholds.get("default", {})) or {}
        signals = apply_thresholds_to_probs(
            prob_series.to_numpy(),
            df,
            thr_cfg,
            regime_col=regime_col,
            gate_mask=gate_mask,
        )
        per_model_positions[model_name] = signals

    aggregated = combine_model_signals(per_model_positions, model_weights, mode=ensemble_mode)
    if long_only:
        aggregated = np.clip(aggregated, 0.0, None)
    aggregated = np.clip(aggregated, -1.0, 1.0)

    returns: list[float] = []
    records: list[dict[str, object]] = []
    prev_pos: Dict[str, float] = defaultdict(float)
    # track turnover per calendar day (date)
    turnover_used: Dict[pd.Timestamp, float] = defaultdict(float)

    grouped = df_sorted.groupby("timestamp", sort=True)
    for ts, group in grouped:
        idx_list = group["row"].tolist()
        target_by_symbol: Dict[str, float] = {}
        # Desired targets for this timestamp
        for _, row in group.iterrows():
            symbol = row.get("symbol")
            if symbol is None:
                continue
            target = float(aggregated[row["row"]])
            if long_only and target < 0:
                target = 0.0
            target = float(np.clip(target, -max_symbol_weight, max_symbol_weight))
            target_by_symbol[symbol] = target

        gross = sum(abs(v) for v in target_by_symbol.values())
        net = sum(target_by_symbol.values())
        scale = 1.0
        if gross > max_gross and gross > 0:
            scale = min(scale, max_gross / gross)
        if abs(net) > max_net and abs(net) > 0:
            scale = min(scale, max_net / abs(net))
        if scale != 1.0:
            for sym in list(target_by_symbol.keys()):
                target_by_symbol[sym] *= scale

        # Turnover guardrail (per UTC day)
        day_key = pd.Timestamp(ts).normalize()
        remaining_turnover = max_turnover_per_day * capital - turnover_used[day_key]
        if remaining_turnover < 0:
            remaining_turnover = 0.0

        desired_changes = {sym: target_by_symbol[sym] - prev_pos.get(sym, 0.0) for sym in target_by_symbol}
        turnover_this_bar = sum(abs(delta) for delta in desired_changes.values()) * capital

        turnover_scale = 1.0
        if turnover_this_bar > 0 and turnover_this_bar > remaining_turnover and remaining_turnover >= 0:
            turnover_scale = max(0.0, remaining_turnover / turnover_this_bar)

        pnl_this_ts = 0.0
        for _, row in group.iterrows():
            symbol = row["symbol"]
            prev = prev_pos.get(symbol, 0.0)
            delta = desired_changes.get(symbol, 0.0) * turnover_scale
            target = prev + delta
            turnover_abs = abs(delta)
            turnover_used[day_key] += turnover_abs * capital

            spread_val = row.get(spread_col, 0.0)
            cost = ((transaction_cost_bps + slippage_bps) / 1e4) * turnover_abs
            if spread_scale and spread_val is not None:
                try:
                    cost += spread_scale * float(spread_val) * turnover_abs
                except Exception:
                    pass

            ret_val_raw = row[label_col]
            try:
                ret_val = float(ret_val_raw)
            except Exception:
                ret_val = 0.0
            if pd.isna(ret_val):
                ret_val = 0.0
            pnl = prev * ret_val - cost
            record = {
                "row": int(row["row"]),
                "timestamp": ts,
                "symbol": symbol,
                "position": target,
                "prev_position": prev,
                "turnover": turnover_abs,
                "pnl": pnl,
                "gross_exposure": abs(target),
                "net_exposure": target,
            }
            records.append(record)
            prev_pos[symbol] = target
            pnl_this_ts += pnl

        returns.append(pnl_this_ts)

    positions_df = pd.DataFrame.from_records(records)
    regimes = df[regime_col] if regime_col and regime_col in df.columns else None
    metrics = compute_portfolio_metrics(np.array(returns, dtype=float), positions_df, df, regimes=regimes)
    metrics.update(
        {
            "sampling_policy": sampling_policy,
            "weight_policy": weight_policy,
            "model_weights": model_weights,
            "ensemble_mode": ensemble_mode,
        }
    )
    return {
        "metrics": metrics,
        "returns": np.array(returns, dtype=float),
        "positions": positions_df,
    }
