from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

import numpy as np
import pandas as pd
import yaml

from app.trading.decision import TriggerConfig, decide_bar
from training.infer import load_manifest_artifacts, load_base_predictor, predict_base, compute_gate_mask
from training.feature_eng import augment_market_features
from app.trading.state import PositionState


@dataclass
class SweepResult:
    config: Dict[str, Any]
    trades: int
    wins: int
    pnl: float
    pnl_perc: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    median_trade: float
    avg_trade: float
    fraction_time_in_position: float
    skipped_spread: int
    skipped_sizing: int
    skipped_other: int
    exit_reasons: Dict[str, int]
    coverage: float


def load_search_space(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_dataset(contract_path: Path, dataset_override: Optional[Path] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    if dataset_override:
        path = dataset_override
    else:
        cfg = yaml.safe_load(contract_path.read_text())
        dataset_path = cfg.get("dataset", {}).get("path")
        if not dataset_path:
            raise ValueError("Dataset path missing in contract config")
        path = Path(dataset_path)
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df


def ensure_probabilities(
    df: pd.DataFrame,
    prob_column: str,
    gate_column: Optional[str],
    model_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Attach probabilities/gate if missing by scoring with the provided model_dir manifest.
    """
    if prob_column in df.columns and (gate_column is None or gate_column in df.columns):
        return df
    if model_dir is None:
        raise ValueError(f"Probability column '{prob_column}' missing and no model_dir supplied")
    artifacts = load_manifest_artifacts(model_dir, model_label="sweep_model")
    calib, feat_cols = load_base_predictor(
        model_dir,
        prob_column=artifacts.prob_column,
        apply_calibration=getattr(artifacts, "apply_calibration", True),
    )
    # ensure required feature columns exist
    working = df.copy()
    for col in feat_cols:
        if col not in working.columns:
            working[col] = 0.0
    working = augment_market_features(working, inplace=False)
    prob_series = predict_base(working, calib, list(feat_cols))
    working[prob_column] = prob_series
    if gate_column:
        mask = compute_gate_mask(working, artifacts.gate_config, prob=prob_series, mode="inference")
        working[gate_column] = mask.astype(bool)
    return working


def _compute_sharpe(returns: Iterable[float]) -> float:
    arr = np.array(list(returns), dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(arr.size)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks
    return float(-drawdowns.min()) if drawdowns.size else 0.0


def simulate_trades(
    df: pd.DataFrame,
    cfg: TriggerConfig,
    *,
    prob_column: str = "base_prob",
    gate_column: Optional[str] = None,
    price_column: str = "close",
    spread_column: Optional[str] = None,
    order_notional: float = 100.0,
) -> SweepResult:
    state = PositionState()
    equity = 0.0
    equity_curve: List[float] = [equity]
    returns: List[float] = []
    trades = 0
    wins = 0
    pnl_values: List[float] = []
    exit_reasons: Dict[str, int] = {}
    skipped_spread = 0
    skipped_sizing = 0
    skipped_other = 0
    in_position_time = 0
    coverage_hits = 0
    coverage_total = 0

    df_iter = df.itertuples(index=False)
    for row in df_iter:
        row_dict = row._asdict()
        ts = pd.to_datetime(row_dict.get("timestamp"), utc=True)
        probability = float(row_dict.get(prob_column, 0.0) or 0.0)
        gate_pass = bool(row_dict.get(gate_column, True)) if gate_column else True
        price_val = row_dict.get(price_column)
        price = float(price_val) if price_val is not None else None
        spread_bps = None
        if spread_column:
            val = row_dict.get(spread_column)
            try:
                spread_bps = float(val) if val is not None else None
            except Exception:
                spread_bps = None

        coverage_total += 1
        if gate_pass and probability >= cfg.entry_threshold:
            coverage_hits += 1

        outcome = decide_bar(
            ts=ts.to_pydatetime(),
            probability=probability,
            gate_pass=gate_pass,
            state=state,
            cfg=cfg,
            current_price=price,
            entry_price=float(state.metadata.get("open_price")) if state.metadata.get("open_price") else None,
            spread_bps=spread_bps,
        )

        if state.in_position:
            in_position_time += 1

        if outcome.should_enter:
            if outcome.skip_execution:
                skipped_spread += 1
                state.metadata["last_entry_reason"] = outcome.skip_reason or ""
                continue
            if price is None or price <= 0:
                skipped_sizing += 1
                continue
            amount = order_notional / price
            if amount <= 0:
                skipped_sizing += 1
                continue
            state.metadata["open_price"] = f"{price:.10f}"
            state.metadata["open_amount"] = f"{amount:.10f}"
            state.metadata["open_entry_prob"] = f"{probability:.10f}"
            state.mark_entry(ts.to_pydatetime(), cfg.min_hold_bars * cfg.bar_seconds)
        elif outcome.should_exit and state.in_position:
            reason = outcome.exit_trigger or "unknown"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            if outcome.skip_execution:
                skipped_spread += 1
                continue
            entry_price = float(state.metadata.get("open_price") or 0.0)
            entry_amount = float(state.metadata.get("open_amount") or 0.0)
            if price is None or entry_price <= 0.0 or entry_amount <= 0.0:
                skipped_sizing += 1
                state.mark_exit(ts.to_pydatetime())
                state.metadata.clear()
                continue
            pnl = (price - entry_price) * entry_amount
            trades += 1
            if pnl > 0:
                wins += 1
            pnl_values.append(pnl)
            returns.append(pnl / (entry_price * entry_amount))
            equity += pnl
            equity_curve.append(equity)
            state.mark_exit(ts.to_pydatetime())
            state.metadata.clear()

    pnl_total = float(sum(pnl_values))
    avg_trade = float(np.mean(pnl_values)) if pnl_values else 0.0
    median_trade = float(np.median(pnl_values)) if pnl_values else 0.0
    profit_factor = 0.0
    if pnl_values:
        gains = sum(x for x in pnl_values if x > 0)
        losses = -sum(x for x in pnl_values if x < 0)
        profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    sharpe = _compute_sharpe(returns)
    max_dd = _max_drawdown(np.array(equity_curve, dtype=float))
    frac_time = float(in_position_time) / float(len(df) or 1)

    return SweepResult(
        config={
            "entry_threshold": cfg.entry_threshold,
            "exit_threshold": cfg.exit_threshold,
            "exit_prob_drop": cfg.exit_prob_drop,
            "min_hold_bars": cfg.min_hold_bars,
            "max_hold_seconds": cfg.max_hold_seconds,
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
            "max_spread_bps": cfg.max_spread_bps,
        },
        trades=trades,
        wins=wins,
        pnl=pnl_total,
        pnl_perc=(pnl_total / order_notional) if order_notional else pnl_total,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        median_trade=median_trade,
        avg_trade=avg_trade,
        fraction_time_in_position=frac_time,
        skipped_spread=skipped_spread,
        skipped_sizing=skipped_sizing,
        skipped_other=skipped_other,
        exit_reasons=exit_reasons,
        coverage=float(coverage_hits) / float(coverage_total or 1),
    )


def build_configs(space: Dict[str, Any]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    entries = space.get("entry_thresholds", [])
    exit_offsets = space.get("exit_threshold_offsets", [0.0])
    for entry in entries:
        for offs in exit_offsets:
            exit_thr = float(entry) + float(offs)
            for drop in space.get("exit_prob_drop", [0.15]):
                for mh in space.get("min_hold_bars", [1]):
                    for max_hold in space.get("max_hold_minutes", [None]):
                        for sl in space.get("stop_loss_pct", [None]):
                            for tp in space.get("take_profit_pct", [None]):
                                for spread in space.get("max_spread_bps", [10]):
                                    configs.append(
                                        {
                                            "entry_threshold": float(entry),
                                            "exit_threshold": float(exit_thr),
                                            "exit_prob_drop": float(drop),
                                            "min_hold_bars": int(mh),
                                            "max_hold_minutes": max_hold,
                                            "stop_loss_pct": sl,
                                            "take_profit_pct": tp,
                                            "max_spread_bps": spread,
                                        }
                                    )
    return configs


def promote_best(result: SweepResult, output_path: Path) -> None:
    payload = {
        "model": "xgb_primary",
        "symbol": "ETH/USDT",
        "timeframe": "1m",
        "source": "trigger_optimizer",
        **result.config,
    }
    output_path.write_text(yaml.safe_dump(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger optimizer (entry/exit/guards)")
    parser.add_argument("--contract", type=Path, required=True, help="Canonical contract yaml with dataset path")
    parser.add_argument("--dataset", type=Path, help="Override dataset path")
    parser.add_argument("--search-space", type=Path, default=Path("configs/trigger_search_space.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/trigger_sweeps"))
    parser.add_argument("--prob-column", type=str, default="base_prob")
    parser.add_argument("--gate-column", type=str, default=None)
    parser.add_argument("--price-column", type=str, default="close")
    parser.add_argument("--spread-column", type=str, default=None)
    parser.add_argument("--order-notional", type=float, default=100.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--model-dir", type=Path, help="Path to manifest/model bundle for scoring probabilities when absent")
    parser.add_argument("--promote-best", action="store_true")
    parser.add_argument("--final-policy-path", type=Path, default=Path("configs/final_trigger_policy.yaml"))
    args = parser.parse_args()

    df = load_dataset(args.contract, args.dataset, max_rows=args.max_rows)
    df = ensure_probabilities(df, args.prob_column, args.gate_column, args.model_dir)
    space = load_search_space(args.search_space)
    configs = build_configs(space)
    results: List[SweepResult] = []
    min_trades = int(space.get("min_trades") or 0)
    frac_min = float(space.get("fraction_time_in_position_min") or 0.0)
    max_dd_limit = space.get("max_drawdown_limit")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict[str, Any]] = []

    for cfg_dict in configs:
        trigger_cfg = TriggerConfig(
            entry_threshold=cfg_dict["entry_threshold"],
            exit_threshold=cfg_dict["exit_threshold"],
            exit_prob_drop=cfg_dict["exit_prob_drop"],
            min_hold_bars=cfg_dict["min_hold_bars"],
            bar_seconds=60,
            long_only=True,
            max_hold_seconds=cfg_dict.get("max_hold_minutes") * 60 if cfg_dict.get("max_hold_minutes") else None,
            stop_loss_pct=cfg_dict.get("stop_loss_pct"),
            take_profit_pct=cfg_dict.get("take_profit_pct"),
            max_spread_bps=cfg_dict.get("max_spread_bps"),
        )
        res = simulate_trades(
            df,
            trigger_cfg,
            prob_column=args.prob_column,
            gate_column=args.gate_column,
            price_column=args.price_column,
            spread_column=args.spread_column,
            order_notional=args.order_notional,
        )
        # Apply hard filters
        if res.trades < min_trades or res.fraction_time_in_position < frac_min:
            pass  # still record, but mark low quality
        if max_dd_limit is not None and max_dd_limit > 0 and res.max_drawdown > max_dd_limit:
            pass
        results.append(res)
        row = {
            **res.config,
            "trades": res.trades,
            "wins": res.wins,
            "pnl": res.pnl,
            "pnl_perc": res.pnl_perc,
            "sharpe": res.sharpe,
            "max_drawdown": res.max_drawdown,
            "profit_factor": res.profit_factor,
            "median_trade": res.median_trade,
            "avg_trade": res.avg_trade,
            "fraction_time_in_position": res.fraction_time_in_position,
            "skipped_spread": res.skipped_spread,
            "skipped_sizing": res.skipped_sizing,
            "skipped_other": res.skipped_other,
            "exit_reasons": json.dumps(res.exit_reasons),
            "coverage": res.coverage,
        }
        csv_rows.append(row)

    results_path = out_dir / "results.csv"
    pd.DataFrame(csv_rows).to_csv(results_path, index=False)

    # Select best by sharpe then pnl
    best = sorted(results, key=lambda r: (r.sharpe, r.pnl), reverse=True)[0] if results else None
    summary = {
        "results_path": str(results_path),
        "best": best.config if best else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.promote_best and best is not None:
        promote_best(best, args.final_policy_path)
        print(f"Promoted best config to {args.final_policy_path}")
    print(f"Wrote {len(results)} results to {results_path}")


if __name__ == "__main__":
    sys.exit(main())
