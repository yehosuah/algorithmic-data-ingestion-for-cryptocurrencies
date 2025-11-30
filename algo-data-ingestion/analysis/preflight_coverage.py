from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from analysis.validate_deployment_contract import validate_deployment_contract
from portfolio.gating import apply_thresholds_to_probs
from training.data import load_canonical_contract, load_training_dataset
from training.infer import (
    compute_gate_mask,
    load_base_predictor,
    load_manifest_artifacts,
    load_tcn_predictor,
    predict_base,
    predict_tcn,
)
from training.transformer_model import TransformerModel


DEFAULT_FEATURE_CANDIDATES = (
    Path("data_lake/trading/live_features.parquet"),
    Path("data_lake/trading/latest_features.parquet"),
    Path("experiments/dummy_live_features.parquet"),
)


@dataclass
class SymbolCoverage:
    symbol: str
    model: str
    p50: float
    p90: float
    p95: float
    p99: float
    fraction_above_prob_gate_min: float
    fraction_between_entry_exit: float
    implied_trade_proxy: int
    coverage_ratio: float
    samples: int
    prob_gate_min: Optional[float]
    entry_threshold: Optional[float]
    exit_threshold: Optional[float]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_path(base: Path, raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    if not p.exists():
        alt = Path(raw).expanduser().resolve()
        if alt.exists():
            return alt
    return p


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_feature_frame(
    *,
    contract_path: Path,
    dataset_contract_path: Optional[Path],
    feature_path: Optional[Path],
    max_rows: int,
) -> Tuple[pd.DataFrame, Path]:
    base_dir = contract_path.parent
    candidates: Iterable[Path]
    if feature_path:
        candidates = (feature_path,)
    else:
        candidates = [p if p.is_absolute() else (base_dir / p) for p in DEFAULT_FEATURE_CANDIDATES]
    for candidate in candidates:
        if candidate.exists():
            df = _read_frame(candidate)
            df = df.tail(max_rows) if max_rows else df
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.sort_values("timestamp").tail(max_rows) if max_rows else df
            return df.reset_index(drop=True), candidate

    contract = load_canonical_contract(str(dataset_contract_path or contract_path))
    df = load_training_dataset(contract)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").tail(max_rows) if max_rows else df
    elif max_rows:
        df = df.tail(max_rows)
    return df.reset_index(drop=True), Path(contract.get("dataset", {}).get("path", "unknown"))


def _resolve_thresholds(policy: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    thresholds = policy.get("thresholds", {}) if isinstance(policy, Mapping) else {}
    if isinstance(thresholds, Mapping) and model in thresholds:
        return thresholds.get(model) or {}
    return thresholds or {}


def _prob_column(artifacts, default: str = "base_prob") -> str:
    col = getattr(artifacts, "prob_column", None)
    if not col:
        return default
    return str(col)


def _predict_probabilities(df: pd.DataFrame, model_dir: Path, model_name: str) -> pd.Series:
    artifacts = load_manifest_artifacts(model_dir, model_label=model_name)
    prob_col = _prob_column(artifacts)
    apply_calibration = getattr(artifacts, "apply_calibration", True)

    if (model_dir / "model.json").exists():
        calibrator, feat_cols = load_base_predictor(
            model_dir,
            prob_column=prob_col,
            apply_calibration=apply_calibration,
        )
        prob_series = predict_base(df, calibrator, feat_cols)
    elif (model_dir / "tcn.pt").exists():
        model, calib, series_cols, scaler, window = load_tcn_predictor(model_dir)
        prob_df = predict_tcn(df, model, calib, series_cols, scaler, window)
        col = getattr(calib, "prob_column", prob_col)
        if col not in prob_df.columns:
            raise KeyError(f"TCN predictor output missing probability column '{col}'")
        prob_series = prob_df[col]
    elif (model_dir / "transformer.pt").exists():
        transformer = TransformerModel.load(str(model_dir))
        prob_series = pd.Series(transformer.predict_proba(df.values), index=df.index, name=prob_col)
    else:
        raise ValueError(f"Unsupported or missing model artifacts under {model_dir}")

    if prob_series.name != prob_col:
        prob_series = prob_series.rename(prob_col)
    return prob_series.astype(float)


def _extract_thresholds(threshold_cfg: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(threshold_cfg, Mapping):
        return None, None
    entry = threshold_cfg.get("entry_long")
    exit_thr = threshold_cfg.get("exit_long", entry)
    try:
        entry = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        entry = None
    try:
        exit_thr = float(exit_thr) if exit_thr is not None else entry
    except (TypeError, ValueError):
        exit_thr = entry
    return entry, exit_thr


def _quantiles(series: pd.Series) -> Tuple[float, float, float, float]:
    if series.empty:
        return (0.0, 0.0, 0.0, 0.0)
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(values.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99))


def _compute_implied_trades(signals: np.ndarray) -> int:
    if len(signals) == 0:
        return 0
    changes = np.diff(signals)
    return int(np.count_nonzero(changes))


def _resolve_prob_gate_min(gate_cfg: Mapping[str, Any], df: pd.DataFrame, mode: str) -> Optional[pd.Series]:
    gate = gate_cfg.get(mode) if isinstance(gate_cfg, Mapping) else {}
    value = gate.get("prob_gate_min") if isinstance(gate, Mapping) else None
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return pd.Series(float(value), index=df.index)
    if isinstance(value, dict):
        symbols = df["symbol"].astype(str) if "symbol" in df.columns else None
        default = value.get("default")
        series = pd.Series(float(default) if default is not None else np.nan, index=df.index, dtype=float)
        if symbols is not None:
            for key, val in value.items():
                if key == "default":
                    continue
                series.loc[symbols == key] = float(val)
        return series
    return None


def _evaluate_symbol(
    df: pd.DataFrame,
    *,
    symbol: str,
    model_dir: Path,
    model_name: str,
    gate_cfg: Mapping[str, Any],
    gate_mode: str,
    threshold_cfg: Mapping[str, Any],
    regime_col: Optional[str],
) -> SymbolCoverage:
    if "symbol" in df.columns:
        sym_df = df[df["symbol"].astype(str).str.upper() == symbol]
    else:
        sym_df = df
    if sym_df.empty:
        return SymbolCoverage(
            symbol=symbol,
            model=model_name,
            p50=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            fraction_above_prob_gate_min=0.0,
            fraction_between_entry_exit=0.0,
            implied_trade_proxy=0,
            coverage_ratio=0.0,
            samples=0,
            prob_gate_min=None,
            entry_threshold=None,
            exit_threshold=None,
        )

    prob_series = _predict_probabilities(sym_df, model_dir, model_name)
    quantiles = _quantiles(prob_series)
    gate_mask = compute_gate_mask(sym_df, gate_cfg, prob=prob_series, mode=gate_mode)
    coverage_ratio = float(gate_mask.mean()) if len(gate_mask) else 0.0

    prob_thresholds = _resolve_prob_gate_min(gate_cfg, sym_df, gate_mode)
    fraction_above = 0.0
    resolved_prob_gate: Optional[float] = None
    if prob_thresholds is not None:
        fraction_above = float((prob_series >= prob_thresholds).mean())
        try:
            resolved_prob_gate = float(prob_thresholds.dropna().median())
        except Exception:
            resolved_prob_gate = None

    entry_thr, exit_thr = _extract_thresholds(threshold_cfg)
    between_frac = 0.0
    if entry_thr is not None and exit_thr is not None:
        upper = max(entry_thr, exit_thr)
        lower = min(entry_thr, exit_thr)
        between_frac = float(((prob_series >= lower) & (prob_series <= upper)).mean())

    signals = apply_thresholds_to_probs(
        prob_series.to_numpy(),
        sym_df,
        threshold_cfg,
        regime_col=regime_col,
        gate_mask=gate_mask,
    )
    implied_trades = _compute_implied_trades(signals)

    return SymbolCoverage(
        symbol=symbol,
        model=model_name,
        p50=quantiles[0],
        p90=quantiles[1],
        p95=quantiles[2],
        p99=quantiles[3],
        fraction_above_prob_gate_min=fraction_above,
        fraction_between_entry_exit=between_frac,
        implied_trade_proxy=implied_trades,
        coverage_ratio=coverage_ratio,
        samples=len(prob_series),
        prob_gate_min=resolved_prob_gate,
        entry_threshold=entry_thr,
        exit_threshold=exit_thr,
    )


def _render_markdown(report: dict) -> str:
    lines = [
        "# Preflight Coverage",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Contract: {report.get('contract')}",
        f"Features: {report.get('feature_path')}",
        "",
    ]
    if report.get("no_go_reasons"):
        lines.append("## NO-GO reasons")
        for reason in report["no_go_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    for sym, metrics in (report.get("symbols") or {}).items():
        lines.append(f"## {sym}")
        lines.append(f"- Model: {metrics.get('model')}")
        lines.append(f"- Samples: {metrics.get('samples')}")
        lines.append(
            f"- Prob quantiles: p50={metrics.get('p50'):.4f} "
            f"p90={metrics.get('p90'):.4f} p95={metrics.get('p95'):.4f} p99={metrics.get('p99'):.4f}"
        )
        lines.append(
            f"- Prob gate min: {metrics.get('prob_gate_min')} "
            f"| fraction>=gate: {metrics.get('fraction_above_prob_gate_min'):.4f}"
        )
        lines.append(
            f"- Entry/exit: {metrics.get('entry_threshold')} / {metrics.get('exit_threshold')} "
            f"| fraction between: {metrics.get('fraction_between_entry_exit'):.4f}"
        )
        lines.append(f"- Gate coverage ratio: {metrics.get('coverage_ratio'):.4f}")
        lines.append(f"- Implied trade proxy (signal changes): {metrics.get('implied_trade_proxy')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Estimate per-symbol coverage and trade readiness using live deployment artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml", help="Deployment contract path.")
    ap.add_argument("--feature-path", default=None, help="Optional features file (CSV/Parquet).")
    ap.add_argument("--max-rows", type=int, default=512, help="Max recent rows to sample from features.")
    ap.add_argument("--epsilon", type=float, default=1e-4, help="Minimum acceptable fraction above prob gate.")
    ap.add_argument("--output-dir", default="reports", help="Directory to write preflight reports.")
    ap.add_argument(
        "--allow-no-go",
        action="store_true",
        help="Do not hard-fail when coverage/trade proxy are zero (override safety).",
    )
    ap.add_argument(
        "--gate-mode",
        default=None,
        help="Override gate mode (defaults to gate_mode in risk limits or inference).",
    )
    args = ap.parse_args(argv)

    contract_path = Path(args.contract).expanduser().resolve()
    summary = validate_deployment_contract(str(contract_path))
    contract = _load_yaml(contract_path)
    base_dir = contract_path.parent

    dataset_contract_path = _resolve_path(base_dir, contract.get("dataset_contract", contract_path))

    models_root = Path(contract.get("models_root", base_dir / "models")).expanduser().resolve()
    models_map = {k: _resolve_path(models_root, v) for k, v in (contract.get("models") or {}).items()}
    policy_path = _resolve_path(base_dir, contract.get("portfolio_policies", ""))
    policies = _load_yaml(policy_path)
    risk_limits_path = _resolve_path(base_dir, contract.get("risk_limits", "configs/portfolio_risk_limits.yaml"))
    risk_cfg = _load_yaml(risk_limits_path)
    gate_cfg = risk_cfg.get("gate_config") or {}
    gate_mode = args.gate_mode or risk_cfg.get("gate_mode") or "inference"

    symbols_cfg = summary.get("symbols", {}) or {}
    symbol_model_map = symbols_cfg.get("symbol_model_key", {}) or {}
    symbol_policy_map = symbols_cfg.get("symbol_policy_map", {}) or {}

    feature_path = Path(args.feature_path).expanduser() if args.feature_path else None
    features, resolved_feature_path = _resolve_feature_frame(
        contract_path=contract_path,
        dataset_contract_path=dataset_contract_path,
        feature_path=feature_path,
        max_rows=max(1, int(args.max_rows)),
    )
    if "symbol" in features.columns:
        features["symbol"] = features["symbol"].astype(str).str.upper()

    results: Dict[str, SymbolCoverage] = {}
    no_go_reasons: List[str] = []

    for symbol, model_key in symbol_model_map.items():
        policy_id = symbol_policy_map.get(symbol, summary.get("default_policy"))
        policy = policies.get(policy_id, {})
        threshold_cfg = _resolve_thresholds(policy, model_key)
        model_path = models_map.get(model_key)
        if model_path is None:
            no_go_reasons.append(f"{symbol}: missing model path for key {model_key}")
            continue
        metrics = _evaluate_symbol(
            features,
            symbol=symbol,
            model_dir=model_path,
            model_name=model_key,
            gate_cfg=gate_cfg,
            gate_mode=gate_mode,
            threshold_cfg=threshold_cfg,
            regime_col=policy.get("regime_col"),
        )
        results[symbol] = metrics
        if metrics.samples == 0:
            no_go_reasons.append(f"{symbol}: no samples available for coverage estimation")
        if metrics.fraction_above_prob_gate_min <= args.epsilon:
            no_go_reasons.append(f"{symbol}: fraction_above_prob_gate_min<=epsilon ({metrics.fraction_above_prob_gate_min:.6f})")

    total_implied_trades = sum(m.implied_trade_proxy for m in results.values())
    if total_implied_trades == 0:
        no_go_reasons.append("implied_trade_proxy==0 across all symbols")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "feature_path": str(resolved_feature_path),
        "max_rows": args.max_rows,
        "gate_mode": gate_mode,
        "epsilon": args.epsilon,
        "symbols": {
            sym: {
                "model": m.model,
                "p50": m.p50,
                "p90": m.p90,
                "p95": m.p95,
                "p99": m.p99,
                "fraction_above_prob_gate_min": m.fraction_above_prob_gate_min,
                "fraction_between_entry_exit": m.fraction_between_entry_exit,
                "implied_trade_proxy": m.implied_trade_proxy,
                "coverage_ratio": m.coverage_ratio,
                "samples": m.samples,
                "prob_gate_min": m.prob_gate_min,
                "entry_threshold": m.entry_threshold,
                "exit_threshold": m.exit_threshold,
            }
            for sym, m in results.items()
        },
        "no_go_reasons": no_go_reasons,
        "hard_fail": bool(no_go_reasons and not args.allow_no_go),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    json_path = output_dir / f"preflight_coverage_{stamp}.json"
    md_path = output_dir / f"preflight_coverage_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(_render_markdown(report))
    print(f"Wrote coverage preflight report to {json_path} and {md_path}")
    if report["hard_fail"]:
        print("Coverage preflight NO-GO: " + "; ".join(no_go_reasons))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
