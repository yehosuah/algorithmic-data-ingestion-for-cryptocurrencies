from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from training.infer import compute_gate_mask, load_base_predictor, predict_base, load_tcn_predictor, predict_tcn
from training.transformer_model import TransformerModel

from portfolio.gating import apply_thresholds_to_probs
from portfolio.ensemble import combine_model_signals
from portfolio.simulator import _load_gate_config


@lru_cache(maxsize=1)
def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return yaml.safe_load(fh) or {}


def _load_policy(contract: dict, policy_id: str) -> dict:
    pol_path = contract.get("portfolio_policies")
    policies = {}
    if pol_path and Path(pol_path).exists():
        try:
            policies = yaml.safe_load(Path(pol_path).read_text()) or {}
        except Exception:
            policies = {}
    if not policies:
        policies = contract.get("portfolio_policies_payload") or contract.get("policies") or {}
    return policies.get(policy_id) or policies.get("primary") or {}


def _load_probabilities(
    features: pd.DataFrame,
    model_name: str,
    artifacts_root: Path,
    model_path_map: Dict[str, str],
) -> np.ndarray:
    """
    Loader that reuses training.infer helpers for base_xgb, TCN, and transformer manifests.
    """
    prob_col = f"{model_name}_prob"
    if prob_col in features.columns:
        return features[prob_col].to_numpy()

    model_dir = Path(model_path_map.get(model_name, artifacts_root / model_name)).expanduser()
    if (model_dir / "model.json").exists():
        calibrator, feat_cols = load_base_predictor(model_dir)
        return predict_base(features, calibrator, feat_cols).to_numpy()
    if (model_dir / "tcn.pt").exists():
        model, calib, series_cols, scaler, window = load_tcn_predictor(model_dir)
        prob_df = predict_tcn(features, model, calib, series_cols, scaler, window)
        col = getattr(calib, "prob_column", "tcn_prob")
        return prob_df[col].to_numpy()
    if (model_dir / "transformer.pt").exists():
        transformer = TransformerModel.load(str(model_dir))
        if features.values.ndim == 3:
            return transformer.predict_proba(features.values)
        raise ValueError(
            f"Transformer model '{model_name}' requires either precomputed probability column "
            f"('{prob_col}') or a 3D tensor feature matrix."
        )
    raise ValueError(f"Unsupported model artifact layout for {model_name} under {artifacts_root}")


def decide_trades(
    live_features: pd.DataFrame,
    current_positions: pd.DataFrame,
    portfolio_policy_id: str,
    deployment_contract_path: str,
) -> pd.DataFrame:
    """
    Convert live features into target positions using the configured portfolio policy.
    """
    contract = _load_yaml(deployment_contract_path)
    policy = _load_policy(contract, portfolio_policy_id)
    risk_limits_path = contract.get("risk_limits")
    risk_cfg: Dict[str, object] = {}
    if risk_limits_path and Path(risk_limits_path).exists():
        risk_cfg = _load_yaml(risk_limits_path)
    gate_cfg = _load_gate_config(risk_cfg) if risk_cfg else {}
    artifacts_root = Path(contract.get("models_root", contract.get("models_dir", "models"))).expanduser()
    model_path_map: Dict[str, str] = contract.get("models", {})
    models = policy.get("models") or list((contract.get("models") or {}).keys())
    weights = policy.get("weights") or {}
    thresholds = policy.get("thresholds") or {}

    prob_map: Dict[str, np.ndarray] = {}
    for model_name in models:
        prob_map[model_name] = _load_probabilities(live_features, model_name, artifacts_root, model_path_map)

    signals: Dict[str, np.ndarray] = {}
    for model_name, probs in prob_map.items():
        gate_mask = compute_gate_mask(
            live_features,
            gate_cfg,
            prob=pd.Series(probs, index=live_features.index),
            mode=risk_cfg.get("gate_mode", "inference") if risk_cfg else "inference",
        )
        thr_cfg = thresholds if isinstance(thresholds, dict) else {}
        if isinstance(thresholds, dict) and model_name in thresholds:
            thr_cfg = thresholds[model_name]
        signals[model_name] = apply_thresholds_to_probs(
            probs,
            live_features,
            thr_cfg,
            regime_col=policy.get("regime_col"),
            gate_mask=gate_mask,
        )

    combined = combine_model_signals(signals, weights, mode=policy.get("ensemble_mode", "weighted_sum"))
    target = pd.DataFrame(
        {
            "timestamp": live_features["timestamp"] if "timestamp" in live_features else pd.Timestamp.utcnow(),
            "symbol": live_features.get("symbol"),
            "target_position": combined,
        }
    )
    return target


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test for portfolio decision logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--deployment-contract", required=True)
    ap.add_argument("--portfolio-policy-id", default="primary")
    ap.add_argument("--dummy-features-path", required=True, help="CSV/Parquet with live-like features.")
    ap.add_argument("--dummy-positions-path", required=False, help="CSV/Parquet with current positions.")
    ap.add_argument("--output-path", required=True, help="Where to write JSON output.")
    args = ap.parse_args(argv)

    features = _read_frame(Path(args.dummy_features_path))
    positions = _read_frame(Path(args.dummy_positions_path)) if args.dummy_positions_path else pd.DataFrame()

    targets = decide_trades(
        live_features=features,
        current_positions=positions,
        portfolio_policy_id=args.portfolio_policy_id,
        deployment_contract_path=args.deployment_contract,
    )
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(targets.to_json(orient="records"))
    print(f"Wrote target positions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
