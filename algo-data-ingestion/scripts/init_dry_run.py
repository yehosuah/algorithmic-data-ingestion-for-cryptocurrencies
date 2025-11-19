from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from analysis.validate_deployment_contract import validate_deployment_contract
from live.decision_logic import decide_trades
from training.data import load_canonical_contract, load_training_dataset


def _ensure_dummy_features(path: Path, contract_path: Path, max_rows: int = 256) -> Path:
    if path.exists():
        return path
    contract = load_canonical_contract(str(contract_path))
    df = load_training_dataset(contract)
    sample = df.tail(max_rows).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".csv"}:
        sample.to_csv(path, index=False)
    else:
        sample.to_parquet(path)
    return path


def _derive_promoted_scenario(contract: dict) -> Optional[str]:
    models = contract.get("models") or {}
    for _, model_path in models.items():
        parts = Path(model_path).parts
        if "perf_sweeps" in parts:
            idx = parts.index("perf_sweeps")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def _load_summary_metrics(summary_path: Path, scenario_id: Optional[str]) -> dict:
    if not summary_path.exists() or not scenario_id:
        return {}
    df = pd.read_csv(summary_path)
    row = df[df["scenario_id"] == scenario_id]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "scenario_id": scenario_id,
        "primary_pnl_net": r.get("primary_pnl_net"),
        "primary_sharpe": r.get("primary_sharpe"),
        "trade_count": r.get("trade_count"),
        "fraction_time_in_position": r.get("fraction_time_in_position"),
        "avg_gross_exposure": r.get("avg_gross_exposure"),
        "transaction_cost_bps": r.get("transaction_cost_bps"),
    }


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bootstrap dry-run by validating the deployment contract and running a decision smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--deployment-contract", default="configs/deployment_portfolio_contract.yaml")
    ap.add_argument("--portfolio-policy-id", default=None, help="Policy id to use; defaults to contract's primary.")
    ap.add_argument("--dummy-features-path", default="experiments/dummy_live_features.parquet")
    ap.add_argument("--output-path", default="experiments/decision_smoke_dry_run_init.json")
    ap.add_argument("--summary-file", default="experiments/perf_sweeps/summary.csv")
    args = ap.parse_args(argv)

    summary = validate_deployment_contract(args.deployment_contract)
    contract = yaml.safe_load(Path(args.deployment_contract).read_text()) or {}
    policy_id = args.portfolio_policy_id or summary.get("default_policy") or "primary"
    dummy_path = _ensure_dummy_features(
        Path(args.dummy_features_path),
        Path(contract.get("dataset_contract", "configs/canonical_training_contract_market_multi_3symbol_1m.yaml")),
    )

    features = _read_frame(dummy_path)
    targets = decide_trades(
        live_features=features,
        current_positions=pd.DataFrame(),
        portfolio_policy_id=policy_id,
        deployment_contract_path=args.deployment_contract,
    )
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(targets.to_json(orient="records"))

    promoted_scenario = _derive_promoted_scenario(contract)
    metrics = _load_summary_metrics(Path(args.summary_file), promoted_scenario)
    summary_lines = [
        "Dry-run bootstrap completed.",
        f"- deployment contract: {args.deployment_contract}",
        f"- policy id: {policy_id}",
        f"- decision smoke output: {out_path}",
        f"- dummy features: {dummy_path}",
    ]
    if metrics:
        summary_lines.append(
            f"- promoted scenario: {metrics.get('scenario_id')} | "
            f"sharpe={metrics.get('primary_sharpe')}, "
            f"trade_count={metrics.get('trade_count')}, "
            f"fraction_time_in_position={metrics.get('fraction_time_in_position')}"
        )
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
