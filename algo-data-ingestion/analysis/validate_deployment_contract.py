from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_policies(contract: dict) -> Tuple[Dict[str, dict], str]:
    policy_path = contract.get("portfolio_policies")
    policies: Dict[str, dict] = {}
    if policy_path:
        p = Path(policy_path).expanduser()
        _require_path(p, "portfolio_policies")
        policies = _load_yaml(p)
    if not policies:
        policies = contract.get("portfolio_policies_payload") or contract.get("policies") or {}
    default_policy = contract.get("dry_run_integration", {}).get("policy_default", "primary")
    return policies, default_policy


def validate_deployment_contract(contract_path: str) -> dict:
    """
    Validate that the deployment contract references existing artifacts and policies.
    """
    contract_file = Path(contract_path).expanduser()
    _require_path(contract_file, "deployment_contract")
    contract = _load_yaml(contract_file)

    dataset_contract = Path(contract.get("dataset_contract", "")).expanduser()
    best_model_cfg = Path(contract.get("best_model_configs", "")).expanduser()
    risk_limits = Path(contract.get("risk_limits", "")).expanduser()
    portfolio_policies_path = Path(contract.get("portfolio_policies", "")).expanduser()

    for path, label in [
        (dataset_contract, "dataset_contract"),
        (best_model_cfg, "best_model_configs"),
        (risk_limits, "risk_limits"),
        (portfolio_policies_path, "portfolio_policies"),
    ]:
        _require_path(path, label)

    risk_cfg = _load_yaml(risk_limits)
    policies, default_policy = _load_policies(contract)
    if not policies:
        raise ValueError("No policies available in deployment contract.")
    if default_policy not in policies:
        raise ValueError(f"Default policy '{default_policy}' missing from portfolio_policies.")

    models = contract.get("models") or {}
    if not models:
        raise ValueError("No models section found in deployment contract.")
    missing_models = [name for name, path in models.items() if not Path(path).expanduser().exists()]
    if missing_models:
        raise FileNotFoundError(f"Missing model artifact paths: {missing_models}")

    summary = {
        "contract_path": str(contract_file),
        "dataset_contract": str(dataset_contract),
        "best_model_configs": str(best_model_cfg),
        "risk_limits": str(risk_limits),
        "portfolio_policies": str(portfolio_policies_path),
        "default_policy": default_policy,
        "policy_ids": sorted(list(policies.keys())),
        "models": {k: str(v) for k, v in models.items()},
        "risk_limits_keys": list(risk_cfg.keys()),
    }
    print("Deployment contract validation passed.")
    print(f"- dataset_contract: {dataset_contract}")
    print(f"- best_model_configs: {best_model_cfg}")
    print(f"- risk_limits: {risk_limits}")
    print(f"- portfolio_policies: {portfolio_policies_path} (default policy: {default_policy})")
    print(f"- models: {summary['models']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate deployment_portfolio_contract.yaml for completeness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True, help="Path to deployment_portfolio_contract.yaml")
    args = ap.parse_args(argv)

    validate_deployment_contract(args.contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
