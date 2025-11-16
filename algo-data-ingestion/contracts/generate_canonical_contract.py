from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _split_features(feature_entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    core = [f["name"] for f in feature_entries if f.get("type") == "core"]
    experimental = [f["name"] for f in feature_entries if f.get("type") != "core"]
    return {"core": core, "experimental": experimental}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate canonical training contract YAML")
    ap.add_argument("--schema", default="configs/schema_market_multi_3symbol_1m.yaml")
    ap.add_argument("--feature-registry", default="configs/feature_registry_market_multi_3symbol_1m.yaml")
    ap.add_argument("--label-spec", default="configs/label_spec_market_multi_3symbol_1m.yaml")
    ap.add_argument("--regime-spec", default="configs/regime_spec_market_multi_3symbol_1m.yaml")
    ap.add_argument("--dataset-path", default="data/features_labels_regimes_market_multi_3symbol_1m.parquet")
    ap.add_argument("--output", default="configs/canonical_training_contract_market_multi_3symbol_1m.yaml")
    args = ap.parse_args(argv)

    schema = _load_yaml(Path(args.schema))
    features = _load_yaml(Path(args.feature_registry)).get("features", [])
    labels = _load_yaml(Path(args.label_spec))
    regimes = _load_yaml(Path(args.regime_spec))

    split_feats = _split_features(features)
    contract = {
        "dataset": {
            "path": args.dataset_path,
            "freq": schema.get("meta", {}).get("freq", "1min"),
        },
        "features": split_feats,
        "labels": {
            "primary": labels.get("primary_label"),
            "secondary": labels.get("secondary_labels", []),
        },
        "regimes": regimes,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(contract, f, sort_keys=False)
    print(f"Wrote canonical contract to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
