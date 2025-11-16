from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.schema_inspect import load_market_dataset
from features.feature_engineering import apply_feature_registry, FEATURE_REGISTRY


def _load_schema(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build feature set for market_multi_3symbol_1m.parquet")
    ap.add_argument("--input", default="datasets/market_multi_3symbol_1m.parquet")
    ap.add_argument("--output", default="data/features_market_multi_3symbol_1m.parquet")
    ap.add_argument("--config", default="configs/schema_market_multi_3symbol_1m.yaml")
    args = ap.parse_args(argv)

    df = load_market_dataset(args.input)
    schema = _load_schema(Path(args.config))
    if schema:
        df.attrs["_schema"] = schema

    feat_df = apply_feature_registry(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    print(f"Built features for {len(df)} rows using {len(FEATURE_REGISTRY)} features -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
