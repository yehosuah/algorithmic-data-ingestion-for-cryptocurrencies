#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


def _extract_params(row: pd.Series) -> Dict:
    params = {}
    for col, val in row.items():
        if col.startswith("param_"):
            params[col.replace("param_", "")] = val
    return params


def _process_model(model: str, res_path: Path, min_sharpe: float, top_n: int) -> List[Dict]:
    if not res_path.exists():
        return []
    df = pd.read_csv(res_path)
    if "mean_sharpe_cv" in df.columns:
        df = df[df["mean_sharpe_cv"] >= min_sharpe]
    df = df.sort_values(["mean_sharpe_cv", "mean_pnl_net_cv"], ascending=False)
    rows = []
    for _, r in df.head(top_n).iterrows():
        rows.append({
            "model": model,
            "trial_id": r["trial_id"],
            "mean_sharpe_cv": float(r.get("mean_sharpe_cv", 0.0)),
            "mean_pnl_net_cv": float(r.get("mean_pnl_net_cv", 0.0)),
            "params": _extract_params(r),
        })
    return rows


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promote best hyperparameter configs across models.")
    ap.add_argument(
        "--search-roots",
        nargs="*",
        default=None,
        help="List of folders containing per-model search results. If omitted, uses --search-root.",
    )
    ap.add_argument("--search-root", default="experiments/hparam_search", help="Fallback folder containing per-model search results.")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of models to process. Defaults to subdirectories under search-root.")
    ap.add_argument("--min-sharpe", type=float, default=0.0)
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--output", default="configs/best_model_configs.yaml")
    args = ap.parse_args(argv)

    roots: List[Path] = []
    if args.search_roots:
        roots = [Path(r) for r in args.search_roots]
    else:
        roots = [Path(args.search_root)]

    candidates = args.models
    if candidates is None:
        discovered = []
        for root in roots:
            discovered.extend([p.name for p in root.iterdir() if p.is_dir()])
        candidates = sorted(set(discovered))

    promoted: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []
    for model in candidates:
        res_path: Optional[Path] = None
        for root in roots:
            candidate = root / model / "results.csv"
            if candidate.exists():
                res_path = candidate
                break
        if res_path is None:
            # Allow passing fully-qualified per-model folders
            for root in roots:
                candidate = root / "results.csv"
                if candidate.exists() and (root.name == model or root.name.startswith(model)):
                    res_path = candidate
                    break
        if res_path is None:
            continue
        best = _process_model(model, res_path, args.min_sharpe, args.top_n)
        if not best:
            continue
        promoted[model] = {
            "best_config_id": best[0]["trial_id"],
            "params": best[0]["params"],
        }
        summary_rows.extend(best)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(promoted, f, sort_keys=True)
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"[Promote] Wrote {len(promoted)} promoted configs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
