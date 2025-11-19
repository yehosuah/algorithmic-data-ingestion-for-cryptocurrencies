#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


def summarize(df: pd.DataFrame) -> str:
    lines = ["# Sampling & Weighting Summary", ""]
    if df.empty:
        lines.append("No results available.")
        return "\n".join(lines)
    grouped = df.groupby(["model_name", "sampling_policy", "weight_policy"], dropna=False).agg(
        mean_pnl=("mean_pnl_net_cv", "mean"),
        mean_sharpe=("mean_sharpe_cv", "mean"),
        count=("config_hash", "count"),
    ).reset_index()
    best = grouped.sort_values(["mean_sharpe", "mean_pnl"], ascending=False)
    lines.append("## Top combinations")
    for _, row in best.head(10).iterrows():
        lines.append(
            f"- {row['model_name']} | sampling={row['sampling_policy']} | weight={row['weight_policy']} | sharpe={row['mean_sharpe']:.4f} | pnl={row['mean_pnl']:.6f} | n={int(row['count'])}"
        )
    lines.append("")
    for model in df["model_name"].unique():
        sub = best[best["model_name"] == model]
        if sub.empty:
            continue
        top = sub.iloc[0]
        lines.append(f"## Recommendation for {model}")
        lines.append(
            f"- Recommend sampling={top['sampling_policy']} weight={top['weight_policy']} (sharpe={top['mean_sharpe']:.4f}, pnl={top['mean_pnl']:.6f})"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze sampling/weighting comparison results.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="reports/sampling_weighting_summary.md")
    args = ap.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    md = summarize(df)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    summary_json = {
        "recommendations": md,
        "top_rows": df.sort_values(["mean_sharpe_cv", "mean_pnl_net_cv"], ascending=False).head(10).to_dict(orient="records"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary_json, indent=2))
    print(f"[SamplingAnalysis] Wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
