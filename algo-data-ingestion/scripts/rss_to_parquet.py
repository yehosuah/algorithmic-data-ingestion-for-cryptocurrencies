#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import json
from typing import Optional, List
import pandas as pd
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.ingestion_service.config import settings
from app.ingestion_service.utils import write_to_parquet
from app.common.time_norm import add_dt_partition
from app.adapters.news_adapter import fetch_news_rss_once


async def _enrich_sentiment(df: pd.DataFrame, *, mode: str, ml_base_url: str, model_id: str, batch_size: int = 64) -> pd.DataFrame:
    """Enrich sentiment using either the local ML endpoint (remote) or in-process transformers (local).

    mode: 'remote' | 'local'
    """
    if df.empty:
        return df
    import httpx
    texts: List[str] = []
    for _, r in df.iterrows():
        t = (r.get("title") or "")
        if not t:
            t = (r.get("description") or "")
        texts.append(str(t))
    scores: List[float] = []
    labels: List[str] = []
    if mode == 'remote':
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                try:
                    resp = await client.post(f"{ml_base_url.rstrip('/')}/ml/sentiment/predict", json={"texts": chunk})
                    resp.raise_for_status()
                    items = resp.json().get("items", [])
                except Exception:
                    items = [{} for _ in chunk]
                for it in items:
                    labels.append(it.get("label"))
                    scores.append(float(it.get("score_signed", 0.0)))
    else:
        # local transformers pipeline
        try:
            from transformers import pipeline
            nlp = pipeline("sentiment-analysis", model=model_id, return_all_scores=True)
        except Exception as e:
            print("[warn] local transformers load failed; neutral sentiment used:", e)
            nlp = None
        if nlp is None:
            labels = [None]*len(texts)
            scores = [0.0]*len(texts)
        else:
            # batch process
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                try:
                    raw = nlp(chunk, truncation=True)
                except Exception:
                    raw = [[] for _ in chunk]
                for scores_all in raw:
                    p_pos, p_neg = 0.0, 0.0
                    for sc in scores_all:
                        lab = str(sc.get("label", "")).lower()
                        if "pos" in lab:
                            p_pos = float(sc.get("score", 0.0))
                        elif "neg" in lab:
                            p_neg = float(sc.get("score", 0.0))
                    score_signed = max(-1.0, min(1.0, p_pos - p_neg))
                    if p_pos >= 0.6:
                        lab = "positive"
                    elif p_pos <= 0.4:
                        lab = "negative"
                    else:
                        lab = "neutral"
                    labels.append(lab)
                    scores.append(score_signed)
    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df


async def main_async(args) -> int:
    df = await fetch_news_rss_once(args.feed, limit=args.limit)
    if df.empty:
        print("No entries fetched.")
        return 0
    # Optional date filters
    if args.start_date:
        df = df[df['published_at'] >= pd.to_datetime(args.start_date, utc=True)]
    if args.end_date:
        df = df[df['published_at'] <= pd.to_datetime(args.end_date, utc=True)]
    # Optional ML enrichment
    if args.enrich_sentiment:
        try:
            mode = 'local' if args.local_model else 'remote'
            df = await _enrich_sentiment(df, mode=mode, ml_base_url=args.ml_base_url, model_id=args.model_id, batch_size=args.ml_batch_size)
        except Exception as e:
            print("[warn] sentiment enrichment failed:", e)
    base = settings.NEWS_PATH.rstrip("/") + "/rss"
    src = urlparse(args.feed).netloc.replace(":", "-")
    # Ensure one-day-per-write to satisfy Parquet writer invariant
    if "dt" not in df.columns:
        add_dt_partition(df, ts_col="published_at")
    wrote = 0
    for d in sorted([x for x in df["dt"].dropna().unique().tolist() if x]):
        day_df = df[df["dt"] == d].copy()
        if day_df.empty:
            continue
        path = write_to_parquet(day_df, base, {"source": src})
        print(f"Wrote RSS parquet: {path}")
        wrote += len(day_df)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch RSS feed once and write to Parquet")
    ap.add_argument("--feed", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--start-date", default=None, help="Optional ISO date to filter from (e.g., 2025-08-01)")
    ap.add_argument("--end-date", default=None, help="Optional ISO date to filter to (e.g., 2025-08-15)")
    ap.add_argument("--enrich-sentiment", action="store_true", help="Call local ML endpoint to score sentiment")
    ap.add_argument("--ml-base-url", default="http://localhost:8000", help="Base URL for ML service")
    ap.add_argument("--ml-batch-size", type=int, default=64)
    ap.add_argument("--local-model", action="store_true", help="Use local transformers model instead of HTTP endpoint")
    ap.add_argument("--model-id", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    args = ap.parse_args(argv)
    import asyncio
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
