from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .thresholds import select_prob_threshold


BLENDER_NON_FEAT = {"timestamp", "y_dir", "ret_next", "dt", "symbol", "exchange", "timeframe", "feature_version", "close"}
DEFAULT_CANDIDATE_COLS: Sequence[str] = (
    "base_prob",
    "tcn_prob",
    "prob_diff",
    "prob_diff_mom_1",
    "base_prob_mom_1",
    "tcn_prob_mom_1",
    "rss_count",
    "rss_count_minute",
    "rss_count_minute_rollsum_5",
    "rss_count_minute_rollsum_15",
    "rss_spike_active",
    "rss_spike_active_mom_1",
    "rss_spike_decay",
    "rss_sent_mean",
    "rss_sent_mean_lag_1",
    "rss_sent_mean_minute",
    "rss_sent_mean_minute_lag_1",
    "rss_sent_mean_minute_ewm",
    "rss_sent_minute_gap",
    "rss_sent_minute_gap_lag_1",
    "rss_minutes_since_spike",
    "rss_minutes_to_next_spike",
    "rss_spike_streak",
    "rvol_5",
    "rvol_20",
    "rvol_delta",
)
RSS_PREFIX = ("rss_",)


def _diff(series: pd.Series, group_keys: Optional[pd.Series], periods: int = 1) -> pd.Series:
    if group_keys is not None:
        return series.groupby(group_keys, sort=False).diff(periods)
    return series.diff(periods)


def _shift(series: pd.Series, group_keys: Optional[pd.Series], periods: int = 1) -> pd.Series:
    if group_keys is not None:
        return series.groupby(group_keys, sort=False).shift(periods)
    return series.shift(periods)


def _augment_blender_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_keys: Optional[pd.Series] = out["symbol"] if "symbol" in out.columns else None

    if {"base_prob", "tcn_prob"}.issubset(out.columns):
        out["prob_diff"] = (out["base_prob"].astype(float) - out["tcn_prob"].astype(float)).fillna(0.0)
        diff_mom = _diff(out["prob_diff"], group_keys, periods=1)
        out["prob_diff_mom_1"] = diff_mom.fillna(0.0) if diff_mom is not None else 0.0
    else:
        out["prob_diff"] = 0.0
        out["prob_diff_mom_1"] = 0.0

    if "base_prob" in out.columns:
        base_series = out["base_prob"].astype(float)
        base_mom = _diff(base_series, group_keys, periods=1)
        out["base_prob_mom_1"] = base_mom.fillna(0.0) if base_mom is not None else 0.0
    else:
        out["base_prob_mom_1"] = 0.0

    if "tcn_prob" in out.columns:
        tcn_series = out["tcn_prob"].astype(float)
        tcn_mom = _diff(tcn_series, group_keys, periods=1)
        out["tcn_prob_mom_1"] = tcn_mom.fillna(0.0) if tcn_mom is not None else 0.0
    else:
        out["tcn_prob_mom_1"] = 0.0

    if "rss_sent_mean" in out.columns:
        rss_sent = out["rss_sent_mean"].astype(float)
        lag = _shift(rss_sent, group_keys, periods=1)
        out["rss_sent_mean_lag_1"] = lag.fillna(0.0) if lag is not None else 0.0
    else:
        out["rss_sent_mean_lag_1"] = 0.0

    if "rss_sent_mean_minute" in out.columns:
        rss_sent_min = out["rss_sent_mean_minute"].astype(float)
        lag_min = _shift(rss_sent_min, group_keys, periods=1)
        out["rss_sent_mean_minute_lag_1"] = lag_min.fillna(0.0) if lag_min is not None else 0.0
    else:
        out["rss_sent_mean_minute_lag_1"] = 0.0

    if "rss_spike_active" in out.columns:
        spike = out["rss_spike_active"].astype(float)
        spike_mom = _diff(spike, group_keys, periods=1)
        out["rss_spike_active"] = spike.fillna(0.0)
        out["rss_spike_active_mom_1"] = spike_mom.fillna(0.0) if spike_mom is not None else 0.0
    else:
        out["rss_spike_active"] = 0.0
        out["rss_spike_active_mom_1"] = 0.0

    if "rss_spike_decay" not in out.columns:
        out["rss_spike_decay"] = 0.0
    else:
        out["rss_spike_decay"] = pd.to_numeric(out["rss_spike_decay"], errors="coerce").fillna(0.0)

    if "rss_sent_minute_gap" in out.columns:
        gap = out["rss_sent_minute_gap"].astype(float)
        gap_lag = _shift(gap, group_keys, periods=1)
        out["rss_sent_minute_gap"] = gap.fillna(0.0)
        out["rss_sent_minute_gap_lag_1"] = gap_lag.fillna(0.0) if gap_lag is not None else 0.0
    else:
        out["rss_sent_minute_gap"] = 0.0
        out["rss_sent_minute_gap_lag_1"] = 0.0

    for col in (
        "rss_count_minute_rollsum_5",
        "rss_count_minute_rollsum_15",
        "rss_sent_mean_minute_ewm",
        "rss_minutes_since_spike",
        "rss_minutes_to_next_spike",
        "rss_spike_streak",
    ):
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for col in (
        "prob_diff",
        "prob_diff_mom_1",
        "base_prob_mom_1",
        "tcn_prob_mom_1",
        "rss_sent_mean_lag_1",
        "rss_sent_mean_minute_lag_1",
        "rss_spike_active",
        "rss_spike_active_mom_1",
        "rss_spike_decay",
        "rss_sent_minute_gap",
        "rss_sent_minute_gap_lag_1",
        "rss_count_minute_rollsum_5",
        "rss_count_minute_rollsum_15",
        "rss_sent_mean_minute_ewm",
        "rss_minutes_since_spike",
        "rss_minutes_to_next_spike",
        "rss_spike_streak",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def social_signal_audit(
    df: pd.DataFrame,
    *,
    min_daily_coverage: float = 0.80,
    min_minute_spike_share: float = 0.0005,
) -> Dict[str, Optional[float] | bool | List[str]]:
    """
    Evaluate RSS coverage metrics to determine whether RSS-derived features are reliable.
    Returns metadata describing coverage and whether the RSS feature set should be used.
    """
    result: Dict[str, Optional[float] | bool | List[str]] = {
        "min_daily_coverage": float(min_daily_coverage),
        "min_minute_spike_share": float(min_minute_spike_share),
        "daily_coverage": None,
        "minute_spike_share": None,
        "passed": True,
        "reasons": [],
        "minute_indicator_column": None,
    }

    if "rss_has_signal" in df.columns:
        daily_cov = pd.to_numeric(df["rss_has_signal"], errors="coerce").fillna(0.0).mean()
        result["daily_coverage"] = float(daily_cov)
        if daily_cov < min_daily_coverage:
            result["passed"] = False
            result["reasons"].append("rss_has_signal_below_threshold")
    else:
        result["passed"] = False
        result["reasons"].append("rss_has_signal_missing")

    minute_indicator_col: Optional[str] = None
    if "rss_spike_active" in df.columns:
        minute_indicator_col = "rss_spike_active"
    elif "rss_count_minute" in df.columns:
        minute_indicator_col = "rss_count_minute"

    if minute_indicator_col is not None:
        minute_share = pd.to_numeric(df[minute_indicator_col], errors="coerce").fillna(0.0)
        minute_share = float((minute_share > 0.0).mean())
        result["minute_spike_share"] = minute_share
        result["minute_indicator_column"] = minute_indicator_col
        if minute_share < min_minute_spike_share:
            result["passed"] = False
            result["reasons"].append("rss_minute_spike_share_below_threshold")
    else:
        result["passed"] = False
        result["reasons"].append("rss_minute_indicator_missing")

    result["fallback_to_no_rss"] = not result["passed"]
    return result


def build_blender_features(
    df: pd.DataFrame,
    *,
    candidate_cols: Optional[Sequence[str]] = None,
    use_rss_features: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    prepared = _augment_blender_frame(df)
    if candidate_cols is None:
        candidate_cols = DEFAULT_CANDIDATE_COLS

    if not use_rss_features:
        candidate_cols = [c for c in candidate_cols if not c.startswith(RSS_PREFIX)]

    cols = [c for c in candidate_cols if c in prepared.columns]
    if not cols:
        raise ValueError("No blender feature columns found. Provide candidate_cols or ensure columns exist.")
    X = prepared[cols].astype(float).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return X, list(cols)


def train_blender(
    df: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    spread_series: Optional[pd.Series] = None,
    spread_scale: float = 0.0,
    slippage_bps: float = 0.0,
    l1_ratio_grid: Optional[Sequence[float]] = None,
    calibration_cv: int = 5,
    max_total_turnover: float = 200.0,
    min_total_turnover: float = 50.0,
    min_toggle_count: int = 2,
    min_daily_rss_coverage: float = 0.80,
    min_minute_rss_share: float = 0.0005,
    turnover_bonus_weight: float = 0.001,
    sharpe_bonus_weight: float = 0.05,
    threshold_grid: Optional[Sequence[float]] = None,
    long_only: bool = True,
) -> Tuple[CalibratedClassifierCV, float, Dict, List[str]]:
    df_proc = df.reset_index(drop=True)
    min_total_turnover = float(min_total_turnover)
    audit = social_signal_audit(
        df_proc,
        min_daily_coverage=min_daily_rss_coverage,
        min_minute_spike_share=min_minute_rss_share,
    )
    use_rss = audit.get("passed", False)

    X, cols = build_blender_features(df_proc, use_rss_features=bool(use_rss))
    y = df_proc["y_dir"].astype(int).values

    if threshold_grid is not None:
        grid = np.array([float(x) for x in threshold_grid], dtype=float)
    else:
        grid = np.concatenate([
            np.linspace(0.60, 0.85, 11),
            np.linspace(0.86, 0.95, 10),
            np.linspace(0.951, 0.995, 10),
        ])
    grid = np.unique(np.clip(grid, 0.0, 0.995))
    if grid.size == 0:
        raise ValueError("Threshold grid cannot be empty.")
    turnover_bonus_weight = float(turnover_bonus_weight)
    sharpe_bonus_weight = float(sharpe_bonus_weight)

    if spread_series is not None:
        spread_series = pd.Series(spread_series).astype(float).reset_index(drop=True)
        if len(spread_series) != len(df_proc):
            spread_series = spread_series.reindex(df_proc.index)
        spread_series = spread_series.fillna(0.0)
    else:
        spread_series = None

    l1_grid = list(l1_ratio_grid) if l1_ratio_grid is not None else [0.15, 0.35, 0.55, 0.75, 0.9]
    best_model: Optional[CalibratedClassifierCV] = None
    best_score = -float("inf")
    best_thr = 0.0
    best_report: Dict = {}
    grid_reports: List[Dict] = []
    calibration_cv = max(2, int(calibration_cv))

    for ratio in l1_grid:
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=float(ratio),
            solver="saga",
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=calibration_cv, ensemble=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            calibrated.fit(X.values, y)
        prob = calibrated.predict_proba(X.values)[:, 1]
        prob_series = pd.Series(prob, index=df_proc.index, name="blender_prob")
        thr, rep = select_prob_threshold(
            df_proc["ret_next"],
            prob_series,
            cost_bps=cost_bps,
            grid=grid,
            spread_series=spread_series,
            spread_scale=spread_scale,
            slippage_bps=slippage_bps,
            min_total_turnover=min_total_turnover,
            max_total_turnover=max_total_turnover,
            long_only=long_only,
        )
        rep = dict(rep)
        rep.update({
            "model_family": "logistic_elastic_net",
            "l1_ratio": float(ratio),
            "calibration_cv": calibration_cv,
        })
        turnover = float(rep.get("total_turnover", 0.0))
        rep["total_turnover"] = turnover
        if turnover < float(min_total_turnover):
            rep["rejected"] = True
            rep["rejected_reason"] = f"total_turnover<{float(min_total_turnover)}"
            grid_reports.append(rep)
            continue

        toggle_count = rep.get("toggle_count", 0)
        if toggle_count <= int(min_toggle_count):
            rep["rejected"] = True
            rep["rejected_reason"] = f"toggle_count<= {int(min_toggle_count)}"
            grid_reports.append(rep)
            continue

        rep["rejected"] = False
        grid_reports.append(rep)

        score = rep.get("final_equity", -float("inf"))
        if turnover_bonus_weight > 0.0:
            turnover_cap = float(max_total_turnover) if max_total_turnover is not None else turnover
            turnover_adjust = min(turnover, turnover_cap)
            score += turnover_bonus_weight * turnover_adjust
        if sharpe_bonus_weight != 0.0:
            score += sharpe_bonus_weight * float(rep.get("sharpe", 0.0))
        if score > best_score:
            best_score = score
            best_model = calibrated
            best_thr = float(thr)
            best_report = rep

    if best_model is None:
        raise RuntimeError("No viable blender configuration found; all candidates violated the turnover/toggle guards.")

    best_report = dict(best_report)
    best_report.update({
        "candidate_feature_count": len(cols),
        "candidate_features": list(cols),
        "rss_audit": audit,
        "max_total_turnover_guard": float(max_total_turnover) if max_total_turnover is not None else None,
        "min_total_turnover_guard": float(min_total_turnover),
        "min_toggle_count_guard": int(min_toggle_count),
        "l1_ratio_grid": [float(x) for x in l1_grid],
        "threshold_grid": [float(x) for x in grid],
        "turnover_bonus_weight": turnover_bonus_weight,
        "sharpe_bonus_weight": sharpe_bonus_weight,
        "grid_reports": grid_reports,
        "long_only": bool(long_only),
    })
    return best_model, best_thr, best_report, cols


def save_blender(out_dir: Path, model: object, feat_cols: List[str], threshold: float, report: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "blender.joblib")
    (out_dir / "blender_features.txt").write_text("\n".join(feat_cols))
    (out_dir / "threshold.txt").write_text(str(float(threshold)))
    (out_dir / "report.json").write_text(__import__("json").dumps(report, indent=2))
