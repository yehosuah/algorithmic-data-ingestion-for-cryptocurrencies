from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .thresholds import select_prob_threshold
from .reporting import ensure_kpi_schema, social_signal_audit
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
    "rss_spike_decay_long",
    "rss_spike_decay_fast",
    "rss_spike_presence",
    "rss_spike_velocity",
    "rss_spike_trailing_15",
    "rss_spike_leading_15",
    "rss_spike_halo",
    "rss_spike_proximity",
    "rss_spike_proximity_flag",
    "rss_sent_mean",
    "rss_sent_mean_lag_1",
    "rss_sent_mean_minute",
    "rss_sent_mean_minute_lag_1",
    "rss_sent_mean_minute_ewm",
    "rss_sent_mean_minute_abs",
    "rss_sent_mean_minute_delta",
    "rss_sent_minute_gap",
    "rss_sent_minute_gap_lag_1",
    "rss_sent_minute_gap_ewm",
    "rss_minutes_since_spike",
    "rss_minutes_to_next_spike",
    "rss_spike_streak",
    "rss_count_minute_log1p",
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

    def _coerce(column: str) -> pd.Series:
        if column not in out.columns:
            out[column] = 0.0
        series = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        out[column] = series
        return series

    def _rolling_sum(column: str, window: int) -> pd.Series:
        series = _coerce(column)
        if group_keys is not None:
            rolled = (
                out.groupby(group_keys, sort=False)[column]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
        else:
            rolled = series.rolling(window=window, min_periods=1).sum()
        return pd.to_numeric(rolled, errors="coerce").fillna(0.0)

    def _rolling_max(column: str, window: int) -> pd.Series:
        series = _coerce(column)
        if group_keys is not None:
            rolled = (
                out.groupby(group_keys, sort=False)[column]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
        else:
            rolled = series.rolling(window=window, min_periods=1).max()
        return pd.to_numeric(rolled, errors="coerce").fillna(0.0)

    def _ewm(column: str, span: int) -> pd.Series:
        series = _coerce(column)
        if group_keys is not None:
            ew = (
                out.groupby(group_keys, sort=False)[column]
                .transform(lambda s: s.ewm(span=span, adjust=False).mean())
            )
        else:
            ew = series.ewm(span=span, adjust=False).mean()
        return pd.to_numeric(ew, errors="coerce").fillna(0.0)

    def _compute_spike_windows(spike_series: pd.Series) -> pd.DataFrame:
        arr = spike_series.to_numpy(dtype=float)
        n = len(arr)
        since = np.full(n, np.nan, dtype=float)
        until = np.full(n, np.nan, dtype=float)
        streak = np.zeros(n, dtype=float)
        last_idx = -1
        running = 0
        for i in range(n):
            if arr[i] > 0.0:
                last_idx = i
                running += 1
                streak[i] = running
                since[i] = 0.0
            else:
                running = 0
                streak[i] = 0.0
                if last_idx != -1:
                    since[i] = float(i - last_idx)
        next_idx = -1
        for i in range(n - 1, -1, -1):
            if arr[i] > 0.0:
                next_idx = i
                until[i] = 0.0
            elif next_idx != -1:
                until[i] = float(next_idx - i)
        return pd.DataFrame({
            "rss_minutes_since_spike": since,
            "rss_minutes_to_next_spike": until,
            "rss_spike_streak": streak,
        }, index=spike_series.index)

    base_series: Optional[pd.Series] = _coerce("base_prob") if "base_prob" in out.columns else None
    tcn_series: Optional[pd.Series] = _coerce("tcn_prob") if "tcn_prob" in out.columns else None

    if base_series is not None and tcn_series is not None:
        diff = (base_series - tcn_series).fillna(0.0)
        out["prob_diff"] = diff
        diff_mom = _diff(diff, group_keys, periods=1)
        out["prob_diff_mom_1"] = diff_mom.fillna(0.0) if diff_mom is not None else 0.0
    else:
        out["prob_diff"] = 0.0
        out["prob_diff_mom_1"] = 0.0

    if base_series is not None:
        base_mom = _diff(base_series, group_keys, periods=1)
        out["base_prob_mom_1"] = base_mom.fillna(0.0) if base_mom is not None else 0.0
    else:
        out["base_prob_mom_1"] = 0.0

    if tcn_series is not None:
        tcn_mom = _diff(tcn_series, group_keys, periods=1)
        out["tcn_prob_mom_1"] = tcn_mom.fillna(0.0) if tcn_mom is not None else 0.0
    else:
        out["tcn_prob_mom_1"] = 0.0

    rss_sent = _coerce("rss_sent_mean")
    lag_daily = _shift(rss_sent, group_keys, periods=1)
    out["rss_sent_mean_lag_1"] = lag_daily.fillna(0.0) if lag_daily is not None else 0.0

    rss_sent_minute = _coerce("rss_sent_mean_minute")
    lag_minute = _shift(rss_sent_minute, group_keys, periods=1)
    out["rss_sent_mean_minute_lag_1"] = lag_minute.fillna(0.0) if lag_minute is not None else 0.0
    diff_minute = _diff(rss_sent_minute, group_keys, periods=1)
    out["rss_sent_mean_minute_delta"] = diff_minute.fillna(0.0) if diff_minute is not None else 0.0
    out["rss_sent_mean_minute_abs"] = rss_sent_minute.abs()
    if "rss_sent_mean_minute_ewm" in out.columns:
        out["rss_sent_mean_minute_ewm"] = _coerce("rss_sent_mean_minute_ewm")
    else:
        out["rss_sent_mean_minute_ewm"] = _ewm("rss_sent_mean_minute", 15)

    if "rss_count_minute" in out.columns:
        count_minute = _coerce("rss_count_minute")
        out["rss_count_minute_rollsum_5"] = _rolling_sum("rss_count_minute", 5)
        out["rss_count_minute_rollsum_15"] = _rolling_sum("rss_count_minute", 15)
        rollmax_30 = _rolling_max("rss_count_minute", 30)
        existing_spike = pd.to_numeric(out["rss_spike_active"], errors="coerce").fillna(0.0) if "rss_spike_active" in out.columns else None
        if existing_spike is None or existing_spike.max() == 0.0:
            out["rss_spike_active"] = (rollmax_30 > 0.0).astype(float)
        else:
            out["rss_spike_active"] = existing_spike
        spike_active = pd.to_numeric(out["rss_spike_active"], errors="coerce").fillna(0.0)
        spike_decay = pd.to_numeric(out["rss_spike_decay"], errors="coerce").fillna(0.0) if "rss_spike_decay" in out.columns else None
        if spike_decay is None or spike_decay.max() == 0.0 and spike_active.max() > 0.0:
            spike_decay = _ewm("rss_spike_active", 30)
        out["rss_spike_decay"] = spike_decay
        spike_decay_long = pd.to_numeric(out["rss_spike_decay_long"], errors="coerce").fillna(0.0) if "rss_spike_decay_long" in out.columns else None
        if spike_decay_long is None or spike_decay_long.max() == 0.0 and spike_active.max() > 0.0:
            spike_decay_long = _ewm("rss_spike_active", 60)
        out["rss_spike_decay_long"] = spike_decay_long
        out["rss_spike_presence"] = (pd.to_numeric(out["rss_spike_decay_long"], errors="coerce").fillna(0.0) > 1e-6).astype(float)
        out["rss_count_minute_log1p"] = np.log1p(count_minute.clip(lower=0.0))
    else:
        out["rss_count_minute"] = _coerce("rss_count_minute")
        out["rss_count_minute_rollsum_5"] = _rolling_sum("rss_count_minute", 5)
        out["rss_count_minute_rollsum_15"] = _rolling_sum("rss_count_minute", 15)
        out["rss_spike_active"] = pd.to_numeric(out["rss_spike_active"], errors="coerce").fillna(0.0) if "rss_spike_active" in out.columns else 0.0
        out["rss_spike_decay"] = pd.to_numeric(out["rss_spike_decay"], errors="coerce").fillna(0.0) if "rss_spike_decay" in out.columns else 0.0
        out["rss_spike_decay_long"] = pd.to_numeric(out["rss_spike_decay_long"], errors="coerce").fillna(0.0) if "rss_spike_decay_long" in out.columns else 0.0
        out["rss_spike_presence"] = (pd.to_numeric(out["rss_spike_decay_long"], errors="coerce").fillna(0.0) > 1e-6).astype(float)
        out["rss_count_minute_log1p"] = np.log1p(pd.to_numeric(out["rss_count_minute"], errors="coerce").fillna(0.0).clip(lower=0.0))

    spike_series = pd.to_numeric(out["rss_spike_active"], errors="coerce").fillna(0.0)
    spike_mom = _diff(spike_series, group_keys, periods=1)
    out["rss_spike_active_mom_1"] = spike_mom.fillna(0.0) if spike_mom is not None else 0.0
    spike_velocity = _diff(pd.to_numeric(out["rss_spike_decay"], errors="coerce").fillna(0.0), group_keys, periods=1)
    out["rss_spike_velocity"] = spike_velocity.fillna(0.0) if spike_velocity is not None else 0.0

    if "rss_spike_trailing_15" in out.columns:
        out["rss_spike_trailing_15"] = _coerce("rss_spike_trailing_15")
    else:
        if group_keys is not None:
            trailing = (
                out.groupby(group_keys, sort=False)["rss_spike_active"]
                .transform(lambda s: s.rolling(window=15, min_periods=1).max())
            )
        else:
            trailing = spike_series.rolling(window=15, min_periods=1).max()
        out["rss_spike_trailing_15"] = pd.to_numeric(trailing, errors="coerce").fillna(0.0)

    if "rss_spike_leading_15" in out.columns:
        out["rss_spike_leading_15"] = _coerce("rss_spike_leading_15")
    else:
        if group_keys is not None:
            leading = (
                out.groupby(group_keys, sort=False)["rss_spike_active"]
                .transform(lambda s: s.iloc[::-1].rolling(window=15, min_periods=1).max().iloc[::-1])
            )
        else:
            leading = spike_series.iloc[::-1].rolling(window=15, min_periods=1).max().iloc[::-1]
        out["rss_spike_leading_15"] = pd.to_numeric(leading, errors="coerce").fillna(0.0)

    if "rss_spike_decay_fast" in out.columns:
        out["rss_spike_decay_fast"] = _coerce("rss_spike_decay_fast")
    else:
        if group_keys is not None:
            decay_fast = (
                out.groupby(group_keys, sort=False)["rss_spike_active"]
                .transform(lambda s: s.ewm(span=10, adjust=False).mean())
            )
        else:
            decay_fast = spike_series.ewm(span=10, adjust=False).mean()
        out["rss_spike_decay_fast"] = pd.to_numeric(decay_fast, errors="coerce").fillna(0.0)

    if "rss_spike_halo" in out.columns:
        out["rss_spike_halo"] = _coerce("rss_spike_halo")
    else:
        halo = np.maximum(
            pd.to_numeric(out["rss_spike_trailing_15"], errors="coerce").fillna(0.0),
            pd.to_numeric(out["rss_spike_leading_15"], errors="coerce").fillna(0.0),
        )
        out["rss_spike_halo"] = pd.to_numeric(halo, errors="coerce").clip(0.0, 1.0).fillna(0.0)

    has_proximity_cols = "rss_spike_proximity" in out.columns and "rss_spike_proximity_flag" in out.columns
    if has_proximity_cols:
        out["rss_spike_proximity"] = _coerce("rss_spike_proximity")
        out["rss_spike_proximity_flag"] = _coerce("rss_spike_proximity_flag")

    if group_keys is not None:
        time_feats = out.groupby(group_keys, sort=False)["rss_spike_active"].apply(_compute_spike_windows)
        time_feats.index = time_feats.index.droplevel(0)
    else:
        time_feats = _compute_spike_windows(out["rss_spike_active"])
    for col in ("rss_minutes_since_spike", "rss_minutes_to_next_spike", "rss_spike_streak"):
        out[col] = pd.to_numeric(time_feats[col], errors="coerce").reindex(out.index).fillna(0.0)

    if not has_proximity_cols:
        proximity = pd.concat(
            [
                pd.to_numeric(out["rss_minutes_since_spike"], errors="coerce").replace(0.0, np.nan),
                pd.to_numeric(out["rss_minutes_to_next_spike"], errors="coerce").replace(0.0, np.nan),
            ],
            axis=1,
        ).min(axis=1, skipna=True)
        proximity = proximity.fillna(np.inf)
        mask_inf = pd.Series(~np.isfinite(proximity.to_numpy()), index=proximity.index)
        prox_values = pd.Series(np.exp(-proximity / 5.0), index=proximity.index)
        prox_values[mask_inf] = 0.0
        out["rss_spike_proximity"] = pd.to_numeric(prox_values, errors="coerce").clip(0.0, 1.0).fillna(0.0)
        flag = (proximity <= 15).astype(float)
        flag[mask_inf] = 0.0
        out["rss_spike_proximity_flag"] = pd.to_numeric(flag, errors="coerce").fillna(0.0)

    out["rss_sent_minute_gap"] = (rss_sent_minute - rss_sent).fillna(0.0)
    gap_series = out["rss_sent_minute_gap"]
    gap_lag = _shift(gap_series, group_keys, periods=1)
    out["rss_sent_minute_gap_lag_1"] = gap_lag.fillna(0.0) if gap_lag is not None else 0.0
    out["rss_sent_minute_gap_ewm"] = _ewm("rss_sent_minute_gap", 15) if "rss_sent_minute_gap_ewm" not in out.columns else _coerce("rss_sent_minute_gap_ewm")

    return out


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
    threshold_penalty_weight: float = 0.15,
    threshold_penalty_floor: float = 0.92,
    threshold_grid: Optional[Sequence[float]] = None,
    long_only: bool = True,
    gate_series: Optional[pd.Series] = None,
    class_weight: Optional[str] = "balanced",
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
    if gate_series is not None:
        gate_series = pd.Series(gate_series).astype(float).reset_index(drop=True)
        if len(gate_series) != len(df_proc):
            gate_series = gate_series.reindex(df_proc.index)
        gate_series = gate_series.fillna(0.0)

    if threshold_grid is not None:
        grid = np.array([float(x) for x in threshold_grid], dtype=float)
    else:
        grid = np.concatenate([
            np.linspace(0.45, 0.80, 15),
            np.linspace(0.81, 0.92, 12),
            np.linspace(0.921, 0.995, 10),
        ])
    grid = np.unique(np.clip(grid, 0.0, 0.995))
    if grid.size == 0:
        raise ValueError("Threshold grid cannot be empty.")
    turnover_bonus_weight = float(turnover_bonus_weight)
    sharpe_bonus_weight = float(sharpe_bonus_weight)
    threshold_penalty_weight = float(threshold_penalty_weight)
    threshold_penalty_floor = float(threshold_penalty_floor)

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
    calibration_cv_int = int(calibration_cv) if calibration_cv is not None else 0
    calibration_cv_int = max(0, calibration_cv_int)

    class_weight_value: Optional[str]
    if class_weight is None or str(class_weight).strip().lower() in {"none", ""}:
        class_weight_value = None
    else:
        class_weight_value = "balanced"

    for ratio in l1_grid:
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=float(ratio),
            solver="saga",
            max_iter=1500,
            tol=1e-3,
            class_weight=class_weight_value,
            random_state=42,
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        if calibration_cv_int <= 1:
            calibrated: Union[CalibratedClassifierCV, Pipeline] = pipe
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                calibrated.fit(X.values, y)
            prob = calibrated.predict_proba(X.values)[:, 1]
        else:
            calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=calibration_cv_int, ensemble=False)
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
            gate_mask=gate_series,
        )
        rep = dict(rep)
        rep.update({
            "model_family": "logistic_elastic_net",
            "l1_ratio": float(ratio),
            "calibration_cv": calibration_cv_int if calibration_cv_int > 1 else "disabled",
            "class_weight": class_weight_value or "none",
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
        threshold_penalty = 0.0
        if threshold_penalty_weight != 0.0:
            over = max(0.0, float(thr) - threshold_penalty_floor)
            if over > 0.0:
                threshold_penalty = threshold_penalty_weight * over * max(1.0, turnover)
                score -= threshold_penalty
        rep["threshold_penalty"] = threshold_penalty
        if score > best_score:
            best_score = score
            best_model = calibrated
            best_thr = float(thr)
            best_report = rep

    if best_model is None:
        debug_info = {
            "candidate_count": len(grid_reports),
            "max_total_turnover_seen": max((rep.get("total_turnover", 0.0) for rep in grid_reports), default=None),
            "min_total_turnover_seen": min((rep.get("total_turnover", 0.0) for rep in grid_reports), default=None),
            "sample_reports": grid_reports[:5],
        }
        raise RuntimeError(f"No viable blender configuration found; all candidates violated the turnover/toggle guards. Debug={debug_info}")

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
        "threshold_penalty_weight": threshold_penalty_weight,
        "threshold_penalty_floor": threshold_penalty_floor,
        "grid_reports": grid_reports,
        "long_only": bool(long_only),
    })
    if gate_series is not None:
        gate_series = pd.Series(gate_series).astype(float)
        best_report["gate_mask_share"] = float(gate_series.fillna(0.0).mean())
    return best_model, best_thr, best_report, cols


def save_blender(out_dir: Path, model: object, feat_cols: List[str], threshold: float, report: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "blender.joblib")
    (out_dir / "blender_features.txt").write_text("\n".join(feat_cols))
    (out_dir / "threshold.txt").write_text(str(float(threshold)))
    sanitized = ensure_kpi_schema(report)
    (out_dir / "report.json").write_text(__import__("json").dumps(sanitized, indent=2))
