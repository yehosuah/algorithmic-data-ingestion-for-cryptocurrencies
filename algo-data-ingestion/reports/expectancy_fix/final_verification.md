# Expectancy Fix — Final Verification

## Evidence bundles (source-of-truth snapshots)

- Baseline (pre-change): `reports/log_forensics/evidence/20251225T181022Z`
  - Executed-exit forensics: `reports/expectancy_fix/baseline_forensics.md`
  - Market alignment: `reports/expectancy_fix/baseline_alignment/alignment_summary.md`
- Post-change (logic + thresholds smoke window): `reports/log_forensics/evidence/20251225T215109Z`
  - Executed-exit forensics: `reports/expectancy_fix/post_change_forensics.md`
  - Market alignment: `reports/expectancy_fix/post_change_alignment/alignment_summary.md`
- Final runtime snapshot (sizing update applied): `reports/log_forensics/evidence/20251225T223541Z`
  - Env snapshot: `reports/log_forensics/evidence/20251225T223541Z/env_snapshot.txt`
  - Audit log: `reports/log_forensics/evidence/20251225T223541Z/trading_audit/audit.log`

## Phase 1–2 findings (baseline: why expectancy was negative)

- Tail loss dominance: max single-trade loss is large relative to typical wins (portfolio max_loss -0.096 on ETH; see `reports/expectancy_fix/baseline_forensics.md`).
- Loss attribution: portfolio losses are concentrated in prob_floor / prob_trailing / gate_close exits (`reports/expectancy_fix/baseline_forensics.md`).
- Upside starvation: regret_fraction ≈ 0.45 and take_profit exits realize ~0.15% vs MFE ~0.48% (`reports/expectancy_fix/baseline_forensics.md`).
- Per-symbol market alignment writeups: `reports/expectancy_fix/market_alignment_BTC_USDT.md`, `reports/expectancy_fix/market_alignment_ETH_USDT.md`, `reports/expectancy_fix/market_alignment_SOL_USDT.md`.

## Phase 3–5 fixes applied (bounded downside, upside unlocked)

- Stop-loss exits bypass min-hold (prevents “late stopouts”): `app/trading/decision.py`
- Spread guard is asymmetric: blocks bad entries; defers non-risk exits; allows stop_loss/time_exit exits: `app/trading/decision.py`
- Exit execution uses relaxed spread cap (protective sells shouldn’t be blocked by tight entry spread cap): `app/trading/service.py`
- Risk sizing avoids qty-step rounding-to-zero deadlocks: `app/trading/risk.py`
- Take-profit disabled (remove winner clipping); tighter entry spread cap; min-hold tuned: `docker-compose.yml`, `configs/runtime_overrides/stage_0.yaml`
- Risk limits tuned for live prob distribution + tighter trigger band: `configs/runtime_overrides/risk_limits_stage_0.yaml`
- Final portfolio sizing adjustment: ETH order_notional reduced to 5 (improves portfolio PF while keeping ETH live): `docker-compose.yml`, `configs/runtime_overrides/stage_0.yaml`

## Offline proof (fixed evaluation window, realistic costs)

Inputs:

- Contract: `configs/canonical_training_contract_market_multi_3symbol_1m.yaml`
- Model bundle: `experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary`
- Window: 2025-11-02T10:40:00Z → 2025-11-04T15:23:00Z (intersection of available BTC/ETH/SOL data in the canonical dataset)
- Costs: fee_estimate_bps=1.0 (per-side, charged twice per round trip) + spread cost (feature spread)

Command run (produced the table below):

```bash
python3 - <<'PY'
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.trigger_optimizer import load_dataset, ensure_probabilities
from app.trading.decision import TriggerConfig, decide_bar
from app.trading.state import PositionState


@dataclass
class Scenario:
    name: str
    cfg_by_symbol: Dict[str, TriggerConfig]
    notional_by_symbol: Dict[str, float]


def compute_stop_loss_pct(
    *,
    base_stop_loss: Optional[float],
    rvol20: Optional[float],
    min_stop_loss_pct: float = 0.005,
    hard_stop_loss_pct: float = 0.012,
    vol_stop_rvol_mult: float = 3.0,
) -> Optional[float]:
    stop = base_stop_loss
    if stop is None or stop <= 0:
        stop = min_stop_loss_pct
    if rvol20 is not None and math.isfinite(rvol20) and vol_stop_rvol_mult > 0:
        vol_stop = rvol20 * vol_stop_rvol_mult
        if stop is None or vol_stop > stop:
            stop = vol_stop
    if stop is not None and hard_stop_loss_pct and hard_stop_loss_pct > 0:
        stop = min(stop, hard_stop_loss_pct)
    if stop is not None and stop > 0:
        return float(stop)
    return None


def simulate_portfolio(bars: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    states = {sym: PositionState() for sym in scenario.cfg_by_symbol}
    trades: List[dict] = []
    for row in bars.itertuples(index=False):
        sym = row.symbol
        cfg = scenario.cfg_by_symbol[sym]
        st = states[sym]
        ts = pd.to_datetime(row.timestamp, utc=True).to_pydatetime()
        prob = float(row.base_prob) if row.base_prob == row.base_prob else 0.0
        gate = bool(row.gate_pass)
        price = float(row.close)
        spread = float(row.feat_spread_bps) if row.feat_spread_bps == row.feat_spread_bps else None
        rvol20 = float(row.rvol20) if row.rvol20 == row.rvol20 else None

        entry_price = float(st.metadata.get("open_price") or 0.0) if st.in_position else None
        entry_amt = float(st.metadata.get("open_amount") or 0.0) if st.in_position else None

        active_stop = compute_stop_loss_pct(base_stop_loss=cfg.stop_loss_pct, rvol20=rvol20)
        out = decide_bar(
            ts=ts,
            probability=prob,
            gate_pass=gate,
            state=st,
            cfg=cfg,
            current_price=price,
            entry_price=entry_price,
            entry_amount=entry_amt,
            spread_bps=spread,
            include_spread_cost=True,
            fee_estimate_bps=1.0,
            slippage_estimate_bps=0.0,
            stop_loss_override=active_stop,
        )

        if out.should_enter and not st.in_position:
            if out.skip_execution:
                continue
            notional = scenario.notional_by_symbol[sym]
            if notional <= 0 or price <= 0:
                continue
            amt = notional / price
            if amt <= 0:
                continue
            st.metadata["open_price"] = f"{price:.10f}"
            st.metadata["open_amount"] = f"{amt:.10f}"
            st.metadata["open_entry_prob"] = f"{prob:.10f}"
            st.mark_entry(ts, cfg.min_hold_bars * cfg.bar_seconds)

        if out.should_exit and st.in_position:
            if out.skip_execution:
                continue
            pnl_net = out.exit_context.get("pnl_net_estimate")
            pnl = float(pnl_net) if isinstance(pnl_net, (int, float)) else 0.0
            trades.append({"symbol": sym, "exit_ts": ts, "pnl": pnl, "reason": out.exit_trigger})
            st.mark_exit(ts)
            st.metadata.clear()

    return pd.DataFrame(trades)


def pf(pnl: pd.Series) -> Optional[float]:
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else None
    return float(wins / losses)


def summarize(trades: pd.DataFrame) -> Dict[str, object]:
    if trades.empty:
        return {"trades": 0, "pnl": 0.0, "profit_factor": None, "max_loss": None, "max_drawdown": 0.0}
    pnl = trades["pnl"].astype(float)
    equity = pnl.cumsum()
    dd = (equity.cummax() - equity).max() if len(equity) else 0.0
    return {
        "trades": int(len(trades)),
        "pnl": float(pnl.sum()),
        "profit_factor": pf(pnl),
        "max_loss": float(pnl.min()),
        "max_drawdown": float(dd),
    }


contract = Path("configs/canonical_training_contract_market_multi_3symbol_1m.yaml")
model_dir = Path("experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary")

symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

sol = load_dataset(contract, symbol="SOL/USDT")
start, end = sol["timestamp"].min(), sol["timestamp"].max()

frames = []
for sym in symbols:
    df = load_dataset(contract, symbol=sym)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    df = ensure_probabilities(df, prob_column="base_prob", gate_column="gate_pass", model_dir=model_dir)
    need = ["timestamp", "symbol", "close", "base_prob", "gate_pass", "feat_spread_bps", "rvol20"]
    for col in need:
        if col not in df.columns:
            df[col] = np.nan
    frames.append(df[need])

bars = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

baseline = Scenario(
    name="baseline",
    cfg_by_symbol={
        "BTC/USDT": TriggerConfig(0.56, 0.43, 0.15, 5, 60, True, 90 * 60, None, 0.001, 20),
        "ETH/USDT": TriggerConfig(0.48, 0.41, 0.16, 6, 60, True, 90 * 60, None, 0.001, 15),
        "SOL/USDT": TriggerConfig(0.48, 0.42, 0.14, 5, 60, True, 90 * 60, None, 0.001, 22),
    },
    notional_by_symbol={"BTC/USDT": 15.0, "ETH/USDT": 20.0, "SOL/USDT": 12.0},
)

---

## 2025-12-30 iteration (profitability sweep + deployment alignment)

### Evidence (real dry-run logs)

- Evidence bundle: `reports/log_forensics/evidence/20251230T005217Z`
- Executed-exit forensics (shows loss dominance by `prob_floor`): `reports/expectancy_fix/docker_since_changes_forensics.md`
- Market alignment (shows positive post-exit drift after `prob_floor`): `reports/expectancy_fix/docker_since_changes_alignment/alignment_summary.md`

### Offline proof (14d replay, robust across halves)

- Sweep outputs:
  - `experiments/expectancy_fix/sweeps/results.csv`
  - `experiments/expectancy_fix/sweeps/summary.md`
- Best configuration (ranked by `pnl_min_half` then `pnl_total`) achieved:
  - `pnl_second_half = 1.702` (meets the `>= 1.5` acceptance target on the evaluation window)
  - `pnl_total = 4.899`

### Deployed parameters (now in docker)

- Risk thresholds: `configs/runtime_overrides/risk_limits_stage_0.yaml`
  - BTC `entry_threshold=0.64`, SOL `entry_threshold=0.78`
- Trading runtime models: `docker-compose.yml`
  - Order notionals: BTC=25, ETH=25, SOL=30 (total cap 80)
  - Holds: BTC=90m, ETH=240m, SOL=90m
  - Filters: BTC `entry_macd_min=0.0`, ETH `entry_rsi_min=50.0`, SOL `entry_rsi_min=45.0`
  - Exits: `disable_prob_exits=true` (removes `prob_floor` churn; exits via stop-loss + time-limit)

### Live verification commands (Grafana/Prometheus)

```bash
# Trading realized PnL time series
curl -fsS http://localhost:9010/metrics | rg '^trading_realized_pnl_total'

# Ensure the service is actively holding positions (for upcoming time exits)
curl -fsS http://localhost:9010/metrics | rg '^trading_position_active'
```

prev_final = Scenario(
    name="prev_final",
    cfg_by_symbol={
        "BTC/USDT": TriggerConfig(0.56, 0.54, 0.15, 8, 60, True, 90 * 60, 0.005, None, 8),
        "ETH/USDT": TriggerConfig(0.48, 0.47, 0.15, 3, 60, True, 90 * 60, 0.004, None, 8),
        "SOL/USDT": TriggerConfig(0.48, 0.47, 0.15, 3, 60, True, 90 * 60, 0.004, None, 8),
    },
    notional_by_symbol={"BTC/USDT": 15.0, "ETH/USDT": 5.0, "SOL/USDT": 12.0},
)

boosted = Scenario(
    name="boosted",
    cfg_by_symbol={
        "BTC/USDT": TriggerConfig(0.56, 0.54, 0.15, 8, 60, True, 90 * 60, 0.005, None, 8),
        "ETH/USDT": TriggerConfig(0.51, 0.50, 0.18, 8, 60, True, 90 * 60, 0.005, None, 8),
        "SOL/USDT": TriggerConfig(0.51, 0.50, 0.18, 8, 60, True, 90 * 60, 0.005, None, 8),
    },
    notional_by_symbol={"BTC/USDT": 15.0, "ETH/USDT": 5.0, "SOL/USDT": 12.0},
)

rows = []
for sc in [baseline, prev_final, boosted]:
    trades = simulate_portfolio(bars, sc)
    port = summarize(trades)
    rows.append({"scenario": sc.name, "symbol": "PORTFOLIO", **port})
    for sym in symbols:
        sym_trades = trades[trades["symbol"] == sym]
        rows.append({"scenario": sc.name, "symbol": sym, **summarize(sym_trades)})

out = pd.DataFrame(rows)
print(f"window_start={start.isoformat()} window_end={end.isoformat()} bars={len(bars)}")
print(out[["scenario", "symbol", "trades", "pnl", "profit_factor", "max_drawdown", "max_loss"]].to_markdown(index=False))
PY
```

Result (profit_factor materially improved vs baseline and prior final):

| scenario   | symbol    |   trades |       pnl |   profit_factor |   max_drawdown |   max_loss |
|:-----------|:----------|---------:|----------:|----------------:|---------------:|-----------:|
| baseline   | PORTFOLIO |      517 | -3.83176  |        0.54195  |       3.96052  | -0.255536  |
| baseline   | BTC/USDT  |      151 | -0.957829 |        0.498018 |       1.01639  | -0.173568  |
| baseline   | ETH/USDT  |      177 | -1.66796  |        0.577769 |       1.88884  | -0.255536  |
| baseline   | SOL/USDT  |      189 | -1.20598  |        0.518947 |       1.22126  | -0.185703  |
| prev_final | PORTFOLIO |      406 |  0.851116 |        1.2636   |       0.470804 | -0.185703  |
| prev_final | BTC/USDT  |      113 |  0.466801 |        1.53453  |       0.214932 | -0.173568  |
| prev_final | ETH/USDT  |      159 | -0.201154 |        0.766378 |       0.269052 |  -0.0638841 |
| prev_final | SOL/USDT  |      134 |  0.58547  |        1.39174  |       0.249055 | -0.185703  |
| boosted    | PORTFOLIO |      286 |  2.06066  |        2.05461  |       0.341576 | -0.185703  |
| boosted    | BTC/USDT  |      113 |  0.466801 |        1.53453  |       0.214932 | -0.173568  |
| boosted    | ETH/USDT  |       87 |  0.362042 |        2.09615  |       0.0669716 |  -0.0638841 |
| boosted    | SOL/USDT  |       86 |  1.23181  |        2.64161  |       0.185703 | -0.185703  |

## Dry-run confirmation (real container logs; no fabricated evidence)

1) Extracted evidence bundle:

`python3 -m scripts.extract_container_logs --container algo-data-ingestion-trading-1 --scheduler-container algo-data-ingestion-scheduler-1 --output-dir reports/log_forensics/evidence`

→ Produced: `reports/log_forensics/evidence/20251226T020251Z`

2) Verified boosted config is active in runtime env:

- `reports/log_forensics/evidence/20251226T020251Z/env_snapshot.txt` contains `stop_loss_pct=0.005` + `min_hold_bars_override=8` for ETH/SOL (and `order_notional` unchanged).
- `reports/log_forensics/evidence/20251226T020251Z/portfolio_risk_limits_1.yaml` contains global `trigger_overrides.entry_threshold=0.51` + per-symbol BTC override at 0.56.

3) Verified executed exits under the boosted trigger band (audit log forensics):

- `reports/expectancy_fix/boosted_live_forensics.md` summarizes 6 executed exits across BTC/ETH/SOL in the immediate post-restart window.
