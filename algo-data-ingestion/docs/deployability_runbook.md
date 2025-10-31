# Deployability Runbook (TCN Suite Refresh – Oct 2025)

_Last updated: 2025-10-31 02:39 UTC_

## Immediate Follow-Ups
- **Refresh shortlist** – `python3 scripts/report_shortlist.py --models-root models --out models/report_shortlist.json` to surface the updated TCN manifests alongside base/blender.
- **Assemble release bundle** – Package manifests, `report.json`, `tcn_gate_replay_summary.json`, `oos_replay_summary_latest.json`, `live_gate_coverage.csv`, and the forward matrix parquet per `docs/final_stretch_v1.md`.
- **CI gate coverage check** – GitHub Actions already runs `scripts/run_oos_eval.py --family tcn --stride 30` for h60/h120/h180; keep the thresholds strict (`gate_coverage < 5e-4` or `final_equity < 1.2` triggers failure) and mirror the guardrail locally before adjusting manifests.
- **Documentation touch-up** – Update `docs/oct_2025_forward_replay.md` and `TRAINING_STATUS.md` with the new TCN gate thresholds/coverage before circulating status reports.
- **Monitoring baseline update** – Append Oct 2025 figures to `live_gate_coverage.csv` so alert thresholds reflect the relaxed yet deployable gates.

## Connecting to Live Trading
- **Manifest enforcement in inference**  
  - Load manifests with `training.infer.load_manifest_artifacts` at service startup.  
  - Apply `training.infer.apply_manifest_gates` after scoring to emit `gate_pass` alongside probabilities.
- **Scheduled inference pipeline**  
  - Configure the scheduler to run minute-level jobs that pull fresh features, compute TCN/base probabilities (stride 30), apply manifests, and push decisions to a queue (Redis/Kafka).
- **Execution layer integration**  
  - Extend the trading microservice to subscribe to the decision queue, enforce `min_hold_bars`, and route qualified signals through the CCXT exchange adapter.  
  - Persist per-symbol state (positions, hold timers) to survive restarts.
- **Monitoring & alerting**  
  - Export `model_gate_coverage_ratio`, `model_probability_sigma`, and `model_rss_minute_spike_share` via Prometheus.  
  - Update `monitoring/alert.rules.yml` to watch the new coverage floor (~5 e‑4) and probability σ guardrails.  
  - Publish Grafana panels for gate coverage, turnover, P&L, and order execution latency.
- **Dry-run then cutover**  
  - Execute a full paper-trading rehearsal, capturing coverage/toggle stats and verifying latency.  
  - Once validated, enable production credentials, keep heightened alerting for the first week, and prepare rollback scripts (e.g., revert to base-only mode) in case coverage collapses.
