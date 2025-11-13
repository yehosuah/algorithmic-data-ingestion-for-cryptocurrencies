# Launch Roadmap – Calmon Stack

_Last updated: 2025-11-13 04:43 UTC_

> Update 2025-11-13: Roadmap items now call out the sanitizer + symbol-gate workflow, parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`), and the Redis-backed trading defaults baked into `.env` so launch criteria reference the exact gates and metrics in git.

## Executive Summary
- Training gates remain profitable (base `final_equity 4.48`, TCN `1.28/3.62/1.85`, blender `4.48`), and the retuned Oct 1–Oct 28 2025 replay (`models/oos_replay_summary_latest.json`) now shows **deployable coverage across all manifests**: base logs 12 gate hits (8 trades, `final_equity 1.2336`), TCN horizons 60/120/180 clear the guardrail (`gate_coverage 4.73e-4/7.71e-4/4.23e-4` with 4/62/2 toggles), and the eased blender manifest fires ≈15.8 % of bars (6 346 toggles) while the stride‑1 sandbox variant bounds turnover at 134 toggles.
- CI now enforces manifest/report alignment, shortlist viability, and a forward replay guardrail that fails when TCN deployable coverage drops below 5e-4 or `final_equity` slips under 1.2; wiring those predicates into the production API remains critical.
- Launch is gated on keeping the new coverage floor stable, packaging refreshed artifacts (including forward matrices + manifests), and finalising monitoring/fallback playbooks.
- Scheduler-driven inference now pushes decisions into Redis and the `trading` service consumes them with dry-run order execution, Prometheus metrics, and Redis/Postgres audit trails; launch readiness requires rehearsing this loop end-to-end and capturing ops runbooks.

## Critical Blockers (Must Resolve Before Cutover)
- **Deployable gate stability** – Base, TCN, and blender now clear the coverage hurdle; keep the widened TCN manifests above the 5e-4 floor via the CI guardrail, document fallback modes, and ensure turnover stays within agreed limits as thresholds evolve.
- **Inference parity** – Mirror the manifest-driven gates inside the FastAPI ingestion/inference path via `training/infer.py::score_base_with_manifest`, exercising the new stride-aware batching in `predict_tcn`, and keep regression tests that replay historical batches tied to the new Prometheus gauges.
- **Fallback definition** – Document and implement the fallback hierarchy (no-RSS blender, base-only mode) that activates when gating coverage or RSS audits breach thresholds.
- **Trading dry-run coverage** – Validate the scheduler `INFER_JOBS` → Redis → trading service loop, ensuring queue depth stays bounded, audit streams populate, and metrics (`trading_trade_attempts_total`, `trading_position_active`, `trading_realized_pnl_total`) remain healthy before considering live order routing.

## High-Priority Tasks (1–2 Weeks)
- Recompute forward replay as TCN manifests evolve; update manifests, `live_gate_coverage.csv`, and `models/oos_replay_summary_latest.json` (keeping the archived `...oct_nov_2025.json` for regression) with each threshold change.
- Keep `release/symbol_gates/*.json` in lockstep with every sanitized dataset refresh (run `scripts/compute_symbol_gate_config.py`) so retrains, scheduler jobs, and `TRADING_MODELS` consume identical per-symbol caps.
- Document the blender gate smoothing window (`gate_smoothing_stride`) alongside each artifact and quantify turnover vs coverage using the new stride‑1 sandbox runs before finalising deployable thresholds.
- Extend the regression suite with an inference replay test (fixture-driven) so CI fails if deployable thresholds drift again.
- Package release bundle: models, manifests, forward matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`), coverage CSVs, shortlist, and updated documentation.
- Harden the FastAPI service to consume manifest-defined gates (probability threshold, spread/rvol caps, min-hold) at inference time and emit Prometheus metrics for gate hits/misses via `app/monitoring/model_metrics.py`.
- Stand up the scheduler `INFER_JOBS` + trading dry-run loop in staging, document Redis/Postgres state backends, and capture Grafana screenshots (`trading-overview`, `scheduler-overview`) showing steady gate coverage and bounded queue depth.
- Automate the feature parity workflow: export Redis slices via `scripts/export_feature_slice.py`, diff against the sanitized training parquet with `scripts/compare_feature_stats.py`, and archive the JSON (e.g., `release/calibration/latest/feature_parity.json`) per rehearsal so threshold changes cite concrete drift stats.

## Monitoring & Ops Checklist
- Add alerts for gate coverage < historical floor (±2× `live_gate_coverage.csv`), RSS minute spike share <5e-4, probability σ <0.03, and inference parity mismatches, leveraging the new Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma` + thresholds).
- Mirror the parity JSON payloads (training vs live `hl_spread`, `hl_spread_z`, `rvol_20`, `base_prob`) on Grafana so operators can correlate alert noise with concrete drift observed by `scripts/compare_feature_stats.py`.
- Surface the recorded `gate_smoothing_stride` and stride-aware inference metrics in Grafana so operators see when smoothing changes or stride experiments deviate from the release baseline.
- Publish Grafana panels overlaying deployable vs relaxed coverage, RSS audits, and model equity curves for the Oct 2025 forward window onward.
- Ensure `.github/workflows/ci.yml` remains green and add a nightly job (cron or scheduled workflow) that reruns `pytest tests/regression` plus the new inference replay test over the latest manifests.
- Add trading-specific alerts: stale Redis decision queue, zero `trading_trade_attempts_total` for >15 min, `trading_position_active` stuck > max hold window, and missing audit events; wire them to the new dashboard and include runbook links.

## Documentation & Comms
- Update `docs/final_stretch_v1.md`, `TRAINING_STATUS.md`, `TRAINING_WALKTHROUGH.md`, and README excerpts once gate tuning completes.
- Produce a short launch note summarising: final thresholds, coverage expectations, fallback triggers, and monitoring dashboards.
- Record dataset/model SHAs and upload large artifacts to the agreed storage (Git LFS or external bucket) with retention policy.

## Success Criteria for Launch Readiness
- Deployable gate delivers ≥0.5 bps coverage (or agreed minimum) on Oct 2025 replay with equity ≥1.2 across base, TCN, and blender (base/blender already there; TCN must meet the same bar or ship with a documented fallback).
- Inference path uses manifests end-to-end and passes automated regression plus manual smoke against production-like data.
- Monitoring and runbooks are published, with on-call acknowledging alert thresholds.
- Release bundle (artifacts + docs) reviewed and signed off by modeling + platform leads.
- Scheduler → trading dry run exhibits bounded queue depth, audit coverage, and Prometheus metrics for the configured symbol set over a continuous 7-day rehearsal, with documented go/no-go criteria before enabling live orders (`TRADING_DRY_RUN=0`).
