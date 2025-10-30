# Launch Roadmap – Calmon Stack

_Last updated: 2025-10-30 16:05 UTC_

## Executive Summary
- Training gates remain profitable (base `final_equity 4.48`, TCN `1.05–1.33`, blender `1.84`), and the retuned Oct 1–Oct 27 2025 replay (`models/oos_replay_summary_latest.json`) now shows **minimal deployable coverage**: base logs 12 gate hits (8 trades, `final_equity 1.23`), blender fires ≈16 % of bars (5 870 toggles), while TCN manifests remain idle. Blender training now smooths gate masks over the stride (captured as `gate_smoothing_stride`) and stride‑1 sandbox runs expose the turnover bounds for future manifest tweaks.
- CI now enforces manifest/report alignment and shortlist viability, yet live-readiness hinges on retuning gating thresholds and wiring those predicates into the production API.
- Launch is gated on restoring minimal deployable coverage, packaging refreshed artifacts (including forward matrices), and finalising monitoring/fallback playbooks.

## Critical Blockers (Must Resolve Before Cutover)
- **Deployable gate retune** – Base and blender now clear the coverage hurdle; finish widening or layering the TCN manifests (or ship a fallback mode) so Oct 2025 replay yields sustainable coverage without blowing turnover limits.
- **Inference parity** – Mirror the manifest-driven gates inside the FastAPI ingestion/inference path via `training/infer.py::score_base_with_manifest`, exercising the new stride-aware batching in `predict_tcn`, and keep regression tests that replay historical batches tied to the new Prometheus gauges.
- **Fallback definition** – Document and implement the fallback hierarchy (no-RSS blender, base-only mode) that activates when gating coverage or RSS audits breach thresholds.

## High-Priority Tasks (1–2 Weeks)
- Recompute forward replay as TCN manifests evolve; update manifests, `live_gate_coverage.csv`, and `models/oos_replay_summary_latest.json` (keeping the archived `...oct_nov_2025.json` for regression) with each threshold change.
- Document the blender gate smoothing window (`gate_smoothing_stride`) alongside each artifact and quantify turnover vs coverage using the new stride‑1 sandbox runs before finalising deployable thresholds.
- Extend the regression suite with an inference replay test (fixture-driven) so CI fails if deployable thresholds drift again.
- Package release bundle: models, manifests, forward matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`), coverage CSVs, shortlist, and updated documentation.
- Harden the FastAPI service to consume manifest-defined gates (probability threshold, spread/rvol caps, min-hold) at inference time and emit Prometheus metrics for gate hits/misses via `app/monitoring/model_metrics.py`.

## Monitoring & Ops Checklist
- Add alerts for gate coverage < historical floor (±2× `live_gate_coverage.csv`), RSS minute spike share <5e-4, probability σ <0.03, and inference parity mismatches, leveraging the new Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma` + thresholds).
- Surface the recorded `gate_smoothing_stride` and stride-aware inference metrics in Grafana so operators see when smoothing changes or stride experiments deviate from the release baseline.
- Publish Grafana panels overlaying deployable vs relaxed coverage, RSS audits, and model equity curves for the Oct 2025 forward window onward.
- Ensure `.github/workflows/ci.yml` remains green and add a nightly job (cron or scheduled workflow) that reruns `pytest tests/regression` plus the new inference replay test over the latest manifests.

## Documentation & Comms
- Update `docs/final_stretch_v1.md`, `TRAINING_STATUS.md`, `TRAINING_WALKTHROUGH.md`, and README excerpts once gate tuning completes.
- Produce a short launch note summarising: final thresholds, coverage expectations, fallback triggers, and monitoring dashboards.
- Record dataset/model SHAs and upload large artifacts to the agreed storage (Git LFS or external bucket) with retention policy.

## Success Criteria for Launch Readiness
- Deployable gate delivers ≥0.5 bps coverage (or agreed minimum) on Oct 2025 replay with equity ≥1.2 across base, TCN, and blender (base/blender already there; TCN must meet the same bar or ship with a documented fallback).
- Inference path uses manifests end-to-end and passes automated regression plus manual smoke against production-like data.
- Monitoring and runbooks are published, with on-call acknowledging alert thresholds.
- Release bundle (artifacts + docs) reviewed and signed off by modeling + platform leads.
