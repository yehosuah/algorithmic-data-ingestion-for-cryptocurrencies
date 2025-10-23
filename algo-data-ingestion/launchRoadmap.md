# Launch Roadmap – Calmon Stack

_Last updated: 2025-10-23 01:00 UTC_

## Executive Summary
- Training gates remain profitable (base `final_equity 4.48`, TCN `1.05–1.33`, blender `1.84`), but Oct 1–Oct 21 2025 replay shows the deployable mask producing **zero trades across all models** (`models/oos_replay_oct_nov_2025.json`).
- CI now enforces manifest/report alignment and shortlist viability, yet live-readiness hinges on retuning gating thresholds and wiring those predicates into the production API.
- Launch is gated on restoring minimal deployable coverage, packaging refreshed artifacts (including forward matrices), and finalising monitoring/fallback playbooks.

## Critical Blockers (Must Resolve Before Cutover)
- **Deployable gate retune** – Adjust `hl_spread`, `rvol20`, and `prob` thresholds (or add layered gating) so Oct 2025 forward replay yields non-zero coverage while respecting turnover limits for base, TCN, and blender.
- **Inference parity** – Mirror the updated gate predicates inside the FastAPI ingestion/inference path and add regression tests that replay historical batches through `training/infer.py`.
- **Fallback definition** – Document and implement the fallback hierarchy (no-RSS blender, base-only mode) that activates when gating coverage or RSS audits breach thresholds.

## High-Priority Tasks (1–2 Weeks)
- Recompute forward replay after gate tuning; update manifests, `live_gate_coverage.csv`, and `models/oos_replay_oct_nov_2025.json` with the new thresholds.
- Extend the regression suite with an inference replay test (fixture-driven) so CI fails if deployable thresholds drift again.
- Package release bundle: models, manifests, forward matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`), coverage CSVs, shortlist, and updated documentation.
- Harden the FastAPI service to consume manifest-defined gates (probability threshold, spread/rvol caps, min-hold) at inference time and emit Prometheus metrics for gate hits/misses.

## Monitoring & Ops Checklist
- Add alerts for gate coverage < historical floor (±2× `live_gate_coverage.csv`), RSS minute spike share <5e-4, probability σ <0.03, and inference parity mismatches.
- Publish Grafana panels overlaying deployable vs relaxed coverage, RSS audits, and model equity curves for the Oct 2025 forward window onward.
- Ensure `.github/workflows/ci.yml` remains green and add a nightly job (cron or scheduled workflow) that reruns `pytest tests/regression` plus the new inference replay test over the latest manifests.

## Documentation & Comms
- Update `docs/final_stretch_v1.md`, `TRAINING_STATUS.md`, `TRAINING_WALKTHROUGH.md`, and README excerpts once gate tuning completes.
- Produce a short launch note summarising: final thresholds, coverage expectations, fallback triggers, and monitoring dashboards.
- Record dataset/model SHAs and upload large artifacts to the agreed storage (Git LFS or external bucket) with retention policy.

## Success Criteria for Launch Readiness
- Deployable gate delivers ≥0.5 bps coverage (or agreed minimum) on Oct 2025 replay with equity ≥1.2 across base, TCN, and blender.
- Inference path uses manifests end-to-end and passes automated regression plus manual smoke against production-like data.
- Monitoring and runbooks are published, with on-call acknowledging alert thresholds.
- Release bundle (artifacts + docs) reviewed and signed off by modeling + platform leads.

