# Live Readiness Check

Status: **GO**

Timestamp (UTC): 2025-12-13T01:30:20.291864+00:00
Mode: live_like
Deployment contract: /Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/configs/deployment_portfolio_contract.yaml
Audit log: data_lake/trading/audit.log
Lookback hours: 48

## Orchestrated checks (existing tools)
- `analysis/validate_deployment_contract.py` (contract + live invariants; enforces Redis/HMAC in live mode)
- `analysis/preflight_coverage.py` (coverage fraction + implied-trades proxy; deadlock-preflight)
- `analysis/shadow_readiness.py` + `analysis/preflight_symbol_promotion.py` (audit provenance/HMAC + promotion gates)
- `analysis/evaluate_launch_stage.py` (optional stage ladder GO/NO-GO; enforces Redis/HMAC for live stages)

## Checks
- **deployment_contract_validation**: `PASS` (required, 30 ms) — Deployment contract validated successfully.
- **coverage_preflight**: `PASS` (required, 3719 ms) — Coverage preflight passed.
  - artifact: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/preflight_coverage_20251213_0130.json`
  - artifact: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/preflight_coverage_20251213_0130.md`
- **shadow_readiness_and_promotion**: `SKIP` (optional, 0 ms) — Skipped (no shadow symbols and not required).
- **launch_stage_evaluation**: `SKIP` (optional, 0 ms) — Skipped (no --stage provided).

## Artifacts
- readiness_json: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/20251213T013020Z_live_readiness.json`
- readiness_md: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/20251213T013020Z_live_readiness.md`
- coverage_reports: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/preflight_coverage_20251213_0130.json`
- coverage_reports: `/Users/yehosuahercules/Desktop/EcuacionesDiferenciales/CCXT_Testing/algo-data-ingestion/reports/live_readiness/preflight_coverage_20251213_0130.md`
