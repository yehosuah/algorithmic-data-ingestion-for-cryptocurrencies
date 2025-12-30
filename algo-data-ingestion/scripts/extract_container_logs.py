"""
Extract trading dry-run evidence from a running container.

Usage:
  python -m scripts.extract_container_logs --container algo-data-ingestion-trading-1 \
      --output-dir reports/log_forensics/evidence [--scheduler-container algo-data-ingestion-scheduler-1]

Outputs a timestamped folder containing:
  - audit logs (and rotations if present)
  - docker-compose resolved config
  - deployment contract, risk limits, trigger policy, deadlock policy
  - active manifest(s) referenced by the contract
  - env snapshot (non-secret keys)
  - metrics snapshots (trading/scheduler if reachable)
  - optional scheduler logs (stdout tail)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import yaml
from urllib.request import urlopen
from urllib.error import URLError


REDACT_KEYS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "HMAC", "API", "CREDENTIAL", "BEARER")


def _run(cmd: List[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    return proc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _container_path_exists(container: str, path: str) -> bool:
    try:
        cmd = ["docker", "exec", container, "sh", "-c", f"test -e {shlex.quote(path)} && echo 1 || echo 0"]
        out = _run(cmd, check=True).stdout.strip()
        return out == "1"
    except subprocess.CalledProcessError:
        return False


def _copy_from_container(container: str, src: str, dest: Path) -> bool:
    if not _container_path_exists(container, src):
        return False
    _ensure_dir(dest.parent)
    try:
        _run(["docker", "cp", f"{container}:{src}", str(dest)], check=True)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"[warn] failed to copy {src}: {exc}\n")
        return False


def _write_text(dest: Path, content: str) -> None:
    _ensure_dir(dest.parent)
    dest.write_text(content)


def _fetch_metrics(url: str, dest: Path) -> None:
    try:
        with urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            _write_text(dest, body)
    except (URLError, OSError, ValueError) as exc:
        _write_text(dest, f"# failed to fetch metrics from {url}: {exc}\n")
    except Exception as exc:  # pragma: no cover - defensive
        _write_text(dest, f"# failed to fetch metrics from {url}: {exc}\n")


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _gather_manifest_paths(contract_path: Path) -> List[Path]:
    contract = _load_yaml(contract_path)
    models_root = Path(contract.get("models_root") or "/opt/models")
    manifests: List[Path] = []
    for _, rel_path in (contract.get("models") or {}).items():
        rel = str(rel_path)
        candidates = [models_root / rel / "manifest.json"]
        if rel.startswith("experiments/"):
            candidates.append(models_root / rel.split("experiments/", 1)[1] / "manifest.json")
        for cand in candidates:
            manifests.append(cand)
    return manifests


def _normalize_paths(raw_path: str, repo_root: Path, *, app_root: str = "/app") -> List[str]:
    candidates = []
    if not raw_path:
        return candidates
    candidates.append(raw_path)
    try:
        rp = Path(raw_path)
        if not rp.is_absolute():
            candidates.append(str(Path(app_root) / rp))
    except Exception:
        pass
    try:
        rp = Path(raw_path)
        if rp.is_absolute() and str(rp).startswith(str(repo_root)):
            rel = rp.relative_to(repo_root)
            candidates.append(str(Path(app_root) / rel))
    except Exception:
        pass
    if "experiments/" in raw_path:
        candidates.append(raw_path.replace("experiments/", "", 1))
    return list(dict.fromkeys(candidates))


def _gather_risk_paths(contract_path: Path, env_map: dict, repo_root: Path) -> List[str]:
    contract = _load_yaml(contract_path)
    paths = []
    risk_contract = contract.get("risk_limits")
    if isinstance(risk_contract, str):
        paths.extend(_normalize_paths(risk_contract, repo_root))
    env_risk = env_map.get("TRADING_RISK_LIMITS_PATH")
    if env_risk:
        paths.extend(_normalize_paths(env_risk, repo_root))
    return list(dict.fromkeys(paths))


def _filter_env(env_list: Iterable[str]) -> List[str]:
    cleaned = []
    for item in env_list:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        upper = key.upper()
        if any(marker in upper for marker in REDACT_KEYS):
            continue
        cleaned.append(f"{key}={value}")
    return cleaned


def _container_env(container: str) -> dict:
    cmd = ["docker", "inspect", container, "--format", "{{json .Config.Env}}"]
    proc = _run(cmd, check=True)
    try:
        env_list = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env_list = []
    env_map = {}
    for item in env_list:
        if "=" not in item:
            continue
        k, _, v = item.partition("=")
        env_map[k] = v
    return env_map


def _copy_audit_logs(container: str, dest_dir: Path) -> List[Path]:
    copied: List[Path] = []
    listing_cmd = [
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        "ls /app/data_lake/trading/audit.log* 2>/dev/null",
    ]
    try:
        ls_out = _run(listing_cmd, check=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        ls_out = []
    for line in ls_out:
        name = Path(line).name
        dest = dest_dir / name
        if _copy_from_container(container, line, dest):
            copied.append(dest)
    return copied


def _copy_optional(container: str, src: str, dest: Path, label: str) -> None:
    if _copy_from_container(container, src, dest):
        print(f"[ok] copied {label}: {src} -> {dest}")
    else:
        print(f"[skip] {label} not found: {src}")


def _copy_optional_candidates(container: str, candidates: List[str], dest: Path, label: str) -> None:
    for cand in candidates:
        if _copy_from_container(container, cand, dest):
            print(f"[ok] copied {label}: {cand} -> {dest}")
            return
    if candidates:
        print(f"[skip] {label} not found (tried: {', '.join(candidates)})")
    else:
        print(f"[skip] {label} not found: <no candidates>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract trading dry-run evidence bundle from a running container.")
    parser.add_argument("--container", required=True, help="Trading container name (e.g., algo-data-ingestion-trading-1)")
    parser.add_argument("--scheduler-container", default=None, help="Scheduler container name (optional)")
    parser.add_argument("--output-dir", default="reports/log_forensics/evidence", help="Base output directory")
    parser.add_argument("--timestamp", default=None, help="Override timestamp suffix (default: UTC now)")
    parser.add_argument(
        "--contract-path",
        default="/app/configs/deployment_portfolio_contract.yaml",
        help="Path to deployment contract inside the container",
    )
    args = parser.parse_args()

    ts = args.timestamp or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_out = Path(args.output_dir).expanduser().resolve() / ts
    _ensure_dir(base_out)
    repo_root = Path(__file__).resolve().parents[1]

    print(f"[info] writing evidence to {base_out}")

    # Resolved docker-compose config
    try:
        compose_out = _run(["docker", "compose", "config"], cwd=repo_root, check=True)
        _write_text(base_out / "docker_compose_resolved.txt", compose_out.stdout)
        print("[ok] captured docker compose config")
    except subprocess.CalledProcessError as exc:
        _write_text(base_out / "docker_compose_resolved.txt", f"# failed to resolve compose: {exc}\n{exc.stderr}")
        print("[warn] failed to capture docker compose config")

    # Environment snapshot
    env_map = _container_env(args.container)
    env_lines = _filter_env([f"{k}={v}" for k, v in env_map.items()])
    _write_text(base_out / "env_snapshot.txt", "\n".join(env_lines))
    print("[ok] captured env snapshot (non-secret keys)")

    # Deployment contract
    contract_dest = base_out / "deployment_contract.yaml"
    contract_candidates = _normalize_paths(args.contract_path, repo_root)
    _copy_optional_candidates(args.container, contract_candidates, contract_dest, "deployment contract")

    # Risk limits (from contract/env)
    risk_paths = _gather_risk_paths(contract_dest, env_map, repo_root)
    for idx, risk_path in enumerate(risk_paths):
        dest_name = "portfolio_risk_limits.yaml" if idx == 0 else f"portfolio_risk_limits_{idx}.yaml"
        _copy_optional(args.container, risk_path, base_out / dest_name, "risk limits")

    # Trigger / deadlock configs
    _copy_optional(args.container, "/app/configs/final_trigger_policy.yaml", base_out / "final_trigger_policy.yaml", "final trigger policy")
    deadlock_candidates = _normalize_paths(
        env_map.get("TRADING_DEADLOCK_POLICY_PATH", "/app/configs/runtime_overrides/deadlock_policy_stage_0.yaml"),
        repo_root,
    )
    _copy_optional_candidates(args.container, deadlock_candidates, base_out / "deadlock_policy.yaml", "deadlock policy")

    # Manifest(s)
    for manifest_path in _gather_manifest_paths(contract_dest):
        dest = base_out / "manifest" / manifest_path.name
        _copy_optional(args.container, str(manifest_path), dest, f"manifest {manifest_path}")

    # Audit logs
    audit_dir = base_out / "trading_audit"
    copied_audits = _copy_audit_logs(args.container, audit_dir)
    if copied_audits:
        print(f"[ok] copied audit logs: {len(copied_audits)} file(s)")
    else:
        print("[warn] no audit logs found to copy")

    # Trading state snapshot (optional)
    _copy_optional(args.container, "/app/trading_state/state.json", base_out / "trading_state.json", "trading state")

    # Scheduler logs (tail)
    scheduler_container = args.scheduler_container
    if scheduler_container is None:
        # best-effort auto-detect scheduler container
        try:
            names = _run(["docker", "ps", "--format", "{{.Names}}"]).stdout.strip().splitlines()
            for name in names:
                if "scheduler" in name:
                    scheduler_container = name
                    break
        except subprocess.CalledProcessError:
            scheduler_container = None
    if scheduler_container:
        try:
            logs = _run(["docker", "logs", "--tail", "2000", scheduler_container], check=True).stdout
            _write_text(base_out / "scheduler_logs.txt", logs)
            print(f"[ok] captured scheduler logs from {scheduler_container}")
        except subprocess.CalledProcessError as exc:
            _write_text(base_out / "scheduler_logs.txt", f"# failed to fetch logs: {exc}\n{exc.stderr}")
            print(f"[warn] failed to capture scheduler logs from {scheduler_container}")
    else:
        print("[info] scheduler container not provided and not auto-detected; skipping scheduler logs")

    # Metrics snapshots
    _fetch_metrics("http://localhost:9010/metrics", base_out / "trading_metrics.txt")
    _fetch_metrics("http://localhost:9002/metrics", base_out / "scheduler_metrics.txt")
    print("[ok] captured metrics snapshots (best effort)")

    print("[done] evidence bundle ready")


if __name__ == "__main__":
    main()
