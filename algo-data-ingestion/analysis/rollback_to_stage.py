from __future__ import annotations

import argparse
from pathlib import Path

from analysis.apply_launch_stage import _apply_stage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rollback trading config to a specific launch ladder stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stage", default="stage_0", help="Target rollback stage (default: stage_0).")
    ap.add_argument("--ladder", default="configs/live_launch_ladder.yaml", help="Launch ladder config path.")
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml", help="Deployment contract path.")
    ap.add_argument(
        "--runtime-overrides-dir",
        default="configs/runtime_overrides",
        help="Runtime overrides directory used by apply_launch_stage.",
    )
    args = ap.parse_args(argv)

    summary = _apply_stage(
        stage_name=args.stage,
        ladder_path=Path(args.ladder),
        contract_path=Path(args.contract),
        runtime_overrides=Path(args.runtime_overrides_dir),
    )

    print(f"Rollback applied to {summary['contract_path']} using stage {summary['stage']}.")
    print("Immediate operator actions:")
    print("1) Set kill switch ON to stop new entries: export TRADING_KILL_SWITCH=1")
    print("2) Keep safe mode latched for exits-only drains: export TRADING_SAFE_MODE=1")
    print("3) Point runtime to the stage overrides if not already loaded:")
    print(f"   - TRADING_RISK_LIMITS_PATH={summary['risk_limits_path']}")
    print(f"   - TRADING_DEADLOCK_POLICY_PATH={summary['deadlock_policy_path']}")
    print(f"   - stage patch (env bundle): {summary['patch_path']}")
    print("4) Restart trading service so kill switch/safe mode and overrides take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
