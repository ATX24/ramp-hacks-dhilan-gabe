#!/usr/bin/env python3
"""Serial dry-run planner / gated launcher for 3 emergency arms (quota=1).

Default mode is dry-run and never calls CreateTrainingJob. Mutation requires
--confirm I_CONFIRM_SAGEMAKER_SUBMIT, profile gabriel-cli, and non-root identity.
This script still refuses to call AWS unless --execute is passed; even then the
actual boto3 submit path is only enabled when DISTILLERY_AWS_SMOKE_EXECUTE=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.aws_smoke.launch_plan import (  # noqa: E402
    discover_generated_manifest_paths,
    plan_serial_launch,
    plan_to_dict,
    stage_manifest_for_job,
)
from experiments.aws_smoke.pins import load_evidence  # noqa: E402
from experiments.aws_smoke.profile import (  # noqa: E402
    DEFAULT_UNIQUE_LAUNCH_ORDER,
)
from experiments.aws_smoke.safety import CONFIRM_PHRASE, CallerIdentity  # noqa: E402


def _sts_identity(profile: str, region: str) -> CallerIdentity:
    session = boto3.Session(profile_name=profile, region_name=region)
    sts = session.client("sts")
    raw = sts.get_caller_identity()
    return CallerIdentity(
        account=str(raw["Account"]),
        arn=str(raw["Arn"]),
        user_id=str(raw["UserId"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="launch_serial")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--profile", type=str, default="gabriel-cli")
    parser.add_argument("--confirm", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Attempt CreateTrainingJob (also requires DISTILLERY_AWS_SMOKE_EXECUTE=1)",
    )
    parser.add_argument("--plan-out", type=Path, default=None)
    parser.add_argument(
        "--allow-control-as-third",
        action="store_true",
        help=(
            "Allow oracle_sft-equivalent ce_ablation as a third job when "
            "sequence_kd evidence is absent; output remains only two distinct signals"
        ),
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Append ce_ablation after the three distinct default signals",
    )
    args = parser.parse_args()

    evidence = load_evidence(args.evidence)
    manifest_paths = discover_generated_manifest_paths(args.manifests_dir)

    dry_run = bool(args.dry_run) and not bool(args.execute)

    def identity_provider() -> CallerIdentity:
        return _sts_identity(args.profile, evidence.aws_region)

    # Dry-run is strictly local. Execution resolves STS for the non-root gate.
    provider = None if dry_run else identity_provider

    explicit_arms = None
    require_three_distinct = not args.allow_control_as_third
    if args.include_control and "sequence_kd" in manifest_paths:
        explicit_arms = DEFAULT_UNIQUE_LAUNCH_ORDER + ("ce_ablation",)
    plan = plan_serial_launch(
        manifest_paths=manifest_paths,
        evidence=evidence,
        profile_name=args.profile,
        confirm=args.confirm,
        dry_run=dry_run,
        identity_provider=provider,
        arms=explicit_arms,
        require_three_distinct=require_three_distinct,
    )
    payload = plan_to_dict(plan)
    if args.plan_out is not None:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if dry_run:
        print(json.dumps({"ok": True, "mode": "dry_run", "plan": payload}, sort_keys=True))
        return 0

    if os.environ.get("DISTILLERY_AWS_SMOKE_EXECUTE", "") != "1":
        raise SystemExit(
            "refusing CreateTrainingJob: set DISTILLERY_AWS_SMOKE_EXECUTE=1 after gates pass"
        )
    if args.confirm != CONFIRM_PHRASE:
        raise SystemExit(f"missing confirmation phrase {CONFIRM_PHRASE}")

    # Intentionally left as an explicit opt-in mutation path for operators.
    session = boto3.Session(profile_name=args.profile, region_name=evidence.aws_region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")
    submitted: list[dict[str, Any]] = []
    for job in plan.jobs:
        # Serial: submit first only; operator resumes remaining via jobctl.
        if submitted:
            break
        staged_uri = stage_manifest_for_job(
            s3,
            local_manifest_path=job.local_manifest_path,
            request=job.create_training_job_request,
        )
        sm.create_training_job(**job.create_training_job_request)
        submitted.append(
            {
                "arm": job.arm,
                "job_name": job.job_name,
                "manifest_object_uri": staged_uri,
            }
        )
    print(
        json.dumps(
            {
                "ok": True,
                "mode": "execute_first_only",
                "submitted": submitted,
                "remaining_arms": [j.arm for j in plan.jobs[len(submitted) :]],
                "note": "quota=1; wait for terminal then resume via jobctl.py",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
