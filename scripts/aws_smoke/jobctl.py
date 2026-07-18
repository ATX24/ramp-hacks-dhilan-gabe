#!/usr/bin/env python3
"""Resume / stop / status for emergency aws-smoke Training Jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.aws_smoke.job_control import (  # noqa: E402
    inventory_smoke_jobs,
    next_resumable_arm,
    status_campaign,
    status_for_job,
    stop_job,
)
from experiments.aws_smoke.launch_plan import (  # noqa: E402
    build_create_training_job_request,
    discover_generated_manifest_paths,
    load_manifest,
    stage_manifest_for_job,
)
from experiments.aws_smoke.pins import load_evidence  # noqa: E402
from experiments.aws_smoke.profile import (  # noqa: E402
    DEFAULT_UNIQUE_LAUNCH_ORDER,
    RunArm,
    default_launch_order,
)
from experiments.aws_smoke.safety import (  # noqa: E402
    CONFIRM_PHRASE,
    CallerIdentity,
    enforce_safety_gates,
)


def _client(profile: str, region: str):
    return boto3.Session(profile_name=profile, region_name=region).client("sagemaker")


def _s3_client(profile: str, region: str):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")


def _identity(profile: str, region: str) -> CallerIdentity:
    raw = boto3.Session(profile_name=profile, region_name=region).client(
        "sts"
    ).get_caller_identity()
    return CallerIdentity(
        account=str(raw["Account"]),
        arn=str(raw["Arn"]),
        user_id=str(raw["UserId"]),
    )


def _load_job_map(
    manifest_paths: dict[RunArm, Path],
) -> dict[RunArm, str]:
    jobs: dict[RunArm, str] = {}
    for arm, manifest_path in manifest_paths.items():
        meta_path = manifest_path.parent.parent / "jobmeta.json"
        if not meta_path.is_file():
            raise SystemExit(f"missing job meta for {arm}: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        jobs[arm] = str(meta["job_name"])
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(prog="jobctl")
    parser.add_argument("action", choices=["status", "stop", "resume", "inventory"])
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--profile", type=str, default="gabriel-cli")
    parser.add_argument("--arm", type=str, default=None)
    parser.add_argument("--confirm", type=str, default=None)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Required for resume/stop mutations",
    )
    args = parser.parse_args()

    if args.action == "inventory":
        if args.evidence is None:
            raise SystemExit("--evidence required for inventory")
        evidence = load_evidence(args.evidence)
        client = _client(args.profile, evidence.aws_region)
        names = inventory_smoke_jobs(client)
        print(json.dumps({"jobs": names}, sort_keys=True))
        return 0

    manifest_paths = discover_generated_manifest_paths(args.manifests_dir)
    jobs = _load_job_map(manifest_paths)
    ordered_arms = default_launch_order(
        set(manifest_paths),
        require_three_distinct=False,
    )
    if args.include_control and "sequence_kd" in manifest_paths:
        ordered_arms = DEFAULT_UNIQUE_LAUNCH_ORDER + ("ce_ablation",)
    if args.evidence is None:
        raise SystemExit("--evidence required")
    evidence = load_evidence(args.evidence)
    client = _client(args.profile, evidence.aws_region)

    if args.action == "status":
        views = status_campaign(client, jobs)
        print(
            json.dumps(
                {
                    arm: {
                        "job_name": view.job_name,
                        "status": view.status,
                        "secondary_status": view.secondary_status,
                        "terminal": view.terminal,
                        "failure_reason": view.failure_reason,
                    }
                    for arm, view in views.items()
                },
                sort_keys=True,
            )
        )
        return 0

    if args.action == "stop":
        if not args.execute:
            raise SystemExit("refusing stop without --execute")
        target_arm = args.arm
        if target_arm is None:
            raise SystemExit("--arm required for stop")
        view = stop_job(client, jobs[target_arm])  # type: ignore[index]
        print(
            json.dumps(
                {
                    "arm": target_arm,
                    "job_name": view.job_name,
                    "status": view.status,
                    "terminal": view.terminal,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.action == "resume":
        if not args.execute:
            raise SystemExit("refusing resume without --execute")
        enforce_safety_gates(
            profile=args.profile,
            confirm=args.confirm,
            evidence=evidence,
            identity_provider=lambda: _identity(args.profile, evidence.aws_region),
            dry_run=False,
        )
        if args.confirm != CONFIRM_PHRASE:
            raise SystemExit(f"missing confirmation phrase {CONFIRM_PHRASE}")
        views = status_campaign(client, jobs)
        arm = next_resumable_arm(views, ordered_arms=ordered_arms)
        if arm is None:
            print(json.dumps({"ok": True, "action": "wait", "reason": "active_or_done"}))
            return 0
        existing = status_for_job(client, jobs[arm], arm=arm, allow_missing=True)
        if existing.exists:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "action": "already_exists",
                        "arm": arm,
                        "job_name": jobs[arm],
                        "status": existing.status,
                    },
                    sort_keys=True,
                )
            )
            return 0
        manifest = load_manifest(manifest_paths[arm])
        request = build_create_training_job_request(
            manifest=manifest,
            evidence=evidence,
            arm=arm,
        )
        staged_uri = stage_manifest_for_job(
            _s3_client(args.profile, evidence.aws_region),
            local_manifest_path=manifest_paths[arm],
            request=request,
        )
        try:
            client.create_training_job(**request)
        except ClientError:
            raise
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "submitted",
                    "arm": arm,
                    "job_name": request["TrainingJobName"],
                    "manifest_object_uri": staged_uri,
                },
                sort_keys=True,
            )
        )
        return 0

    raise SystemExit(f"unknown action {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
