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
    load_manifest,
)
from experiments.aws_smoke.pins import load_evidence  # noqa: E402
from experiments.aws_smoke.profile import REQUIRED_ARMS, RunArm  # noqa: E402


def _client(profile: str, region: str):
    return boto3.Session(profile_name=profile, region_name=region).client("sagemaker")


def _load_job_map(manifests_dir: Path) -> dict[RunArm, str]:
    jobs: dict[RunArm, str] = {}
    for arm in REQUIRED_ARMS:
        meta_path = manifests_dir / f"jobmeta_{arm}.json"
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

    jobs = _load_job_map(args.manifests_dir)
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
        views = status_campaign(client, jobs)
        arm = next_resumable_arm(views, ordered_arms=REQUIRED_ARMS)
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
        manifest = load_manifest(args.manifests_dir / f"manifest_{arm}.json")
        request = build_create_training_job_request(
            manifest=manifest,
            evidence=evidence,
            arm=arm,
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
                },
                sort_keys=True,
            )
        )
        return 0

    raise SystemExit(f"unknown action {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
