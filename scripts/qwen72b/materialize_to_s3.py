#!/usr/bin/env python3
"""Live-gated Qwen72B materialization coordinator. No rendered worker mode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.qwen72b_fallback.aws_verifier import AwsLiveVerifier  # noqa: E402
from experiments.qwen72b_fallback.materializer import (  # noqa: E402
    build_materialization_plan,
    launch_materialization,
    terminate_orphan,
)
from experiments.qwen72b_fallback.readiness import (  # noqa: E402
    ExecutionAction,
    ReadinessState,
    evaluate_readiness,
    required_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("check", "execute", "terminate-orphan"),
        default="check",
    )
    parser.add_argument("--launch-name")
    parser.add_argument("--confirm")
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    verifier = AwsLiveVerifier.from_boto3(
        repo_root=ROOT,
        profile_name=os.environ.get("AWS_PROFILE"),
    )
    if args.mode == "terminate-orphan":
        if args.instance_id is None or args.confirm is None:
            parser.error("terminate-orphan requires --instance-id and --confirm")
        result = terminate_orphan(
            ec2=verifier.ec2,
            instance_id=args.instance_id,
            typed_confirmation=args.confirm,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.launch_name is None or args.confirm is None:
        parser.error(
            "check/execute require --launch-name and --confirm; exact text: "
            "EXECUTE QWEN72B MATERIALIZE <launch-name>"
        )
    report = evaluate_readiness(
        verifier,
        action=ExecutionAction.MATERIALIZE,
        launch_name=args.launch_name,
        profile=None,
        typed_confirmation=args.confirm,
    )
    if report.state is ReadinessState.BLOCKED:
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        print(
            "required confirmation: "
            + required_confirmation(ExecutionAction.MATERIALIZE, args.launch_name),
            file=sys.stderr,
        )
        return 3
    assert report.authorization is not None
    if args.mode == "check":
        plan = build_materialization_plan(report.authorization)
        print(
            json.dumps(
                {
                    "readiness": report.model_dump(mode="json"),
                    "plan": plan.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = launch_materialization(
        ec2=verifier.ec2,
        s3=verifier.s3,
        authorization=report.authorization,
    )
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "status": result.status,
                "termination": result.termination.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
