#!/usr/bin/env python3
"""Live-gated SageMaker probe/rehearsal/full launch and verified stop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.qwen72b_fallback.aws_verifier import AwsLiveVerifier  # noqa: E402
from experiments.qwen72b_fallback.launch import (  # noqa: E402
    build_training_request,
    launch_training_job,
    stop_training_job_and_verify,
)
from experiments.qwen72b_fallback.profile import (  # noqa: E402
    full_profile,
    rehearsal_profile,
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
        choices=("check", "execute", "stop"),
        default="check",
    )
    parser.add_argument(
        "--action",
        choices=("memory_probe", "rehearsal", "full"),
    )
    parser.add_argument("--target-kind", choices=("rehearsal", "full"))
    parser.add_argument("--launch-name")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    verifier = AwsLiveVerifier.from_boto3(
        repo_root=ROOT,
        profile_name=os.environ.get("AWS_PROFILE"),
    )
    if args.mode == "stop":
        if args.launch_name is None or args.confirm is None:
            parser.error("stop requires --launch-name and --confirm")
        result = stop_training_job_and_verify(
            sagemaker=verifier.sagemaker,
            job_name=args.launch_name,
            typed_confirmation=args.confirm,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    if args.action is None or args.launch_name is None or args.confirm is None:
        parser.error("check/execute require --action, --launch-name, and --confirm")
    action = ExecutionAction(args.action)
    target_kind = args.target_kind
    if action is ExecutionAction.MEMORY_PROBE:
        if target_kind is None:
            parser.error("memory_probe requires --target-kind")
    else:
        expected_target = action.value
        if target_kind is not None and target_kind != expected_target:
            parser.error("non-probe action target-kind must match action")
        target_kind = expected_target
    profile = rehearsal_profile() if target_kind == "rehearsal" else full_profile()
    report = evaluate_readiness(
        verifier,
        action=action,
        launch_name=args.launch_name,
        profile=profile,
        typed_confirmation=args.confirm,
    )
    if report.state is ReadinessState.BLOCKED:
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        print(
            "required confirmation: " + required_confirmation(action, args.launch_name),
            file=sys.stderr,
        )
        return 3
    assert report.authorization is not None
    launch_mode = "memory_probe" if action is ExecutionAction.MEMORY_PROBE else None
    if args.mode == "check":
        request = build_training_request(
            authorization=report.authorization,
            profile=profile,
            input_prefix=f"qwen72b/inputs/{args.launch_name}",
            mode=launch_mode,
        )
        print(
            json.dumps(
                {
                    "readiness": report.model_dump(mode="json"),
                    "request": request,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = launch_training_job(
        sagemaker=verifier.sagemaker,
        s3=verifier.s3,
        authorization=report.authorization,
        profile=profile,
        mode=launch_mode,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
