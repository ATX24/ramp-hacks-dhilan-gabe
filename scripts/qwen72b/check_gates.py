#!/usr/bin/env python3
"""Run live 72B gates; blocked reports exit nonzero and issue no latch."""

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
        "--action",
        choices=tuple(action.value for action in ExecutionAction),
        required=True,
    )
    parser.add_argument("--launch-name", required=True)
    parser.add_argument(
        "--target-kind",
        choices=("rehearsal", "full"),
        default="rehearsal",
    )
    parser.add_argument("--confirm")
    parser.add_argument("--authorization-out", type=Path)
    args = parser.parse_args()
    action = ExecutionAction(args.action)
    profile = None
    if action is not ExecutionAction.MATERIALIZE:
        profile = rehearsal_profile() if args.target_kind == "rehearsal" else full_profile()
    verifier = AwsLiveVerifier.from_boto3(
        repo_root=ROOT,
        profile_name=os.environ.get("AWS_PROFILE"),
    )
    report = evaluate_readiness(
        verifier,
        action=action,
        launch_name=args.launch_name,
        profile=profile,
        typed_confirmation=args.confirm,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.state is ReadinessState.BLOCKED:
        if args.confirm is None:
            print(
                "required confirmation: " + required_confirmation(action, args.launch_name),
                file=sys.stderr,
            )
        return 3
    if args.authorization_out is not None:
        assert report.authorization is not None
        args.authorization_out.parent.mkdir(parents=True, exist_ok=True)
        args.authorization_out.write_text(
            json.dumps(
                report.authorization.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
