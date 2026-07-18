#!/usr/bin/env python3
"""Plan the 3-step 72B QLoRA rehearsal. Real submit is gate-blocked."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.qwen72b_fallback.cost import build_cost_artifact  # noqa: E402
from experiments.qwen72b_fallback.pins import (  # noqa: E402
    SNAPSHOT_S3_URI,
    fallback_role_binding,
    teacher_role_binding,
)
from experiments.qwen72b_fallback.profile import rehearsal_profile  # noqa: E402
from experiments.qwen72b_fallback.synthetic_finance import (  # noqa: E402
    corpus_sha256,
    precompute_trajectory_stub,
    rehearsal_corpus,
)


def build_rehearsal_plan() -> dict:
    profile = rehearsal_profile()
    rows = rehearsal_corpus()
    trajectories = precompute_trajectory_stub(rows)
    cost = build_cost_artifact(
        kind="rehearsal",
        max_runtime_seconds=profile.max_runtime_seconds,
        hourly_usd=profile.hourly_usd,
        price_source=profile.price_source,
        hard_cap_usd=profile.hard_cap_usd,
        instance_type=profile.instance_type,
    )
    return {
        "schema_version": "distillery.qwen72b_fallback.rehearsal_plan.v1",
        "mode_default": "plan",
        "optimizer_steps": 3,
        "profile": profile.model_dump(mode="json"),
        "memory_plan": profile.memory_plan(),
        "roles": {
            "teacher_for_tinyfable": teacher_role_binding().model_dump(mode="json"),
            "oracle_sft_adapted_fallback": fallback_role_binding().model_dump(mode="json"),
        },
        "snapshot_s3_uri": SNAPSHOT_S3_URI,
        "oracle_corpus_sha256": corpus_sha256(rows),
        "trajectories": trajectories,
        "cost": cost,
        "submit_requires": [
            "check_gates.py --action rehearsal may_execute=true",
            "no active p4de/g5/14b work",
            "digest-pinned ECR training image",
            "complete S3 snapshot with SHA256SUMS",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="launch_rehearsal")
    parser.add_argument("--mode", choices=("plan",), default="plan")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "rehearsal execute refused; gates currently require S3 snapshot + "
                        "ECR image + idle transfer plane"
                    ),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(build_rehearsal_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
