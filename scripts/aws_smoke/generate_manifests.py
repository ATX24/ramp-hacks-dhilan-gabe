#!/usr/bin/env python3
"""Generate emergency subset + sealed per-arm manifests (no AWS calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.aws_smoke.dataset_subset import (  # noqa: E402
    materialize_emergency_subset,
)
from experiments.aws_smoke.manifests import write_arm_manifests  # noqa: E402
from experiments.aws_smoke.pins import (  # noqa: E402
    evidence_schema_template,
    load_evidence,
)
from experiments.aws_smoke.profile import (  # noqa: E402
    DEFAULT_EMERGENCY_PROFILE,
    REQUIRED_ARMS,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="generate_manifests")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write-evidence-template",
        type=Path,
        default=None,
        help="Write UNSET evidence schema template and exit",
    )
    parser.add_argument("--dataset-id", type=str, default="ds_awssmoke01")
    args = parser.parse_args()

    if args.write_evidence_template is not None:
        args.write_evidence_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_evidence_template.write_text(
            json.dumps(evidence_schema_template(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "template": str(args.write_evidence_template)}))
        return 0

    evidence = load_evidence(args.evidence)
    profile = DEFAULT_EMERGENCY_PROFILE
    subset_dir = args.output_dir / "subset"
    subset = materialize_emergency_subset(subset_dir, profile=profile)
    if (
        evidence.data_content_sha256 is not None
        and evidence.data_content_sha256 != subset.content_sha256
    ):
        raise SystemExit(
            "evidence.data_content_sha256 does not match generated subset content hash"
        )

    train_rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    example_ids = [str(row["example_id"]) for row in train_rows]
    tasks = [str(row["task"]) for row in train_rows]
    difficulties = [str(row["difficulty"]) for row in train_rows]

    paths = write_arm_manifests(
        output_dir=args.output_dir / "manifests",
        evidence=evidence,
        dataset_id=args.dataset_id,
        dataset_uri=evidence.dataset_s3_uri,
        dataset_sha256=subset.content_sha256,
        split_sha256=subset.split_sha256,
        example_ids=example_ids,
        tasks=tasks,
        difficulties=difficulties,
        arms=REQUIRED_ARMS,
        profile=profile,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "subset_content_sha256": subset.content_sha256,
                "manifests": {arm: str(path) for arm, path in paths.items()},
                "max_run_usd": profile.max_run_usd,
                "max_runtime_seconds": profile.max_runtime_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
