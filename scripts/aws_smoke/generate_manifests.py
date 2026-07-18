#!/usr/bin/env python3
"""Generate the fixed subset and canonical per-arm manifest channels."""

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
    REQUIRED_ARMS,
    EmergencyTrainingProfile,
)
from experiments.aws_smoke.tokenization import (  # noqa: E402
    load_tokenization_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="generate_manifests")
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--tokenization-evidence", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-evidence-template",
        type=Path,
        default=None,
        help="Write an intentionally invalid UNSET evidence template and exit",
    )
    parser.add_argument("--dataset-id", type=str, default="ds_awssmoke01")
    parser.add_argument(
        "--precision-mode",
        choices=["qlora_nf4", "bf16_lora"],
        default="qlora_nf4",
    )
    parser.add_argument(
        "--prepare-subset-only",
        action="store_true",
        help="Materialize the deterministic subset before offline tokenizer evidence",
    )
    args = parser.parse_args()

    if args.write_evidence_template is not None:
        args.write_evidence_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_evidence_template.write_text(
            json.dumps(evidence_schema_template(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "template": str(args.write_evidence_template)}))
        return 0
    if args.evidence is None or args.output_dir is None:
        raise SystemExit("--evidence and --output-dir are required")

    evidence = load_evidence(args.evidence)
    profile = EmergencyTrainingProfile(
        precision_mode=args.precision_mode,
        memory_probe_evidence=evidence.memory_probe_evidence,
    )
    subset_dir = args.output_dir / "subset"
    subset = materialize_emergency_subset(subset_dir, profile=profile)
    if (
        evidence.data_content_sha256 is not None
        and evidence.data_content_sha256 != subset.content_sha256
    ):
        raise SystemExit(
            "evidence.data_content_sha256 does not match generated subset content hash"
        )
    if args.prepare_subset_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "prepare_subset_only",
                    "subset_dir": str(subset_dir),
                    "subset_content_sha256": subset.content_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.tokenization_evidence is None:
        raise SystemExit(
            "--tokenization-evidence is required unless --prepare-subset-only is used"
        )
    tokenization = load_tokenization_evidence(args.tokenization_evidence)
    train_rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    arms = REQUIRED_ARMS
    if "sequence_kd" in tokenization.arms:
        arms = REQUIRED_ARMS + ("sequence_kd",)
    paths = write_arm_manifests(
        output_dir=args.output_dir / "manifests",
        evidence=evidence,
        dataset_id=args.dataset_id,
        dataset_uri=evidence.dataset_s3_uri,
        dataset_sha256=subset.content_sha256,
        split_sha256=subset.split_sha256,
        example_ids=[str(row["example_id"]) for row in train_rows],
        tasks=[str(row["task"]) for row in train_rows],
        difficulties=[str(row["difficulty"]) for row in train_rows],
        tokenization_evidence=tokenization,
        arms=arms,
        profile=profile,
    )
    distinct_signals = 3 if "sequence_kd" in paths else 2
    print(
        json.dumps(
            {
                "ok": True,
                "subset_content_sha256": subset.content_sha256,
                "manifests": {arm: str(path) for arm, path in paths.items()},
                "distinct_signal_count": distinct_signals,
                "default_launch_ready": distinct_signals >= 3,
                "control_disclosure": (
                    "ce_ablation is oracle_sft-equivalent when both use oracle targets"
                ),
                "max_run_usd": profile.max_run_usd,
                "max_runtime_seconds": profile.max_runtime_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
