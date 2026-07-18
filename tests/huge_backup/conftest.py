"""Fixtures for huge_backup pure tests (no AWS, no huge weight downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.huge_backup.memory import HugeBackupMemoryProbeEvidence  # noqa: E402
from experiments.huge_backup.pins import HugeBackupEvidence  # noqa: E402
from experiments.huge_backup.profile import HugeBackupTrainingProfile  # noqa: E402

DIGEST_HEX = "a" * 64
TOKEN_HEX = "b" * 64
LOCK_HEX = "c" * 64
STUDENT_REV = "e" * 40
TEACHER_REV = "f" * 40
SOURCE_REV = "1" * 40
ACCOUNT = "225989358036"
TEACHER_RESP_HEX = "9" * 64


@pytest.fixture
def mini_profile() -> HugeBackupTrainingProfile:
    return HugeBackupTrainingProfile(
        train_examples=16,
        max_updates=4,
        world_size=2,
        global_batch=4,
        grad_accumulation=2,
        max_length=64,
        lora_rank=16,
        lora_alpha=32,
    )


@pytest.fixture
def valid_evidence() -> HugeBackupEvidence:
    image_digest = f"sha256:{DIGEST_HEX}"
    capacity = 80 * 1024**3
    peak = 40 * 1024**3
    return HugeBackupEvidence(
        aws_account_id=ACCOUNT,
        aws_region="us-east-1",
        aws_profile="gabriel-cli",
        iam_role_arn=f"arn:aws:iam::{ACCOUNT}:role/DistillerySageMakerTrainingRole",
        artifact_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/artifacts",
        dataset_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/datasets/ds_hugebackup01/",
        models_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/models/",
        teacher_responses_s3_uri=(
            f"s3://distillery-artifacts-{ACCOUNT}/teacher_responses/hugebackup/"
        ),
        ecr_image_uri=(f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/distillery@{image_digest}"),
        image_digest=image_digest,
        student_revision=STUDENT_REV,
        student_model_config_sha256="2" * 64,
        teacher_revision=TEACHER_REV,
        teacher_model_config_sha256="3" * 64,
        teacher_responses_sha256=TEACHER_RESP_HEX,
        student_tokenizer_sha256=TOKEN_HEX,
        teacher_tokenizer_sha256=TOKEN_HEX,
        package_lock_hash=LOCK_HEX,
        source_revision=SOURCE_REV,
        license_disposition="apache-2.0_approved_for_hackathon_output_use",
        output_use_disposition="synthetic_demo_outputs_ok_no_customer_data",
        price_source="operator_attested_ml.p4de.24xlarge_us-east-1_31.5641",
        hourly_usd=31.5641,
        evidence_attested_by="gabriel",
        flash_attention_2_attested=True,
        memory_probe_evidence=HugeBackupMemoryProbeEvidence(
            passed=True,
            precision_mode="bf16_lora",
            device_type="NVIDIA A100-SXM4-80GB",
            peak_memory_bytes=peak,
            capacity_memory_bytes=capacity,
            headroom_bytes=capacity - peak,
            safe_peak_bytes=int(capacity * 0.85),
            probe_id="synthetic-huge-backup-probe",
            student_model_id="Qwen/Qwen2.5-14B-Instruct",
            student_revision=STUDENT_REV,
            teacher_model_id="Qwen/Qwen2.5-32B-Instruct",
            teacher_revision=TEACHER_REV,
            max_length=768,
            microbatch=1,
            world_size=8,
            runtime_image_digest=image_digest,
            instance_type="ml.p4de.24xlarge",
            flash_attention_2_attested=True,
        ),
    )
