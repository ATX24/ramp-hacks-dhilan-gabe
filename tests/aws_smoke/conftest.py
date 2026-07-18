"""Fixtures for aws_smoke pure tests (mock AWS only; no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.aws_smoke.memory import Bf16MemoryEvidence  # noqa: E402
from experiments.aws_smoke.pins import EmergencyEvidence  # noqa: E402

DIGEST_HEX = "a" * 64
TOKEN_HEX = "b" * 64
LOCK_HEX = "c" * 64
PROOF_HEX = "d" * 64
STUDENT_REV = "e" * 40
TEACHER_REV = "f" * 40
SOURCE_REV = "1" * 40
ACCOUNT = "225989358036"


@pytest.fixture
def valid_evidence() -> EmergencyEvidence:
    image_digest = f"sha256:{DIGEST_HEX}"
    capacity = 24 * 1024**3
    peak = 8 * 1024**3
    return EmergencyEvidence(
        aws_account_id=ACCOUNT,
        aws_region="us-east-1",
        aws_profile="gabriel-cli",
        iam_role_arn=f"arn:aws:iam::{ACCOUNT}:role/DistillerySageMakerTrainingRole",
        artifact_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/artifacts",
        dataset_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/datasets/ds_awssmoke01/",
        models_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/models/",
        model_materialization_uri=(
            f"s3://distillery-artifacts-{ACCOUNT}/models/materialization.json"
        ),
        model_materialization_sha256="4" * 64,
        ecr_image_uri=(
            f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/distillery@{image_digest}"
        ),
        image_digest=image_digest,
        student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=STUDENT_REV,
        student_model_config_sha256="2" * 64,
        teacher_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        teacher_revision=TEACHER_REV,
        teacher_model_config_sha256="3" * 64,
        student_tokenizer_sha256=TOKEN_HEX,
        teacher_tokenizer_sha256=TOKEN_HEX,
        student_chat_template_sha256=TOKEN_HEX,
        teacher_chat_template_sha256=TOKEN_HEX,
        student_special_token_map={"eos_token": 151645, "pad_token": 151643},
        teacher_special_token_map={"eos_token": 151645, "pad_token": 151643},
        package_lock_hash=LOCK_HEX,
        source_revision=SOURCE_REV,
        proof_protocol_id="finance-proof.v1",
        proof_protocol_sha256=PROOF_HEX,
        license_disposition="apache-2.0_approved_for_hackathon_output_use",
        output_use_disposition="synthetic_demo_outputs_ok_no_customer_data",
        data_content_sha256="a8888d489ac5ade418110196c244ff3418620701e5c9ca80dd6bba03288d0775",
        price_source="operator_attested_ml.g5.xlarge_us-east-1_1.408",
        hourly_usd=1.408,
        evidence_attested_by="gabriel",
        evidence_notes="test fixture only",
        memory_probe_evidence=Bf16MemoryEvidence(
            passed=True,
            precision_mode="qlora_nf4",
            device_type="NVIDIA A10G",
            peak_memory_bytes=peak,
            capacity_memory_bytes=capacity,
            headroom_bytes=capacity - peak,
            probe_id="synthetic-test-probe",
            student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            student_revision=STUDENT_REV,
            teacher_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            teacher_revision=TEACHER_REV,
            max_length=512,
            max_completion=128,
            vocab_chunk_size=4096,
            microbatch=1,
            grad_accumulation=1,
            runtime_image_digest=image_digest,
            instance_type="ml.g5.xlarge",
        ),
    )
