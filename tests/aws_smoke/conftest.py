"""Fixtures for aws_smoke pure tests (mock AWS only; no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    return EmergencyEvidence(
        aws_account_id=ACCOUNT,
        aws_region="us-east-1",
        aws_profile="gabriel-cli",
        iam_role_arn=f"arn:aws:iam::{ACCOUNT}:role/DistillerySageMakerTrainingRole",
        artifact_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/artifacts",
        dataset_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/datasets/ds_awssmoke01/",
        ecr_image_uri=(
            f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/distillery@{image_digest}"
        ),
        image_digest=image_digest,
        student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=STUDENT_REV,
        teacher_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        teacher_revision=TEACHER_REV,
        student_tokenizer_sha256=TOKEN_HEX,
        teacher_tokenizer_sha256=TOKEN_HEX,
        student_chat_template_sha256=TOKEN_HEX,
        teacher_chat_template_sha256=TOKEN_HEX,
        package_lock_hash=LOCK_HEX,
        source_revision=SOURCE_REV,
        proof_protocol_id="finance-proof.v1",
        proof_protocol_sha256=PROOF_HEX,
        license_disposition="apache-2.0_approved_for_hackathon_output_use",
        output_use_disposition="synthetic_demo_outputs_ok_no_customer_data",
        data_content_sha256=None,
        price_source="operator_attested_ml.g5.xlarge_us-east-1_1.408",
        hourly_usd=1.408,
        evidence_attested_by="gabriel",
        evidence_notes="test fixture only",
    )
