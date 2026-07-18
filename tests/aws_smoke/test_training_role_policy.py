"""Adversarial model-channel IAM scope regressions."""

from __future__ import annotations

from pathlib import Path

from infra.sagemaker.role import training_role_inline_policy

ROOT = Path(__file__).resolve().parents[2]


def test_model_reads_are_separate_and_prefix_scoped() -> None:
    policy = training_role_inline_policy(
        artifact_bucket_arn="arn:aws:s3:::distillery-225989358036-us-east-1",
        model_bucket_arn="arn:aws:s3:::distillery-225989358036-us-east-1",
        ecr_repository_arn=(
            "arn:aws:ecr:us-east-1:225989358036:repository/distillery-training"
        ),
        run_artifact_prefix="artifacts/runs/run_awssmoke-oracle-sft",
        dataset_prefix="datasets/ds_awssmoke01",
        model_channel_prefix="models",
        model_prefix="models/Qwen",
        model_materialization_key="models/materialization.json",
    )
    statements = {statement["Sid"]: statement for statement in policy["Statement"]}
    assert statements["ModelBucketList"]["Condition"]["StringLike"]["s3:prefix"] == [
        "models",
        "models/",
        "models/*",
    ]
    assert statements["ModelInputRead"]["Resource"] == [
        "arn:aws:s3:::distillery-225989358036-us-east-1/models/Qwen/*",
        (
            "arn:aws:s3:::distillery-225989358036-us-east-1/"
            "models/materialization.json"
        ),
    ]
    assert statements["ManifestAndDatasetRead"]["Resource"] == [
        (
            "arn:aws:s3:::distillery-225989358036-us-east-1/"
            "artifacts/runs/run_awssmoke-oracle-sft/manifest/*"
        ),
        (
            "arn:aws:s3:::distillery-225989358036-us-east-1/"
            "datasets/ds_awssmoke01/*"
        ),
    ]
    assert statements["RunOutputWrite"]["Resource"] == [
        (
            "arn:aws:s3:::distillery-225989358036-us-east-1/"
            "artifacts/runs/run_awssmoke-oracle-sft/sagemaker-output/*"
        )
    ]


def test_role_template_requires_explicit_model_scope() -> None:
    text = (ROOT / "infra/sagemaker/training-role.yaml").read_text(encoding="utf-8")
    for marker in (
        "ModelBucketName:",
        "ModelChannelPrefix:",
        "ModelPrefix:",
        "ModelMaterializationKey:",
        "Sid: ModelBucketList",
        "Sid: ModelInputRead",
    ):
        assert marker in text
    assert (
        'Resource: !Sub "arn:aws:s3:::${ArtifactBucketName}/*"'
        not in text
    )
