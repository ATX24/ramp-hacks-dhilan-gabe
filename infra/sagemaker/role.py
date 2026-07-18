"""Least-privilege policy builder for one sealed SageMaker training run."""

from __future__ import annotations

import re
from typing import Any

_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*[A-Za-z0-9]$")


def _validate_prefix(prefix: str, *, name: str) -> None:
    segments = prefix.split("/")
    if (
        not prefix
        or prefix.startswith("/")
        or prefix.endswith("/")
        or "://" in prefix
        or "\\" in prefix
        or _PREFIX_RE.fullmatch(prefix) is None
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise ValueError(f"{name} must be an unambiguous S3 key prefix")


def training_role_inline_policy(
    *,
    artifact_bucket_arn: str,
    model_bucket_arn: str,
    ecr_repository_arn: str,
    run_artifact_prefix: str,
    dataset_prefix: str,
    model_channel_prefix: str,
    model_prefix: str,
    model_materialization_key: str,
) -> dict[str, Any]:
    """Build separate read/write scopes for one run, dataset, and model set."""
    for value, name in (
        (artifact_bucket_arn, "artifact_bucket_arn"),
        (model_bucket_arn, "model_bucket_arn"),
    ):
        if not value.startswith("arn:aws:s3:::"):
            raise ValueError(f"{name} must be an S3 bucket ARN")
    if ":ecr:" not in ecr_repository_arn or ":repository/" not in ecr_repository_arn:
        raise ValueError("ecr_repository_arn must be an ECR repository ARN")
    for value, name in (
        (run_artifact_prefix, "run_artifact_prefix"),
        (dataset_prefix, "dataset_prefix"),
        (model_channel_prefix, "model_channel_prefix"),
        (model_prefix, "model_prefix"),
        (model_materialization_key, "model_materialization_key"),
    ):
        _validate_prefix(value, name=name)

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "RunAndDatasetList",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [artifact_bucket_arn],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            f"{run_artifact_prefix}/manifest/*",
                            f"{run_artifact_prefix}/sagemaker-output/*",
                            f"{dataset_prefix}/*",
                        ]
                    }
                },
            },
            {
                "Sid": "ManifestAndDatasetRead",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    f"{artifact_bucket_arn}/{run_artifact_prefix}/manifest/*",
                    f"{artifact_bucket_arn}/{dataset_prefix}/*",
                ],
            },
            {
                "Sid": "ModelBucketList",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [model_bucket_arn],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            model_channel_prefix,
                            f"{model_channel_prefix}/",
                            f"{model_channel_prefix}/*",
                        ]
                    }
                },
            },
            {
                "Sid": "ModelInputRead",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    f"{model_bucket_arn}/{model_prefix}/*",
                    f"{model_bucket_arn}/{model_materialization_key}",
                ],
            },
            {
                "Sid": "RunOutputWrite",
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                "Resource": [
                    f"{artifact_bucket_arn}/{run_artifact_prefix}/sagemaker-output/*"
                ],
            },
            {
                "Sid": "DenyManifestMutation",
                "Effect": "Deny",
                "Action": ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"],
                "Resource": [f"{artifact_bucket_arn}/*/manifest/*"],
            },
            {
                "Sid": "ECRAuthorization",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": ["*"],
            },
            {
                "Sid": "ECRPullRepository",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": [ecr_repository_arn],
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                "Resource": ["arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs*"],
            },
            {
                "Sid": "DenyUnrelatedSageMakerMutations",
                "Effect": "Deny",
                "Action": [
                    "sagemaker:CreateEndpoint",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateNotebookInstance",
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:CreateTransformJob",
                ],
                "Resource": ["*"],
            },
        ],
    }
