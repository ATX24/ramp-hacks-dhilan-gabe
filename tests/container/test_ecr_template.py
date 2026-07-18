"""Parsed-template and fully faked ECR deployment regressions."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from conftest import ECR_DEPLOY, ECR_TEMPLATE


def parsed_template() -> dict[str, Any]:
    value = json.loads(ECR_TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_repository_is_retained_encrypted_scanned_and_immutable() -> None:
    template = parsed_template()
    repository = template["Resources"]["TrainingRepository"]
    assert repository["Type"] == "AWS::ECR::Repository"
    assert repository["DeletionPolicy"] == "Retain"
    assert repository["UpdateReplacePolicy"] == "Retain"
    properties = repository["Properties"]
    assert properties["ImageTagMutability"] == "IMMUTABLE"
    assert properties["ImageScanningConfiguration"] == {"ScanOnPush": True}
    assert properties["EncryptionConfiguration"] == {"EncryptionType": "AES256"}
    assert template["Parameters"]["RepositoryName"]["AllowedValues"] == ["distillery-training"]


def test_lifecycle_never_expires_pinned_manifest_images() -> None:
    template = parsed_template()
    lifecycle_sub = template["Resources"]["TrainingRepository"]["Properties"]["LifecyclePolicy"][
        "LifecyclePolicyText"
    ]["Fn::Sub"]
    policy = json.loads(lifecycle_sub.replace("${CandidateRetentionDays}", "30"))
    rules = policy["rules"]
    assert len(rules) == 2
    assert rules[0]["selection"]["tagStatus"] == "untagged"
    assert rules[0]["selection"]["countNumber"] >= 14
    assert rules[1]["selection"]["tagPrefixList"] == ["candidate-"]
    assert rules[1]["selection"]["countNumber"] >= 14
    assert all(
        "pinned-training-" not in prefix
        for rule in rules
        for prefix in rule["selection"].get("tagPrefixList", [])
    )
    assert all(rule["selection"]["countType"] != "imageCountMoreThan" for rule in rules)


def test_repository_policy_scopes_sagemaker_source_account_and_arn() -> None:
    template = parsed_template()
    policy = template["Resources"]["TrainingRepository"]["Properties"]["RepositoryPolicyText"]
    statement = policy["Statement"][0]
    assert statement["Principal"] == {"Service": "sagemaker.amazonaws.com"}
    assert statement["Condition"]["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}
    assert statement["Condition"]["ArnLike"]["aws:SourceArn"]["Fn::Sub"] == (
        "arn:${AWS::Partition}:sagemaker:${AWS::Region}:${AWS::AccountId}:training-job/distillery-*"
    )
    assert set(statement["Action"]) == {
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
    }


def test_push_and_pull_policies_are_repository_scoped() -> None:
    template = parsed_template()
    resources = template["Resources"]
    for name in ("TrainingImagePushPolicy", "TrainingImagePullPolicy"):
        statements = resources[name]["Properties"]["PolicyDocument"]["Statement"]
        auth, repository = statements
        assert auth["Action"] == ["ecr:GetAuthorizationToken"]
        assert auth["Resource"] == "*"
        assert repository["Resource"] == {"Fn::GetAtt": ["TrainingRepository", "Arn"]}
        assert "ecr:DeleteRepository" not in repository["Action"]
        assert "ecr:BatchDeleteImage" not in repository["Action"]


def write_fake_aws(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "aws.log"
    fake = fake_bin / "aws"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_AWS_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[:2] == ["sts", "get-caller-identity"]:
    print(json.dumps({
        "Account": "123456789012",
        "Arn": os.environ.get(
            "FAKE_CALLER_ARN",
            "arn:aws:iam::123456789012:user/hackathon-builder",
        ),
    }))
elif args[:2] in (
    ["cloudformation", "validate-template"],
    ["cloudformation", "deploy"],
):
    print("{}")
else:
    raise SystemExit(90)
""",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake_bin, log


def test_deploy_preflight_refuses_default_or_root_identity(tmp_path: Path) -> None:
    fake_bin, log = write_fake_aws(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_AWS_LOG"] = str(log)
    env.pop("AWS_PROFILE", None)
    no_profile = subprocess.run(
        [str(ECR_DEPLOY), "--preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert no_profile.returncode != 0
    assert "AWS_PROFILE" in no_profile.stderr

    env["AWS_PROFILE"] = "unsafe"
    env["FAKE_CALLER_ARN"] = "arn:aws:iam::123456789012:root"
    root = subprocess.run(
        [str(ECR_DEPLOY), "--preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert root.returncode != 0
    assert "account root identity" in root.stderr


def test_deploy_preflight_is_read_only_and_apply_is_typed(tmp_path: Path) -> None:
    fake_bin, log = write_fake_aws(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_AWS_LOG": str(log),
            "AWS_PROFILE": "hackathon-builder",
            "AWS_REGION": "us-east-1",
        }
    )
    preflight = subprocess.run(
        [str(ECR_DEPLOY), "--preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    calls = log.read_text(encoding="utf-8")
    assert "validate-template" in calls
    assert '"deploy"' not in calls

    log.write_text("", encoding="utf-8")
    env["DISTILLERY_ECR_DEPLOY_CONFIRM"] = "YES"
    refused = subprocess.run(
        [str(ECR_DEPLOY), "--apply"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "CREATE distillery-training IN 123456789012/us-east-1" in refused.stderr
    assert '"deploy"' not in log.read_text(encoding="utf-8")

    log.write_text("", encoding="utf-8")
    env["DISTILLERY_ECR_DEPLOY_CONFIRM"] = "CREATE distillery-training IN 123456789012/us-east-1"
    applied = subprocess.run(
        [str(ECR_DEPLOY), "--apply"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    deploy_call = next(call for call in calls if call[:2] == ["cloudformation", "deploy"])
    assert "CAPABILITY_IAM" in deploy_call
    assert "RepositoryName=distillery-training" in deploy_call
    assert "CandidateRetentionDays=30" in deploy_call
