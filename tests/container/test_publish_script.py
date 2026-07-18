"""Fully faked behavioral tests for ECR publish safety gates."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from conftest import (
    BUILD_MANIFEST_SCHEMA,
    MANIFEST_TOOL,
    PUBLISH_SCRIPT,
    valid_manifest,
)

ACCOUNT = "123456789012"
REGION = "us-east-1"
REPOSITORY_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/distillery-training"
CONFIG_ID = "sha256:" + ("1" * 64)
REGISTRY_DIGEST = "sha256:" + ("2" * 64)


def initialize_clean_publish_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "scripts" / "container").mkdir(parents=True)
    (repo / "containers" / "training").mkdir(parents=True)
    shutil.copy2(
        PUBLISH_SCRIPT,
        repo / "scripts" / "container" / "publish_training_image.sh",
    )
    shutil.copy2(
        MANIFEST_TOOL,
        repo / "scripts" / "container" / "manifest_tool.py",
    )
    shutil.copy2(
        BUILD_MANIFEST_SCHEMA,
        repo / "containers" / "training" / "build-manifest.schema.json",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Container Tests",
            "-c",
            "user.email=container-tests@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return repo, head


def write_fake_tools(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    state = {
        "pushed": tmp_path / "pushed",
        "docker_log": tmp_path / "docker.log",
        "aws_log": tmp_path / "aws.log",
        "password": tmp_path / "password.stdin",
    }
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_AWS_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")

if args[:2] == ["sts", "get-caller-identity"]:
    arn = os.environ.get(
        "FAKE_CALLER_ARN",
        "arn:aws:iam::123456789012:user/hackathon-builder",
    )
    print(json.dumps({"Account": "123456789012", "Arn": arn, "UserId": "TEST"}))
elif args[:2] == ["ecr", "describe-repositories"]:
    print(json.dumps({"repositories": [{
        "repositoryUri": os.environ["DISTILLERY_ECR_REPOSITORY_URI"],
        "imageTagMutability": os.environ.get("FAKE_TAG_MUTABILITY", "IMMUTABLE"),
        "imageScanningConfiguration": {"scanOnPush": True},
        "encryptionConfiguration": {"encryptionType": "AES256"},
    }]}))
elif args[:2] == ["ecr", "describe-images"]:
    pushed = Path(os.environ["FAKE_PUSHED"]).exists()
    exists = os.environ.get("FAKE_TAG_EXISTS") == "1"
    if pushed:
        print(json.dumps({"imageDetails": [{
            "imageDigest": "sha256:" + ("2" * 64),
            "imageTags": [os.environ["FAKE_TAG"]],
        }]}))
    elif exists:
        print(json.dumps({"imageDetails": [{"imageDigest": "sha256:" + ("9" * 64)}]}))
    else:
        print("ImageNotFoundException: requested image not found", file=sys.stderr)
        raise SystemExit(254)
elif args[:2] == ["ecr", "get-login-password"]:
    print("password with spaces")
elif args[:2] == ["ecr", "describe-image-scan-findings"]:
    print(json.dumps({
        "imageScanStatus": {"status": "COMPLETE"},
        "imageScanFindings": {"findingSeverityCounts": {
            "CRITICAL": int(os.environ.get("FAKE_CRITICAL", "0")),
            "HIGH": int(os.environ.get("FAKE_HIGH", "0")),
        }},
    }))
else:
    print(f"unsupported fake aws invocation: {args}", file=sys.stderr)
    raise SystemExit(97)
""",
        encoding="utf-8",
    )
    fake_aws.chmod(fake_aws.stat().st_mode | stat.S_IXUSR)

    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_DOCKER_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")

if args[:2] == ["image", "inspect"]:
    print(json.dumps([{
        "Id": "sha256:" + ("1" * 64),
        "Config": {"Labels": {
            "distillery.source.sha": os.environ["FAKE_COMMIT_SHA"],
            "distillery.source.tree.sha256": "b" * 64,
            "distillery.package.lock.sha256": "c" * 64,
        }},
    }]))
elif args and args[0] == "login":
    Path(os.environ["FAKE_PASSWORD"]).write_text(sys.stdin.read(), encoding="utf-8")
elif args and args[0] == "tag":
    pass
elif args and args[0] == "push":
    Path(os.environ["FAKE_PUSHED"]).write_text("pushed\\n", encoding="utf-8")
else:
    print(f"unsupported fake docker invocation: {args}", file=sys.stderr)
    raise SystemExit(98)
""",
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    return fake_bin, state


def publish_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str], dict[str, Path]]:
    repo, head = initialize_clean_publish_repo(tmp_path)
    manifest = tmp_path / "manifest with spaces.json"
    payload = valid_manifest()
    payload["source"]["commit_sha"] = head
    payload["source"]["reviewed_commit_sha"] = head
    payload["tag"] = f"pinned-training-{head[:12]}-cccccccccccc"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fake_bin, state = write_fake_tools(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "AWS_PROFILE": "hackathon-builder",
            "AWS_REGION": REGION,
            "DISTILLERY_ECR_REPOSITORY_URI": REPOSITORY_URI,
            "DISTILLERY_BUILD_MANIFEST": str(manifest),
            "DISTILLERY_PACKAGING_PYTHON": sys.executable,
            "DISTILLERY_SCAN_POLL_SECONDS": "0",
            "FAKE_PUSHED": str(state["pushed"]),
            "FAKE_DOCKER_LOG": str(state["docker_log"]),
            "FAKE_AWS_LOG": str(state["aws_log"]),
            "FAKE_PASSWORD": str(state["password"]),
            "FAKE_TAG": payload["tag"],
            "FAKE_COMMIT_SHA": head,
        }
    )
    return repo, manifest, env, state


def run_publish(
    repo: Path,
    env: dict[str, str],
    mode: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "container" / "publish_training_image.sh"), mode],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_publish_dry_run_is_read_only_and_checks_tag_availability(
    tmp_path: Path,
) -> None:
    repo, manifest, env, state = publish_fixture(tmp_path)
    before = manifest.read_bytes()
    result = run_publish(repo, env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "no login, tag, push, or manifest mutation" in result.stdout
    assert manifest.read_bytes() == before
    assert not state["pushed"].exists()
    assert not state["password"].exists()
    aws_calls = state["aws_log"].read_text(encoding="utf-8")
    assert "get-login-password" not in aws_calls
    assert "describe-images" in aws_calls


def test_publish_requires_exact_dynamic_confirmation(tmp_path: Path) -> None:
    repo, manifest, env, state = publish_fixture(tmp_path)
    tag = json.loads(manifest.read_text(encoding="utf-8"))["tag"]
    env["DISTILLERY_PUBLISH_CONFIRM"] = "PUSH"
    result = run_publish(repo, env, "--publish")
    assert result.returncode != 0
    assert f"PUSH {tag} TO {ACCOUNT}/{REGION}" in result.stderr
    assert not state["pushed"].exists()


def test_publish_rejects_root_wrong_uri_and_existing_tag(tmp_path: Path) -> None:
    repo, _manifest, env, state = publish_fixture(tmp_path)
    env["FAKE_CALLER_ARN"] = f"arn:aws:iam::{ACCOUNT}:root"
    root = run_publish(repo, env, "--dry-run")
    assert root.returncode != 0
    assert "account root identity" in root.stderr

    env["FAKE_CALLER_ARN"] = f"arn:aws:iam::{ACCOUNT}:user/hackathon-builder"
    env["DISTILLERY_ECR_REPOSITORY_URI"] = "evil.example/distillery-training"
    wrong_uri = run_publish(repo, env, "--dry-run")
    assert wrong_uri.returncode != 0
    assert "must exactly match" in wrong_uri.stderr

    env["DISTILLERY_ECR_REPOSITORY_URI"] = REPOSITORY_URI
    env["FAKE_TAG_MUTABILITY"] = "MUTABLE"
    mutable = run_publish(repo, env, "--dry-run")
    assert mutable.returncode != 0
    assert "immutable tags" in mutable.stderr

    env["FAKE_TAG_MUTABILITY"] = "IMMUTABLE"
    env["FAKE_TAG_EXISTS"] = "1"
    existing = run_publish(repo, env, "--dry-run")
    assert existing.returncode != 0
    assert "immutable ECR tag already exists" in existing.stderr
    assert not state["pushed"].exists()


def test_publish_rejects_local_image_provenance_mismatch(tmp_path: Path) -> None:
    repo, _manifest, env, state = publish_fixture(tmp_path)
    env["FAKE_COMMIT_SHA"] = "0" * 40
    result = run_publish(repo, env, "--dry-run")
    assert result.returncode != 0
    assert "distillery.source.sha" in result.stderr
    assert not state["aws_log"].exists()
    assert not state["pushed"].exists()


def test_publish_uses_stdin_and_records_verified_ecr_digest(tmp_path: Path) -> None:
    repo, manifest, env, state = publish_fixture(tmp_path)
    tag = json.loads(manifest.read_text(encoding="utf-8"))["tag"]
    env["DISTILLERY_PUBLISH_CONFIRM"] = f"PUSH {tag} TO {ACCOUNT}/{REGION}"
    result = run_publish(repo, env, "--publish")
    assert result.returncode == 0, result.stderr
    assert "password with spaces" not in result.stdout
    assert "password with spaces" not in result.stderr
    assert state["password"].read_text(encoding="utf-8") == "password with spaces\n"
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["local"]["config_id"] == CONFIG_ID
    assert updated["registry"]["image_digest"] == REGISTRY_DIGEST
    assert updated["registry"]["digest_uri"] == (f"{REPOSITORY_URI}@{REGISTRY_DIGEST}")
    assert updated["registry"]["verified"] is True
    docker_calls = [
        json.loads(line) for line in state["docker_log"].read_text(encoding="utf-8").splitlines()
    ]
    assert any(call[0] == "login" and "--password-stdin" in call for call in docker_calls)
    assert not any("latest" in component.lower() for call in docker_calls for component in call)


def test_scan_policy_blocks_manifest_verification(tmp_path: Path) -> None:
    repo, manifest, env, state = publish_fixture(tmp_path)
    before = manifest.read_bytes()
    tag = json.loads(manifest.read_text(encoding="utf-8"))["tag"]
    env["FAKE_CRITICAL"] = "1"
    env["DISTILLERY_PUBLISH_CONFIRM"] = f"PUSH {tag} TO {ACCOUNT}/{REGION}"
    result = run_publish(repo, env, "--publish")
    assert result.returncode != 0
    assert "critical image findings" in result.stderr
    assert state["pushed"].exists()
    assert manifest.read_bytes() == before


def test_malformed_manifest_rejected_before_aws(tmp_path: Path) -> None:
    repo, manifest, env, state = publish_fixture(tmp_path)
    manifest.write_text("{malformed\n", encoding="utf-8")
    result = run_publish(repo, env, "--dry-run")
    assert result.returncode != 0
    assert "malformed JSON" in result.stderr
    assert not state["aws_log"].exists()


def test_script_is_executable() -> None:
    assert PUBLISH_SCRIPT.stat().st_mode & stat.S_IXUSR
