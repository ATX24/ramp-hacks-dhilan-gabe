"""Behavioral tests for manifest schema, atomic writes, and ECR binding."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from conftest import (
    BUILD_MANIFEST_SCHEMA,
    MANIFEST_TOOL,
    load_module,
    valid_manifest,
)


@pytest.fixture(scope="module")
def manifest_module():
    return load_module(MANIFEST_TOOL, "distillery_container_manifest_tool")


def test_schema_accepts_unverified_and_verified_lifecycles() -> None:
    schema = json.loads(BUILD_MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    validator.validate(valid_manifest(config_id=None))
    validator.validate(valid_manifest(config_id="sha256:" + ("1" * 64)))
    validator.validate(valid_manifest(verified=True))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(
                tag="latest",
            ),
            "latest",
        ),
        (
            lambda payload: payload["registry"].update(
                verified=True,
                repository_uri=("123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-training"),
                image_digest="sha256:" + ("2" * 64),
                digest_uri="evil.example/image@sha256:" + ("2" * 64),
                verified_at="2026-07-18T12:00:00Z",
                scan_status="COMPLETE",
                critical_findings=0,
                high_findings=0,
            ),
            "digest_uri",
        ),
        (
            lambda payload: payload["source"].update(
                clean=False,
                commit_bound=True,
            ),
            "commit_bound",
        ),
    ],
)
def test_semantic_validation_rejects_unsafe_values(
    manifest_module,
    mutation,
    message: str,
) -> None:
    payload = valid_manifest()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        manifest_module.validate_manifest(payload, BUILD_MANIFEST_SCHEMA)


def test_atomic_write_validates_before_replace(
    tmp_path: Path,
    manifest_module,
) -> None:
    destination = tmp_path / "manifest.json"
    original = valid_manifest()
    manifest_module.atomic_write_manifest(
        destination,
        original,
        schema_path=BUILD_MANIFEST_SCHEMA,
    )
    original_bytes = destination.read_bytes()

    invalid = valid_manifest()
    invalid["unexpected"] = "must fail"
    with pytest.raises(ValueError, match="schema validation failed"):
        manifest_module.atomic_write_manifest(
            destination,
            invalid,
            schema_path=BUILD_MANIFEST_SCHEMA,
        )

    assert destination.read_bytes() == original_bytes
    assert list(tmp_path.glob(".manifest.json.*")) == []


def test_manifest_get_preserves_spaces_and_rejects_malformed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest with spaces.json"
    manifest.write_text(
        json.dumps(valid_manifest(compatibility="blocked"), indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(MANIFEST_TOOL),
        "get",
        "--schema",
        str(BUILD_MANIFEST_SCHEMA),
        "--manifest",
        str(manifest),
        "--path",
        "ml_compatibility.reasons",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert result.stdout == '["blocked by lock"]\n'

    manifest.write_text("{not-json\n", encoding="utf-8")
    invalid = subprocess.run(command, check=False, capture_output=True, text=True)
    assert invalid.returncode == 2
    assert "malformed JSON" in invalid.stderr


@pytest.mark.parametrize(
    ("uri", "account", "region", "repository"),
    [
        (
            "999999999999.dkr.ecr.us-east-1.amazonaws.com/distillery-training",
            "123456789012",
            "us-east-1",
            "distillery-training",
        ),
        (
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/distillery-training",
            "123456789012",
            "us-east-1",
            "distillery-training",
        ),
        (
            "https://123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-training",
            "123456789012",
            "us-east-1",
            "distillery-training",
        ),
        (
            "evil.example/distillery-training",
            "123456789012",
            "us-east-1",
            "distillery-training",
        ),
        (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/other",
            "123456789012",
            "us-east-1",
            "other",
        ),
    ],
)
def test_ecr_uri_rejects_wrong_identity_region_host_or_repository(
    manifest_module,
    uri: str,
    account: str,
    region: str,
    repository: str,
) -> None:
    with pytest.raises(ValueError):
        manifest_module.validate_ecr_repository_uri(
            uri,
            account=account,
            region=region,
            repository=repository,
        )


def test_ecr_uri_accepts_exact_sts_region_and_allowlist(manifest_module) -> None:
    uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-training"
    assert (
        manifest_module.validate_ecr_repository_uri(
            uri,
            account="123456789012",
            region="us-east-1",
            repository="distillery-training",
        )
        == uri
    )


def test_local_config_id_is_not_a_registry_digest(
    tmp_path: Path,
    manifest_module,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = valid_manifest(config_id=None)
    manifest_module.atomic_write_manifest(
        manifest_path,
        payload,
        schema_path=BUILD_MANIFEST_SCHEMA,
    )

    args = manifest_module.build_parser().parse_args(
        [
            "set-local",
            "--schema",
            str(BUILD_MANIFEST_SCHEMA),
            "--manifest",
            str(manifest_path),
            "--config-id",
            "sha256:" + ("1" * 64),
        ]
    )
    args.handler(args)
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["local"]["config_id"] == "sha256:" + ("1" * 64)
    assert updated["registry"]["image_digest"] is None
    assert updated["registry"]["digest_uri"] is None
    assert updated["registry"]["verified"] is False


def test_registry_binding_is_exact_and_scan_gated(
    tmp_path: Path,
    manifest_module,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_module.atomic_write_manifest(
        manifest_path,
        valid_manifest(),
        schema_path=BUILD_MANIFEST_SCHEMA,
    )
    repository = "123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-training"
    digest = "sha256:" + ("2" * 64)
    base_args = [
        "set-registry",
        "--schema",
        str(BUILD_MANIFEST_SCHEMA),
        "--manifest",
        str(manifest_path),
        "--repository-uri",
        repository,
        "--account",
        "123456789012",
        "--region",
        "us-east-1",
        "--image-digest",
        digest,
        "--scan-status",
        "COMPLETE",
        "--critical-findings",
        "0",
        "--high-findings",
        "0",
    ]
    args = manifest_module.build_parser().parse_args(base_args)
    args.handler(args)
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["registry"]["digest_uri"] == f"{repository}@{digest}"
    assert updated["registry"]["verified"] is True

    unsafe_path = tmp_path / "unsafe.json"
    manifest_module.atomic_write_manifest(
        unsafe_path,
        valid_manifest(),
        schema_path=BUILD_MANIFEST_SCHEMA,
    )
    unsafe_args = base_args.copy()
    unsafe_args[unsafe_args.index(str(manifest_path))] = str(unsafe_path)
    unsafe_args[unsafe_args.index("--critical-findings") + 1] = "1"
    parsed = manifest_module.build_parser().parse_args(unsafe_args)
    with pytest.raises(ValueError, match="critical image findings"):
        parsed.handler(parsed)
    unchanged = json.loads(unsafe_path.read_text(encoding="utf-8"))
    assert unchanged["registry"]["verified"] is False
