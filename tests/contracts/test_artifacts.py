"""Model artifact checksum-to-URI integrity invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.artifacts import ModelArtifact

HEX64 = "a" * 64


def _artifact(**updates: object) -> ModelArtifact:
    payload: dict[str, object] = {
        "artifact_id": "art_checksums_001",
        "run_id": "run_checksums_001",
        "student_base_id": "student/model",
        "student_revision": "a" * 40,
        "adapter_uri": "s3://bucket/adapter.tar.gz",
        "tokenizer_uri": "s3://bucket/tokenizer.json",
        "chat_template_uri": "s3://bucket/chat-template.txt",
        "license_record": {"status": "approved", "sources": ["model-card"]},
        "checksums": {
            "adapter_sha256": HEX64,
            "tokenizer_sha256": HEX64,
            "chat_template_sha256": HEX64,
        },
        "load_instructions": "Load the pinned base and verified adapter.",
        "created_at": datetime(2026, 7, 18, tzinfo=UTC),
    }
    payload.update(updates)
    return ModelArtifact.model_validate(payload)


@pytest.mark.parametrize(
    "missing",
    ["adapter_sha256", "tokenizer_sha256", "chat_template_sha256"],
)
def test_every_required_uri_has_a_named_checksum(missing: str) -> None:
    checksums = {
        "adapter_sha256": HEX64,
        "tokenizer_sha256": HEX64,
        "chat_template_sha256": HEX64,
    }
    checksums.pop(missing)
    with pytest.raises(ValidationError):
        _artifact(checksums=checksums)


def test_unrelated_only_checksum_map_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _artifact(checksums={"readme_sha256": HEX64})


def test_merged_uri_and_checksum_are_an_optional_pair() -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        _artifact(merged_uri="s3://bucket/merged/")
    with pytest.raises(ValidationError, match="must be supplied together"):
        _artifact(
            checksums={
                "adapter_sha256": HEX64,
                "tokenizer_sha256": HEX64,
                "chat_template_sha256": HEX64,
                "merged_sha256": HEX64,
            }
        )


def test_artifact_nested_metadata_is_immutable() -> None:
    artifact = _artifact()
    before = artifact.resource_hash()
    with pytest.raises(TypeError):
        artifact.license_record["sources"][0] = "changed"  # type: ignore[index]
    assert artifact.resource_hash() == before
