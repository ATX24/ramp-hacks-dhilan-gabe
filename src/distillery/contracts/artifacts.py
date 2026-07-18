"""Immutable ModelArtifact resource contract."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    StrictStr,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    GitCommitSha,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.ids import ArtifactId, RunId


class ArtifactChecksums(FrozenModel):
    """Checksums corresponding one-for-one with artifact URI fields."""

    adapter_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    chat_template_sha256: Sha256Hex
    merged_sha256: Sha256Hex | None = None


class ModelArtifact(FrozenModel):
    """Immutable portable student artifact metadata."""

    schema_version: Literal["distillery.model_artifact.v1"] = (
        "distillery.model_artifact.v1"
    )
    artifact_id: ArtifactId
    run_id: RunId
    student_base_id: StrictStr = Field(min_length=1)
    student_revision: GitCommitSha
    adapter_uri: StrictStr = Field(min_length=1)
    merged_uri: StrictStr | None = Field(default=None, min_length=1)
    tokenizer_uri: StrictStr = Field(min_length=1)
    chat_template_uri: StrictStr = Field(min_length=1)
    license_record: FrozenJsonObject
    checksums: ArtifactChecksums
    load_instructions: StrictStr = Field(min_length=1)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _merged_uri_checksum_pair(self) -> ModelArtifact:
        if (self.merged_uri is None) != (self.checksums.merged_sha256 is None):
            raise ValueError(
                "merged_uri and checksums.merged_sha256 must be supplied together"
            )
        return self

    def resource_hash(self) -> str:
        payload = self.model_dump(mode="python", exclude={"created_at"})
        return content_sha256(payload)
