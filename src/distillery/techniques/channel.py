"""Content-addressed external plan channel for future backend integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, StrictStr, ValidationInfo, model_validator

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import PrefixedSha256, content_sha256
from distillery.techniques.errors import TechniqueError, TechniqueErrorCode, raise_technique_error
from distillery.techniques.protocol import verify_protocol_hash
from distillery.techniques.runtime import TechniquePlan

TECHNIQUE_PLAN_FILENAME = "technique_plan.json"
CHANNEL_SCHEMA_VERSION: Literal["distillery.technique.channel.v2"] = (
    "distillery.technique.channel.v2"
)


class TechniqueChannelContract(FrozenModel):
    """Exact identity a future backend must enforce before execution."""

    schema_version: Literal["distillery.technique.channel.v2"] = CHANNEL_SCHEMA_VERSION
    technique_id: StrictStr
    version: StrictStr
    plan_filename: Literal["technique_plan.json"] = TECHNIQUE_PLAN_FILENAME
    image_uri: StrictStr = Field(min_length=1)
    image_digest: PrefixedSha256
    network_isolation_required: Literal[True] = True
    reviewed_source_repository: StrictStr
    reviewed_source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    reviewed_source_tree_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_source_review_record_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_contract_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class TechniqueChannelEnvelope(FrozenModel):
    """One strict content-addressed channel payload."""

    schema_version: Literal["distillery.technique.channel_envelope.v2"] = (
        "distillery.technique.channel_envelope.v2"
    )
    contract: TechniqueChannelContract
    plan: TechniquePlan
    config: FrozenJsonObject
    envelope_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _exact_identity(self, info: ValidationInfo) -> Self:
        plan = self.plan
        external = plan.external_execution
        if external is None or plan.training_load_plan is not None:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
                "channel plan must contain exactly one external execution plan",
            )
        verify_protocol_hash(plan)
        expected = {
            "technique_id": plan.technique_id,
            "version": plan.version,
            "image_uri": external.image_uri,
            "image_digest": external.image_digest,
            "reviewed_source_repository": external.reviewed_source_repository,
            "reviewed_source_commit": external.reviewed_source_commit,
            "reviewed_source_tree_sha256": external.reviewed_source_tree_sha256,
            "reviewed_source_review_record_sha256": (external.reviewed_source_review_record_sha256),
            "descriptor_sha256": plan.descriptor_sha256,
            "config_sha256": plan.config_sha256,
            "protocol_sha256": plan.protocol_sha256,
            "environment_sha256": plan.compatibility.environment_sha256,
            "artifact_contract_sha256": content_sha256(
                plan.artifact_contract.model_dump(mode="json")
            ),
        }
        actual = self.contract.model_dump(
            mode="json",
            exclude={
                "schema_version",
                "plan_filename",
                "network_isolation_required",
            },
        )
        if actual != expected:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
                "channel contract identity does not exactly match sealed plan",
                details={
                    "expected_contract_identity": expected,
                    "actual_contract_identity": actual,
                },
            )
        config = dict(self.config)
        if config != dict(plan.resolved_config) or content_sha256(config) != plan.config_sha256:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
                "channel config does not match sealed plan/config_sha256",
            )
        if not info.context or not info.context.get("skip_envelope_hash_validation", False):
            self.assert_integrity()
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"envelope_sha256"})

    def assert_integrity(self) -> None:
        expected = content_sha256(self.canonical_payload())
        if self.envelope_sha256 != expected:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
                "channel envelope_sha256 does not match envelope content",
                details={"expected": expected, "actual": self.envelope_sha256},
            )

    @classmethod
    def seal(
        cls,
        *,
        contract: TechniqueChannelContract,
        plan: TechniquePlan,
    ) -> TechniqueChannelEnvelope:
        provisional = cls.model_validate(
            {
                "contract": contract,
                "plan": plan,
                "config": plan.resolved_config,
                "envelope_sha256": "0" * 64,
            },
            context={"skip_envelope_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "envelope_sha256": content_sha256(payload)})


def build_channel_contract(*, plan: TechniquePlan) -> TechniqueChannelContract:
    external = plan.external_execution
    if external is None or plan.training_load_plan is not None:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "channel contract requires an external-only TechniquePlan",
        )
    return TechniqueChannelContract(
        technique_id=plan.technique_id,
        version=plan.version,
        image_uri=external.image_uri,
        image_digest=external.image_digest,
        reviewed_source_repository=external.reviewed_source_repository,
        reviewed_source_commit=external.reviewed_source_commit,
        reviewed_source_tree_sha256=external.reviewed_source_tree_sha256,
        reviewed_source_review_record_sha256=(external.reviewed_source_review_record_sha256),
        descriptor_sha256=plan.descriptor_sha256,
        config_sha256=plan.config_sha256,
        protocol_sha256=plan.protocol_sha256,
        environment_sha256=plan.compatibility.environment_sha256,
        artifact_contract_sha256=content_sha256(plan.artifact_contract.model_dump(mode="json")),
    )


def write_channel_plan(channel_dir: Path, *, plan: TechniquePlan) -> Path:
    """Write one verified envelope. Existing entries are never overwritten."""
    channel_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in channel_dir.iterdir())
    if existing:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel directory must be empty before materialization",
            details={"found": existing},
        )
    envelope = TechniqueChannelEnvelope.seal(
        contract=build_channel_contract(plan=plan),
        plan=plan,
    )
    path = channel_dir / TECHNIQUE_PLAN_FILENAME
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_channel_plan(channel_dir: Path) -> TechniqueChannelEnvelope:
    if not channel_dir.is_dir():
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel directory is missing",
        )
    path = channel_dir / TECHNIQUE_PLAN_FILENAME
    entries = sorted(item.name for item in channel_dir.iterdir())
    if entries != [TECHNIQUE_PLAN_FILENAME] or not path.is_file():
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel must contain exactly one file: technique_plan.json",
            details={"found": entries},
        )
    try:
        return TechniqueChannelEnvelope.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except TechniqueError:
        raise
    except (TypeError, ValueError) as exc:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel envelope is malformed",
            details={"error": str(exc)},
        )


__all__ = [
    "CHANNEL_SCHEMA_VERSION",
    "TECHNIQUE_PLAN_FILENAME",
    "TechniqueChannelContract",
    "TechniqueChannelEnvelope",
    "build_channel_contract",
    "load_channel_plan",
    "write_channel_plan",
]
