"""Network-isolated channel contract for external technique containers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictStr

from distillery.backends.safety import (
    NETWORK_ISOLATION_TAG,
    parse_digest_pinned_ecr_image,
)
from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import PrefixedSha256, content_sha256
from distillery.techniques.descriptor import PluginImageBinding, TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error

TECHNIQUE_PLAN_FILENAME = "technique_plan.json"
CHANNEL_SCHEMA_VERSION: Literal["distillery.technique.channel.v1"] = (
    "distillery.technique.channel.v1"
)


class TechniqueChannelContract(FrozenModel):
    """
    Standard channel layout for external techniques.

    External code executes only inside the digest-pinned container referenced
    here. The control plane never imports plugin Python modules.
    """

    schema_version: Literal["distillery.technique.channel.v1"] = CHANNEL_SCHEMA_VERSION
    technique_id: StrictStr
    version: StrictStr
    plan_filename: Literal["technique_plan.json"] = TECHNIQUE_PLAN_FILENAME
    image_uri: StrictStr = Field(min_length=1)
    image_digest: PrefixedSha256
    enable_network_isolation: Literal[True] = True
    network_isolation_tag: Literal["EnableNetworkIsolation"] = NETWORK_ISOLATION_TAG
    reviewed_source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    reviewed_source_tree_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    extras: FrozenJsonObject = Field(default_factory=dict)

    def channel_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


def build_channel_contract(
    *,
    descriptor: TechniqueDescriptor,
    config_sha256: str,
    protocol_sha256: str,
) -> TechniqueChannelContract:
    if descriptor.plugin_image is None or descriptor.reviewed_source is None:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "channel contract requires sealed plugin image and reviewed source",
            details={
                "technique_id": descriptor.technique_id,
                "version": descriptor.version,
            },
        )
    _assert_plugin_image(descriptor.plugin_image)
    return TechniqueChannelContract(
        technique_id=descriptor.technique_id,
        version=descriptor.version,
        image_uri=descriptor.plugin_image.image_uri,
        image_digest=descriptor.plugin_image.image_digest,
        reviewed_source_commit=descriptor.reviewed_source.commit_sha,
        reviewed_source_tree_sha256=descriptor.reviewed_source.source_tree_sha256,
        descriptor_sha256=descriptor.descriptor_sha256,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )


def write_channel_plan(
    channel_dir: Path,
    *,
    contract: TechniqueChannelContract,
    plan_payload: dict,
) -> Path:
    """Materialize the canonical technique plan file into a channel directory."""
    channel_dir.mkdir(parents=True, exist_ok=True)
    path = channel_dir / TECHNIQUE_PLAN_FILENAME
    existing = sorted(p.name for p in channel_dir.iterdir() if p.suffix == ".json")
    if existing and existing != [TECHNIQUE_PLAN_FILENAME]:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel must contain only the canonical plan JSON",
            details={"found": existing},
        )
    envelope = {
        "contract": contract.model_dump(mode="json"),
        "plan": plan_payload,
    }
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_channel_plan(channel_dir: Path) -> tuple[TechniqueChannelContract, dict]:
    path = channel_dir / TECHNIQUE_PLAN_FILENAME
    json_files = sorted(p.name for p in channel_dir.iterdir() if p.suffix == ".json")
    if json_files != [TECHNIQUE_PLAN_FILENAME] or not path.is_file():
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel must contain exactly technique_plan.json",
            details={"found": json_files},
        )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    contract = TechniqueChannelContract.model_validate(envelope["contract"])
    plan = envelope["plan"]
    if not isinstance(plan, dict):
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID,
            "technique channel plan payload must be an object",
        )
    return contract, plan


def forbid_control_plane_import(module_name: str) -> None:
    """Hard gate: external technique modules must not be imported in-process."""
    raise_technique_error(
        TechniqueErrorCode.TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN,
        "external technique code cannot be imported into the control plane",
        details={
            "module_name": module_name,
            "execution": "external_container",
            "required_path": "digest-pinned network-isolated container channel",
        },
    )


def _assert_plugin_image(binding: PluginImageBinding) -> None:
    try:
        identity = parse_digest_pinned_ecr_image(binding.image_uri)
    except Exception as exc:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID,
            "plugin image URI must be a digest-pinned private ECR image",
            details={"image_uri": binding.image_uri, "error": str(exc)},
        )
        return
    if identity.digest != binding.image_digest:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID,
            "channel image digest mismatch",
            details={
                "image_uri_digest": identity.digest,
                "image_digest": binding.image_digest,
            },
        )


__all__ = [
    "CHANNEL_SCHEMA_VERSION",
    "TECHNIQUE_PLAN_FILENAME",
    "TechniqueChannelContract",
    "build_channel_contract",
    "forbid_control_plane_import",
    "load_channel_plan",
    "write_channel_plan",
]
