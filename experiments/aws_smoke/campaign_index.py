"""Sealed 4/8-GPU independent-run campaign bundle contract and verification."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    PositiveSafeInt,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
)
from distillery.contracts.ids import RunId
from distillery.contracts.manifest import SealedRunManifest
from experiments.aws_smoke.channels import CANONICAL_MANIFEST_FILENAME, load_manifest
from experiments.aws_smoke.profile import RunArm

SEALED_CAMPAIGN_INDEX_FILENAME = "sealed_campaign_index.json"
SEALED_CAMPAIGN_SHA256_FILENAME = "sealed_campaign_index.sha256"
MAX_GPU_COUNT = 8
MICRO_USD_PER_USD = 1_000_000

HardwareProfileId = Literal[
    "g5-12xlarge-4xa10g-independent-v1",
    "g5-48xlarge-8xa10g-independent-v1",
    "p4de-24xlarge-8xa100-80gb-independent-v1",
]
HardwareInstanceType = Literal[
    "ml.g5.12xlarge",
    "ml.g5.48xlarge",
    "ml.p4de.24xlarge",
]
AcceleratorType = Literal["NVIDIA A10G", "NVIDIA A100 80GB"]

CampaignId = Annotated[
    StrictStr,
    Field(pattern=r"^campaign_[a-z0-9][a-z0-9_-]{1,126}$"),
]
GpuSlot = Annotated[StrictInt, Field(ge=0, lt=MAX_GPU_COUNT)]
Ordinal = Annotated[StrictInt, Field(ge=0, lt=MAX_GPU_COUNT)]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EVIDENCE_VALUES = frozenset(
    {"", "unset", "todo", "pending", "placeholder", "replace_me", "tbd"}
)


def _require_nonplaceholder(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if stripped.lower() in _FORBIDDEN_EVIDENCE_VALUES:
        raise ValueError(f"{field_name} is empty or a placeholder")
    return stripped


def _validate_s3_prefix(value: str, *, field_name: str) -> str:
    value = _require_nonplaceholder(value, field_name=field_name)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{field_name} must be a plain s3:// prefix")
    if "\\" in parsed.path:
        raise ValueError(f"{field_name} must not contain backslashes")
    parts = PurePosixPath(parsed.path).parts
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"{field_name} must not contain dot path components")
    if not value.endswith("/"):
        raise ValueError(f"{field_name} must end with '/'")
    return value


def _validate_manifest_relative_path(value: str, *, run_id: str) -> str:
    if "\\" in value:
        raise ValueError("manifest_path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("manifest_path must be a safe relative path")
    expected = PurePosixPath("arms", run_id, "manifest", CANONICAL_MANIFEST_FILENAME)
    if path != expected:
        raise ValueError(f"manifest_path must be exactly {expected.as_posix()!r}")
    return path.as_posix()


class CampaignHardwareProfile(FrozenModel):
    """One supported node topology; price is deliberately supplied separately."""

    profile_id: HardwareProfileId
    instance_type: HardwareInstanceType
    instance_count: Literal[1] = 1
    gpu_count: Literal[4, 8]
    accelerator: AcceleratorType
    execution_mode: Literal["independent_experiment_per_gpu"] = "independent_experiment_per_gpu"

    @model_validator(mode="after")
    def _known_topology(self) -> CampaignHardwareProfile:
        expected = {
            "g5-12xlarge-4xa10g-independent-v1": (
                "ml.g5.12xlarge",
                4,
                "NVIDIA A10G",
            ),
            "g5-48xlarge-8xa10g-independent-v1": (
                "ml.g5.48xlarge",
                8,
                "NVIDIA A10G",
            ),
            "p4de-24xlarge-8xa100-80gb-independent-v1": (
                "ml.p4de.24xlarge",
                8,
                "NVIDIA A100 80GB",
            ),
        }[self.profile_id]
        actual = (self.instance_type, self.gpu_count, self.accelerator)
        if actual != expected:
            raise ValueError(
                f"hardware profile fields do not match {self.profile_id}: "
                f"expected={expected} actual={actual}"
            )
        return self


def campaign_hardware_profile(profile_id: HardwareProfileId) -> CampaignHardwareProfile:
    profiles: dict[str, tuple[HardwareInstanceType, Literal[4, 8], AcceleratorType]] = {
        "g5-12xlarge-4xa10g-independent-v1": (
            "ml.g5.12xlarge",
            4,
            "NVIDIA A10G",
        ),
        "g5-48xlarge-8xa10g-independent-v1": (
            "ml.g5.48xlarge",
            8,
            "NVIDIA A10G",
        ),
        "p4de-24xlarge-8xa100-80gb-independent-v1": (
            "ml.p4de.24xlarge",
            8,
            "NVIDIA A100 80GB",
        ),
    }
    instance_type, gpu_count, accelerator = profiles[profile_id]
    return CampaignHardwareProfile(
        profile_id=profile_id,
        instance_type=instance_type,
        gpu_count=gpu_count,
        accelerator=accelerator,
    )


class CampaignPricingEvidenceReference(FrozenModel):
    """Attested rate and immutable source reference sealed into the campaign."""

    schema_version: Literal["distillery.aws_smoke.pricing_ref.v1"] = (
        "distillery.aws_smoke.pricing_ref.v1"
    )
    reference: StrictStr = Field(min_length=1)
    evidence_sha256: Sha256Hex
    region: StrictStr = Field(min_length=1)
    instance_type: HardwareInstanceType
    hourly_price_microusd: PositiveSafeInt
    currency: Literal["USD"] = "USD"
    attested_by: StrictStr = Field(min_length=1)
    attested_at: AwareDatetime

    @field_validator("reference")
    @classmethod
    def _valid_reference(cls, value: str) -> str:
        value = _require_nonplaceholder(value, field_name="reference")
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "s3"} or not parsed.netloc:
            raise ValueError("reference must be an https:// or s3:// URI")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("reference must not contain credentials")
        return value

    @field_validator("region", "attested_by")
    @classmethod
    def _valid_attestation_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "attestation")
        return _require_nonplaceholder(value, field_name=str(field_name))


class CampaignArmBinding(FrozenModel):
    """One immutable trainer invocation and its isolated GPU/output assignment."""

    ordinal: Ordinal
    arm: RunArm
    run_id: RunId
    manifest_path: StrictStr = Field(min_length=1)
    manifest_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    gpu_slot: GpuSlot
    output_prefix: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def _safe_binding_paths(self) -> CampaignArmBinding:
        _validate_manifest_relative_path(self.manifest_path, run_id=self.run_id)
        _validate_s3_prefix(self.output_prefix, field_name="output_prefix")
        return self


def campaign_protocol_sha256(arms: Sequence[CampaignArmBinding]) -> str:
    """Bind ordered arm protocols, manifest seals, and physical assignments."""
    return content_sha256(
        {
            "schema_version": "distillery.aws_smoke.campaign_protocol.v1",
            "arms": [
                {
                    "ordinal": binding.ordinal,
                    "arm": binding.arm,
                    "run_id": binding.run_id,
                    "manifest_path": binding.manifest_path,
                    "manifest_sha256": binding.manifest_sha256,
                    "protocol_sha256": binding.protocol_sha256,
                    "gpu_slot": binding.gpu_slot,
                    "output_prefix": binding.output_prefix,
                }
                for binding in arms
            ],
        }
    )


class SealedCampaignIndex(FrozenModel):
    """Canonical content-addressed contract for one multi-GPU SageMaker job."""

    schema_version: Literal["distillery.aws_smoke.campaign.v3"] = "distillery.aws_smoke.campaign.v3"
    campaign_id: CampaignId
    created_at: AwareDatetime
    hardware: CampaignHardwareProfile
    pricing: CampaignPricingEvidenceReference
    input_s3_prefix: StrictStr = Field(min_length=1)
    campaign_output_prefix: StrictStr = Field(min_length=1)
    max_runtime_seconds: PositiveSafeInt
    protocol_sha256: Sha256Hex
    arms: tuple[CampaignArmBinding, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_campaign(self) -> SealedCampaignIndex:
        _validate_s3_prefix(self.input_s3_prefix, field_name="input_s3_prefix")
        _validate_s3_prefix(
            self.campaign_output_prefix,
            field_name="campaign_output_prefix",
        )
        if self.pricing.instance_type != self.hardware.instance_type:
            raise ValueError("pricing instance_type must match hardware profile")
        if len(self.arms) > self.hardware.gpu_count:
            raise ValueError("campaign arm count exceeds hardware GPU slots")

        unique_fields = {
            "run_id": [binding.run_id for binding in self.arms],
            "manifest_path": [binding.manifest_path for binding in self.arms],
            "manifest_sha256": [binding.manifest_sha256 for binding in self.arms],
            "gpu_slot": [binding.gpu_slot for binding in self.arms],
            "output_prefix": [binding.output_prefix for binding in self.arms],
        }
        for field_name, values in unique_fields.items():
            if len(values) != len(set(values)):
                raise ValueError(f"campaign contains duplicate {field_name}")

        for expected, binding in enumerate(self.arms):
            if binding.ordinal != expected or binding.gpu_slot != expected:
                raise ValueError(
                    "arms must be ordered deterministically with ordinal == gpu_slot "
                    "== tuple position"
                )
        normalized_outputs = [binding.output_prefix.rstrip("/") + "/" for binding in self.arms]
        for index, left in enumerate(normalized_outputs):
            for right in normalized_outputs[index + 1 :]:
                if left.startswith(right) or right.startswith(left):
                    raise ValueError("campaign arm output prefixes collide or nest")

        expected_protocol = campaign_protocol_sha256(self.arms)
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("protocol_sha256 does not bind the ordered arm set")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def seal_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class VerifiedCampaignBundle:
    root: Path
    index: SealedCampaignIndex
    index_sha256: str
    manifests: tuple[SealedRunManifest, ...]


def _secure_bundle_file(root: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"unsafe campaign bundle path: {relative!r}")
    candidate = root.joinpath(*relative_path.parts)
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"campaign bundle must not contain symlinks: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"campaign bundle file missing: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError(f"campaign bundle path escapes root: {relative!r}")
    return candidate


def _hourly_price_microusd(manifest: SealedRunManifest) -> int:
    raw = manifest.tags.get("HourlyUsd")
    if raw is None:
        raise ValueError("manifest lacks sealed HourlyUsd")
    try:
        micros = Decimal(raw) * MICRO_USD_PER_USD
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("manifest HourlyUsd is not a decimal price") from exc
    if not micros.is_finite() or micros != micros.to_integral_value() or micros <= 0:
        raise ValueError("manifest HourlyUsd must resolve to positive whole micro-USD")
    return int(micros)


def matched_protocol_inputs_sha256(manifest: SealedRunManifest) -> str:
    """Hash protocol inputs that must match while arm and seed may differ."""
    if manifest.tags.get("PortfolioProtocolVersion") == "distillery.portfolio.training_protocol.v2":
        qlora = manifest.training.qlora.model_dump(mode="json")
        qlora.pop("kd_weight")
        qlora.pop("hard_ce_weight")
        qlora.pop("capability_evidence")
        return content_sha256(
            {
                "schema_version": "distillery.portfolio.campaign_inputs.v1",
                "dataset_bundle": manifest.dataset.model_dump(mode="json"),
                "models": manifest.models.model_dump(mode="json"),
                "training": {
                    "seed": manifest.training.seed,
                    "max_steps": manifest.training.max_steps,
                    "token_budget": manifest.training.token_budget,
                    "max_length": manifest.training.max_length,
                    "qlora_without_declared_treatment": qlora,
                },
                "proof_protocol": manifest.proof_protocol.model_dump(mode="json"),
                "runtime": manifest.runtime.model_dump(mode="json"),
                "package_lock_hash": manifest.package_lock_hash,
                "source_revision": manifest.source_revision,
                "license_dispositions": manifest.license_dispositions,
                "portfolio_protocol_version": manifest.tags.get("PortfolioProtocolVersion"),
                "portfolio_training_protocol_sha256": manifest.tags.get(
                    "PortfolioTrainingProtocolSha256"
                ),
                "portfolio_shared_protocol_sha256": manifest.tags.get(
                    "PortfolioSharedProtocolSha256"
                ),
                "portfolio_gate_sha256": manifest.tags.get("PortfolioGateSha256"),
                "max_runtime_seconds": manifest.tags.get("MaxRuntimeInSeconds"),
                "network_isolation": manifest.tags.get("EnableNetworkIsolation"),
            }
        )
    return content_sha256(
        {
            "dataset": manifest.dataset.model_dump(mode="json"),
            "models": manifest.models.model_dump(mode="json"),
            "training": {
                "max_steps": manifest.training.max_steps,
                "token_budget": manifest.training.token_budget,
                "max_length": manifest.training.max_length,
                "qlora": manifest.training.qlora.model_dump(mode="json"),
            },
            "proof_protocol": manifest.proof_protocol.model_dump(mode="json"),
            "runtime": manifest.runtime.model_dump(mode="json"),
            "package_lock_hash": manifest.package_lock_hash,
            "source_revision": manifest.source_revision,
            "license_dispositions": manifest.license_dispositions,
            "max_runtime_seconds": manifest.tags.get("MaxRuntimeInSeconds"),
            "network_isolation": manifest.tags.get("EnableNetworkIsolation"),
        }
    )


def load_sealed_campaign_index(path: Path) -> SealedCampaignIndex:
    if path.name != SEALED_CAMPAIGN_INDEX_FILENAME:
        raise ValueError(
            f"campaign index must use canonical filename {SEALED_CAMPAIGN_INDEX_FILENAME!r}"
        )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"campaign index must be a regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("campaign index must be a JSON object")
    index = SealedCampaignIndex.model_validate(payload)
    if raw != index.canonical_bytes():
        raise ValueError("campaign index is not RFC 8785 canonical JSON")
    return index


def verify_campaign_bundle(
    root: Path,
    *,
    expected_index_sha256: str | None = None,
) -> VerifiedCampaignBundle:
    """Verify the canonical index, exact file inventory, and every run binding."""
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"campaign bundle root must be a directory: {root}")
    index_path = _secure_bundle_file(root, SEALED_CAMPAIGN_INDEX_FILENAME)
    sidecar_path = _secure_bundle_file(root, SEALED_CAMPAIGN_SHA256_FILENAME)
    index = load_sealed_campaign_index(index_path)
    index_sha256 = index.seal_sha256()
    sidecar = sidecar_path.read_bytes()
    expected_sidecar = (index_sha256 + "\n").encode("ascii")
    if sidecar != expected_sidecar:
        raise ValueError("campaign index sidecar does not match canonical index")
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError("campaign index does not match expected SHA-256")

    expected_files = {
        SEALED_CAMPAIGN_INDEX_FILENAME,
        SEALED_CAMPAIGN_SHA256_FILENAME,
        *(binding.manifest_path for binding in index.arms),
    }
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"campaign bundle contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValueError(
            "campaign bundle file inventory mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )

    manifests: list[SealedRunManifest] = []
    shared_dataset: tuple[str, str] | None = None
    shared_image_digest: str | None = None
    shared_protocol_inputs: str | None = None
    for binding in index.arms:
        manifest_path = _secure_bundle_file(root, binding.manifest_path)
        manifest = load_manifest(manifest_path)
        if manifest.seal_sha256() != binding.manifest_sha256:
            raise ValueError(f"{binding.run_id}: manifest SHA-256 mismatch")
        if manifest.run_id != binding.run_id:
            raise ValueError(f"{binding.run_id}: manifest run_id mismatch")
        if manifest.tags.get("Arm") != binding.arm:
            raise ValueError(f"{binding.run_id}: manifest arm mismatch")
        if manifest.tags.get("TrainingProtocolSha256") != binding.protocol_sha256:
            raise ValueError(f"{binding.run_id}: protocol hash mismatch")
        if manifest.runtime.instance_type != index.hardware.instance_type:
            raise ValueError(f"{binding.run_id}: wrong campaign hardware")
        if manifest.runtime.region != index.pricing.region:
            raise ValueError(f"{binding.run_id}: pricing region mismatch")
        if manifest.tags.get("EnableNetworkIsolation") != "true":
            raise ValueError(f"{binding.run_id}: network isolation is not sealed")
        if manifest.output.prefix != binding.output_prefix:
            raise ValueError(f"{binding.run_id}: output prefix mismatch")
        if _hourly_price_microusd(manifest) != index.pricing.hourly_price_microusd:
            raise ValueError(f"{binding.run_id}: attested hourly price mismatch")

        dataset_identity = (manifest.dataset.uri, manifest.dataset.sha256)
        if shared_dataset is None:
            shared_dataset = dataset_identity
            shared_image_digest = manifest.runtime.image_digest
        elif dataset_identity != shared_dataset:
            raise ValueError("campaign manifests do not share one sealed dataset")
        elif manifest.runtime.image_digest != shared_image_digest:
            raise ValueError("campaign manifests do not share one runtime image")
        protocol_inputs = matched_protocol_inputs_sha256(manifest)
        if shared_protocol_inputs is None:
            shared_protocol_inputs = protocol_inputs
        elif protocol_inputs != shared_protocol_inputs:
            raise ValueError("campaign manifests do not share matched protocol inputs")
        manifests.append(manifest)

    return VerifiedCampaignBundle(
        root=root.resolve(strict=True),
        index=index,
        index_sha256=index_sha256,
        manifests=tuple(manifests),
    )


def stage_campaign_bundle(
    *,
    destination: Path,
    campaign_id: str,
    created_at: object,
    ordered_manifest_paths: Sequence[Path],
    hardware: CampaignHardwareProfile,
    pricing: CampaignPricingEvidenceReference,
    input_s3_prefix: str,
    campaign_output_prefix: str,
) -> VerifiedCampaignBundle:
    """Atomically stage all manifests plus one canonical sealed campaign index."""
    if not ordered_manifest_paths or len(ordered_manifest_paths) > hardware.gpu_count:
        raise ValueError(f"campaign requires between one and {hardware.gpu_count} manifests")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"campaign destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_manifests: list[tuple[Path, SealedRunManifest]] = []
    max_runtime_seconds = 0
    for source in ordered_manifest_paths:
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"manifest source must be a regular file: {source}")
        manifest = load_manifest(source)
        try:
            manifest_runtime = int(manifest.tags["MaxRuntimeInSeconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{manifest.run_id}: invalid sealed MaxRuntimeInSeconds") from exc
        if manifest_runtime <= 0:
            raise ValueError(f"{manifest.run_id}: MaxRuntimeInSeconds must be positive")
        max_runtime_seconds = max(max_runtime_seconds, manifest_runtime)
        source_manifests.append((source, manifest))

    bindings: list[CampaignArmBinding] = []
    for ordinal, (_, manifest) in enumerate(source_manifests):
        protocol_sha256 = manifest.tags.get("TrainingProtocolSha256")
        if protocol_sha256 is None or _SHA256_RE.fullmatch(protocol_sha256) is None:
            raise ValueError(f"{manifest.run_id}: missing sealed training protocol hash")
        arm = manifest.tags.get("Arm")
        if arm not in {"oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"}:
            raise ValueError(f"{manifest.run_id}: unsupported campaign arm {arm!r}")
        bindings.append(
            CampaignArmBinding(
                ordinal=ordinal,
                arm=arm,
                run_id=manifest.run_id,
                manifest_path=(
                    PurePosixPath(
                        "arms",
                        manifest.run_id,
                        "manifest",
                        CANONICAL_MANIFEST_FILENAME,
                    ).as_posix()
                ),
                manifest_sha256=manifest.seal_sha256(),
                protocol_sha256=protocol_sha256,
                gpu_slot=ordinal,
                output_prefix=manifest.output.prefix,
            )
        )

    index = SealedCampaignIndex(
        campaign_id=campaign_id,
        created_at=created_at,
        hardware=hardware,
        pricing=pricing,
        input_s3_prefix=input_s3_prefix,
        campaign_output_prefix=campaign_output_prefix,
        max_runtime_seconds=max_runtime_seconds,
        protocol_sha256=campaign_protocol_sha256(bindings),
        arms=tuple(bindings),
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for (source, _), binding in zip(source_manifests, bindings, strict=True):
            target = temporary.joinpath(*PurePosixPath(binding.manifest_path).parts)
            target.parent.mkdir(parents=True, exist_ok=False)
            target.write_bytes(source.read_bytes())
        (temporary / SEALED_CAMPAIGN_INDEX_FILENAME).write_bytes(index.canonical_bytes())
        (temporary / SEALED_CAMPAIGN_SHA256_FILENAME).write_text(
            index.seal_sha256() + "\n",
            encoding="ascii",
        )
        verify_campaign_bundle(
            temporary,
            expected_index_sha256=index.seal_sha256(),
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_campaign_bundle(
        destination,
        expected_index_sha256=index.seal_sha256(),
    )
