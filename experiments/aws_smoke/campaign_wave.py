"""Sealed two-job partitioning for 12–16 independent g5.48xlarge runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictInt, StrictStr, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    NonNegativeSafeInt,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
)
from distillery.contracts.ids import RunId
from experiments.aws_smoke.campaign_index import (
    MICRO_USD_PER_USD,
    CampaignHardwareProfile,
    CampaignPricingEvidenceReference,
    VerifiedCampaignBundle,
    campaign_hardware_profile,
    matched_protocol_inputs_sha256,
    stage_campaign_bundle,
    verify_campaign_bundle,
)
from experiments.aws_smoke.channels import load_manifest

SEALED_WAVE_INDEX_FILENAME = "sealed_wave_index.json"
SEALED_WAVE_SHA256_FILENAME = "sealed_wave_index.sha256"
PRIMARY_WAVE_PROFILE_ID = "g5-48xlarge-8xa10g-independent-v1"
MIN_WAVE_RUNS = 12
MAX_WAVE_RUNS = 16
WAVE_JOB_COUNT = 2

WaveId = Annotated[
    StrictStr,
    Field(pattern=r"^wave_[a-z0-9][a-z0-9_-]{1,126}$"),
]
WaveOrdinal = Annotated[StrictInt, Field(ge=0, lt=WAVE_JOB_COUNT)]


def _validate_s3_prefix(value: str, *, field_name: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or not value.endswith("/")
    ):
        raise ValueError(f"{field_name} must be a plain s3:// prefix ending in '/'")


def _require_disjoint_prefixes(values: Sequence[str], *, field_name: str) -> None:
    normalized = [value.rstrip("/") + "/" for value in values]
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if left.startswith(right) or right.startswith(left):
                raise ValueError(f"wave {field_name} prefixes collide or nest")


class WaveCampaignBinding(FrozenModel):
    ordinal: WaveOrdinal
    campaign_id: StrictStr = Field(pattern=r"^campaign_[a-z0-9][a-z0-9_-]{1,126}$")
    bundle_path: StrictStr = Field(min_length=1)
    campaign_index_sha256: Sha256Hex
    campaign_protocol_sha256: Sha256Hex
    input_s3_prefix: StrictStr = Field(min_length=1)
    output_s3_prefix: StrictStr = Field(min_length=1)
    run_ids: tuple[RunId, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _safe_bundle_path(self) -> WaveCampaignBinding:
        path = PurePosixPath(self.bundle_path)
        expected = PurePosixPath("campaigns", self.campaign_id)
        if path.is_absolute() or path != expected or ".." in path.parts:
            raise ValueError(f"bundle_path must be exactly {expected.as_posix()!r}")
        _validate_s3_prefix(self.input_s3_prefix, field_name="input_s3_prefix")
        _validate_s3_prefix(self.output_s3_prefix, field_name="output_s3_prefix")
        return self


def wave_protocol_sha256(
    *,
    hardware: CampaignHardwareProfile,
    pricing: CampaignPricingEvidenceReference,
    ordered_run_ids: Sequence[str],
    campaigns: Sequence[WaveCampaignBinding],
) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.aws_smoke.wave_protocol.v1",
            "hardware": hardware.model_dump(mode="json"),
            "pricing_evidence_sha256": pricing.evidence_sha256,
            "ordered_run_ids": list(ordered_run_ids),
            "campaigns": [
                {
                    "ordinal": campaign.ordinal,
                    "campaign_index_sha256": campaign.campaign_index_sha256,
                    "campaign_protocol_sha256": campaign.campaign_protocol_sha256,
                    "run_ids": list(campaign.run_ids),
                }
                for campaign in campaigns
            ],
        }
    )


class SealedWaveIndex(FrozenModel):
    schema_version: Literal["distillery.aws_smoke.wave.v1"] = "distillery.aws_smoke.wave.v1"
    wave_id: WaveId
    created_at: AwareDatetime
    hardware: CampaignHardwareProfile
    pricing: CampaignPricingEvidenceReference
    ordered_run_ids: tuple[RunId, ...] = Field(
        min_length=MIN_WAVE_RUNS,
        max_length=MAX_WAVE_RUNS,
    )
    campaigns: tuple[WaveCampaignBinding, WaveCampaignBinding]
    protocol_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_wave(self) -> SealedWaveIndex:
        if self.hardware.profile_id != PRIMARY_WAVE_PROFILE_ID:
            raise ValueError("two-job wave requires the approved g5.48xlarge profile")
        if self.pricing.instance_type != self.hardware.instance_type:
            raise ValueError("wave pricing evidence does not match hardware")
        for expected, campaign in enumerate(self.campaigns):
            if campaign.ordinal != expected:
                raise ValueError("wave campaigns must be ordered by ordinal")
            if not 1 <= len(campaign.run_ids) <= self.hardware.gpu_count:
                raise ValueError("wave campaign run count exceeds hardware slots")

        unique_fields = {
            "campaign_id": [campaign.campaign_id for campaign in self.campaigns],
            "campaign_index_sha256": [
                campaign.campaign_index_sha256 for campaign in self.campaigns
            ],
            "input_s3_prefix": [campaign.input_s3_prefix for campaign in self.campaigns],
            "output_s3_prefix": [campaign.output_s3_prefix for campaign in self.campaigns],
        }
        for field_name, values in unique_fields.items():
            if len(values) != len(set(values)):
                raise ValueError(f"wave contains duplicate {field_name}")
        _require_disjoint_prefixes(
            [campaign.input_s3_prefix for campaign in self.campaigns],
            field_name="input",
        )
        _require_disjoint_prefixes(
            [campaign.output_s3_prefix for campaign in self.campaigns],
            field_name="output",
        )

        all_run_ids = [run_id for campaign in self.campaigns for run_id in campaign.run_ids]
        if len(all_run_ids) != len(set(all_run_ids)):
            raise ValueError("wave contains duplicate run_id across campaign jobs")
        if set(all_run_ids) != set(self.ordered_run_ids):
            raise ValueError("wave campaign partitions do not cover ordered_run_ids")

        reconstructed: list[str] = []
        for slot in range(max(len(campaign.run_ids) for campaign in self.campaigns)):
            for campaign in self.campaigns:
                if slot < len(campaign.run_ids):
                    reconstructed.append(campaign.run_ids[slot])
        if tuple(reconstructed) != self.ordered_run_ids:
            raise ValueError("wave partitions are not the deterministic round-robin split")

        expected_protocol = wave_protocol_sha256(
            hardware=self.hardware,
            pricing=self.pricing,
            ordered_run_ids=self.ordered_run_ids,
            campaigns=self.campaigns,
        )
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("wave protocol hash does not bind both campaign jobs")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def seal_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class VerifiedWaveBundle:
    root: Path
    index: SealedWaveIndex
    index_sha256: str
    campaigns: tuple[VerifiedCampaignBundle, VerifiedCampaignBundle]


class ParentCampaignCost(FrozenModel):
    campaign_index_sha256: Sha256Hex
    parent_cost_microusd: NonNegativeSafeInt


class AggregateWaveCost(FrozenModel):
    schema_version: Literal["distillery.aws_smoke.wave_cost.v1"] = (
        "distillery.aws_smoke.wave_cost.v1"
    )
    wave_index_sha256: Sha256Hex
    parents: tuple[ParentCampaignCost, ParentCampaignCost]
    aggregate_parent_cost_microusd: NonNegativeSafeInt


def aggregate_wave_cost(
    wave: VerifiedWaveBundle | SealedWaveIndex,
    parent_costs: Sequence[ParentCampaignCost],
) -> AggregateWaveCost:
    index = wave.index if isinstance(wave, VerifiedWaveBundle) else wave
    wave_sha256 = wave.index_sha256 if isinstance(wave, VerifiedWaveBundle) else index.seal_sha256()
    if len(parent_costs) != WAVE_JOB_COUNT:
        raise ValueError("wave cost requires exactly two parent campaign costs")
    by_sha: dict[str, ParentCampaignCost] = {}
    for cost in parent_costs:
        if cost.campaign_index_sha256 in by_sha:
            raise ValueError("duplicate parent campaign cost would double count spend")
        by_sha[cost.campaign_index_sha256] = cost
    expected = [campaign.campaign_index_sha256 for campaign in index.campaigns]
    if set(by_sha) != set(expected):
        raise ValueError("parent campaign costs do not exactly cover the wave")
    ordered = tuple(by_sha[index_sha] for index_sha in expected)
    total = sum(cost.parent_cost_microusd for cost in ordered)
    return AggregateWaveCost(
        wave_index_sha256=wave_sha256,
        parents=ordered,
        aggregate_parent_cost_microusd=total,
    )


def _manifest_price_microusd(path: Path) -> int:
    manifest = load_manifest(path)
    raw = manifest.tags.get("HourlyUsd")
    if raw is None:
        raise ValueError(f"{manifest.run_id}: missing sealed HourlyUsd")
    try:
        micros = Decimal(raw) * MICRO_USD_PER_USD
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{manifest.run_id}: invalid HourlyUsd") from exc
    if not micros.is_finite() or micros != micros.to_integral_value() or micros <= 0:
        raise ValueError(f"{manifest.run_id}: HourlyUsd is not whole micro-USD")
    return int(micros)


def partition_wave_manifests(
    ordered_manifest_paths: Sequence[Path],
    *,
    hardware: CampaignHardwareProfile,
    pricing: CampaignPricingEvidenceReference,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Validate and deterministically balance 12–16 sealed runs across two nodes."""
    if not MIN_WAVE_RUNS <= len(ordered_manifest_paths) <= MAX_WAVE_RUNS:
        raise ValueError("g5.48 wave requires 12–16 sealed runs")
    if hardware.profile_id != PRIMARY_WAVE_PROFILE_ID or hardware.gpu_count != 8:
        raise ValueError("two-node wave partitioning requires 8-GPU g5.48xlarge")
    if pricing.instance_type != hardware.instance_type:
        raise ValueError("pricing evidence does not match selected wave hardware")

    run_ids: list[str] = []
    output_prefixes: list[str] = []
    manifest_hashes: list[str] = []
    shared_identity: str | None = None
    for path in ordered_manifest_paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"wave manifest must be a regular file: {path}")
        manifest = load_manifest(path)
        if manifest.runtime.instance_type != hardware.instance_type:
            raise ValueError(f"{manifest.run_id}: wave hardware mismatch")
        if manifest.runtime.region != pricing.region:
            raise ValueError(f"{manifest.run_id}: wave pricing region mismatch")
        if _manifest_price_microusd(path) != pricing.hourly_price_microusd:
            raise ValueError(f"{manifest.run_id}: wave price mismatch")
        identity = matched_protocol_inputs_sha256(manifest)
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError("wave manifests do not share matched protocol inputs")
        run_ids.append(manifest.run_id)
        output_prefixes.append(manifest.output.prefix)
        manifest_hashes.append(manifest.seal_sha256())

    for field_name, values in (
        ("run_id", run_ids),
        ("output_prefix", output_prefixes),
        ("manifest_sha256", manifest_hashes),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"wave contains duplicate {field_name} across nodes")
    _require_disjoint_prefixes(output_prefixes, field_name="arm output")

    partitions = (
        tuple(ordered_manifest_paths[0::2]),
        tuple(ordered_manifest_paths[1::2]),
    )
    if any(len(partition) > hardware.gpu_count for partition in partitions):
        raise ValueError("wave partition exceeds node GPU slots")
    return partitions


def load_sealed_wave_index(path: Path) -> SealedWaveIndex:
    if path.name != SEALED_WAVE_INDEX_FILENAME:
        raise ValueError(f"wave index must use canonical filename {SEALED_WAVE_INDEX_FILENAME!r}")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"wave index must be a regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("wave index must be a JSON object")
    index = SealedWaveIndex.model_validate(payload)
    if raw != index.canonical_bytes():
        raise ValueError("wave index is not RFC 8785 canonical JSON")
    return index


def verify_wave_bundle(
    root: Path,
    *,
    expected_index_sha256: str | None = None,
) -> VerifiedWaveBundle:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"wave bundle root must be a directory: {root}")
    index_path = root / SEALED_WAVE_INDEX_FILENAME
    sidecar_path = root / SEALED_WAVE_SHA256_FILENAME
    if index_path.is_symlink() or sidecar_path.is_symlink():
        raise ValueError("wave index files must not be symlinks")
    index = load_sealed_wave_index(index_path)
    index_sha256 = index.seal_sha256()
    if sidecar_path.read_bytes() != (index_sha256 + "\n").encode("ascii"):
        raise ValueError("wave index sidecar mismatch")
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError("wave index does not match expected SHA-256")

    expected_top_level = {
        SEALED_WAVE_INDEX_FILENAME,
        SEALED_WAVE_SHA256_FILENAME,
        "campaigns",
    }
    if {path.name for path in root.iterdir()} != expected_top_level:
        raise ValueError("wave bundle top-level inventory mismatch")
    campaigns_root = root / "campaigns"
    if campaigns_root.is_symlink() or not campaigns_root.is_dir():
        raise ValueError("wave campaigns inventory must be a regular directory")
    expected_campaign_dirs = {campaign.campaign_id for campaign in index.campaigns}
    actual_campaign_dirs = {path.name for path in campaigns_root.iterdir()}
    if actual_campaign_dirs != expected_campaign_dirs:
        raise ValueError("wave campaign directory inventory mismatch")

    campaigns: list[VerifiedCampaignBundle] = []
    global_run_ids: list[str] = []
    global_outputs: list[str] = []
    shared_identity: str | None = None
    for binding in index.campaigns:
        campaign_root = root.joinpath(*PurePosixPath(binding.bundle_path).parts)
        campaign = verify_campaign_bundle(
            campaign_root,
            expected_index_sha256=binding.campaign_index_sha256,
        )
        if campaign.index.hardware != index.hardware:
            raise ValueError("wave campaign hardware mismatch")
        if campaign.index.pricing != index.pricing:
            raise ValueError("wave campaign pricing mismatch")
        if campaign.index.protocol_sha256 != binding.campaign_protocol_sha256:
            raise ValueError("wave campaign protocol mismatch")
        if tuple(arm.run_id for arm in campaign.index.arms) != binding.run_ids:
            raise ValueError("wave campaign run binding mismatch")
        if campaign.index.input_s3_prefix != binding.input_s3_prefix:
            raise ValueError("wave campaign input prefix mismatch")
        if campaign.index.campaign_output_prefix != binding.output_s3_prefix:
            raise ValueError("wave campaign output prefix mismatch")
        global_run_ids.extend(binding.run_ids)
        global_outputs.extend(arm.output_prefix for arm in campaign.index.arms)
        for manifest in campaign.manifests:
            identity = matched_protocol_inputs_sha256(manifest)
            if shared_identity is None:
                shared_identity = identity
            elif identity != shared_identity:
                raise ValueError("wave campaigns do not share matched protocol inputs")
        campaigns.append(campaign)

    if len(global_run_ids) != len(set(global_run_ids)):
        raise ValueError("cross-node duplicate run_id")
    if len(global_outputs) != len(set(global_outputs)):
        raise ValueError("cross-node duplicate output prefix")
    _require_disjoint_prefixes(global_outputs, field_name="arm output")
    return VerifiedWaveBundle(
        root=root.resolve(strict=True),
        index=index,
        index_sha256=index_sha256,
        campaigns=(campaigns[0], campaigns[1]),
    )


def stage_two_job_wave(
    *,
    destination: Path,
    wave_id: str,
    created_at: object,
    ordered_manifest_paths: Sequence[Path],
    hardware: CampaignHardwareProfile,
    pricing: CampaignPricingEvidenceReference,
    wave_input_s3_prefix: str,
    wave_output_s3_prefix: str,
) -> VerifiedWaveBundle:
    """Atomically stage two balanced g5.48 campaign bundles and their wave seal."""
    partitions = partition_wave_manifests(
        ordered_manifest_paths,
        hardware=hardware,
        pricing=pricing,
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"wave destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        campaign_bindings: list[WaveCampaignBinding] = []
        for ordinal, partition in enumerate(partitions):
            campaign_id = f"campaign_{wave_id.removeprefix('wave_')}-node-{ordinal}"
            input_prefix = wave_input_s3_prefix.rstrip("/") + f"/campaign-{ordinal}/"
            output_prefix = wave_output_s3_prefix.rstrip("/") + f"/campaign-{ordinal}/"
            campaign = stage_campaign_bundle(
                destination=temporary / "campaigns" / campaign_id,
                campaign_id=campaign_id,
                created_at=created_at,
                ordered_manifest_paths=partition,
                hardware=hardware,
                pricing=pricing,
                input_s3_prefix=input_prefix,
                campaign_output_prefix=output_prefix,
            )
            campaign_bindings.append(
                WaveCampaignBinding(
                    ordinal=ordinal,
                    campaign_id=campaign_id,
                    bundle_path=f"campaigns/{campaign_id}",
                    campaign_index_sha256=campaign.index_sha256,
                    campaign_protocol_sha256=campaign.index.protocol_sha256,
                    input_s3_prefix=input_prefix,
                    output_s3_prefix=output_prefix,
                    run_ids=tuple(arm.run_id for arm in campaign.index.arms),
                )
            )

        ordered_run_ids = tuple(load_manifest(path).run_id for path in ordered_manifest_paths)
        index = SealedWaveIndex(
            wave_id=wave_id,
            created_at=created_at,
            hardware=hardware,
            pricing=pricing,
            ordered_run_ids=ordered_run_ids,
            campaigns=(campaign_bindings[0], campaign_bindings[1]),
            protocol_sha256=wave_protocol_sha256(
                hardware=hardware,
                pricing=pricing,
                ordered_run_ids=ordered_run_ids,
                campaigns=campaign_bindings,
            ),
        )
        (temporary / SEALED_WAVE_INDEX_FILENAME).write_bytes(index.canonical_bytes())
        (temporary / SEALED_WAVE_SHA256_FILENAME).write_text(
            index.seal_sha256() + "\n",
            encoding="ascii",
        )
        verify_wave_bundle(temporary, expected_index_sha256=index.seal_sha256())
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_wave_bundle(
        destination,
        expected_index_sha256=index.seal_sha256(),
    )


def primary_wave_hardware() -> CampaignHardwareProfile:
    return campaign_hardware_profile(PRIMARY_WAVE_PROFILE_ID)
