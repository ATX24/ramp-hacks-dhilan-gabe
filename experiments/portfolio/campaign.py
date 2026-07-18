"""Portfolio adapters for reviewed campaign contracts and plan-only AWS requests."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictStr, model_validator

from distillery.contracts.base import FrozenDict, FrozenModel
from distillery.contracts.hashing import (
    NonNegativeSafeInt,
    PositiveSafeInt,
    PrefixedSha256,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
    sha256_hex,
)
from distillery.contracts.ids import RunId
from experiments.aws_smoke.campaign_index import (
    CampaignHardwareProfile,
    CampaignPricingEvidenceReference,
    VerifiedCampaignBundle,
    campaign_hardware_profile,
    stage_campaign_bundle,
    verify_campaign_bundle,
)
from experiments.aws_smoke.campaign_wave import WaveCampaignBinding
from experiments.aws_smoke.channels import load_manifest
from experiments.aws_smoke.pins import parse_digest_pinned_ecr_image
from experiments.portfolio.materialize import validate_materialized_manifest
from experiments.portfolio.plan import (
    ACCOUNT_CEILING_MICROUSD,
    NotStartedSlot,
    PlannedRunSlot,
    PortfolioPlan,
    PricingEvidence,
    Tier,
    Wave,
    WaveCost,
    cost,
)

PORTFOLIO_WAVE_INDEX_FILENAME = "sealed_portfolio_wave_index.json"
PORTFOLIO_WAVE_SHA256_FILENAME = "sealed_portfolio_wave_index.sha256"
CONTAINER_PYTHON = "/opt/conda/bin/python"
CAMPAIGN_ORCHESTRATOR_MODULE = "experiments.aws_smoke.campaign_orchestrator"
PORTFOLIO_RUNTIME_MODULE = "experiments.portfolio.task_filter_runtime"
REQUIRED_IMAGE_MODULES = frozenset(
    {
        "experiments.aws_smoke.campaign_index",
        CAMPAIGN_ORCHESTRATOR_MODULE,
        "experiments.aws_smoke.train",
        PORTFOLIO_RUNTIME_MODULE,
    }
)


def _s3_prefix(value: str, *, field_name: str) -> str:
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
    return value


class PortfolioSlotBinding(FrozenModel):
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    state: Literal["planned", "not_started"]
    run_id: RunId | None
    manifest_sha256: Sha256Hex | None
    allocated_ceiling_microusd: NonNegativeSafeInt

    @model_validator(mode="after")
    def _binding(self) -> PortfolioSlotBinding:
        if (self.node, self.gpu) != (self.slot % 2, self.slot // 2):
            raise ValueError("portfolio campaign slot changed node/GPU identity")
        if self.state == "planned" and (self.run_id is None or self.manifest_sha256 is None):
            raise ValueError("planned campaign slot requires run and manifest bindings")
        if self.state == "not_started" and (
            self.run_id is not None or self.manifest_sha256 is not None
        ):
            raise ValueError("not-started slot cannot claim a staged manifest")
        return self


class PortfolioWaveIndex(FrozenModel):
    schema_version: Literal["distillery.portfolio.campaign_wave.v1"] = (
        "distillery.portfolio.campaign_wave.v1"
    )
    wave_id: StrictStr
    wave_matrix_sha256: Sha256Hex
    plan_sha256: Sha256Hex
    hardware: CampaignHardwareProfile
    pricing: CampaignPricingEvidenceReference
    campaigns: tuple[WaveCampaignBinding, WaveCampaignBinding]
    slots: tuple[PortfolioSlotBinding, ...] = Field(min_length=16, max_length=16)
    aggregate_ceiling_microusd: PositiveSafeInt
    index_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bound(self) -> PortfolioWaveIndex:
        if [slot.slot for slot in self.slots] != list(range(16)):
            raise ValueError("portfolio campaign must preserve all sixteen slot identities")
        if tuple(campaign.ordinal for campaign in self.campaigns) != (0, 1):
            raise ValueError("portfolio campaign bindings must be node ordered")
        if self.pricing.instance_type != self.hardware.instance_type:
            raise ValueError("portfolio wave pricing/hardware mismatch")
        for node, campaign in enumerate(self.campaigns):
            expected_runs = tuple(
                slot.run_id for slot in self.slots if slot.node == node and slot.state == "planned"
            )
            if campaign.run_ids != expected_runs:
                raise ValueError("campaign run order changed logical GPU assignments")
        if sum(slot.allocated_ceiling_microusd for slot in self.slots) != (
            self.aggregate_ceiling_microusd
        ):
            raise ValueError("portfolio wave cost omits or double-counts slots")
        if self.index_sha256 != _portfolio_wave_index_hash(self):
            raise ValueError("portfolio wave index hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _portfolio_wave_index_hash(value: PortfolioWaveIndex) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"index_sha256"}))


@dataclass(frozen=True, slots=True)
class VerifiedPortfolioWave:
    root: Path
    index: PortfolioWaveIndex
    index_sha256: str
    campaigns: tuple[VerifiedCampaignBundle, VerifiedCampaignBundle]


def _wave_cost(plan: PortfolioPlan, wave: Wave) -> WaveCost:
    matches = [item for item in plan.costs.waves if item.wave_id == wave.wave_id]
    if len(matches) == 1:
        return matches[0]
    price = plan.pricing[0 if wave.tier is Tier.NANO else 1]
    return cost(wave, price, plan.protocol, plan.ceilings)


def _price(plan: PortfolioPlan, wave: Wave) -> PricingEvidence:
    price = plan.pricing[0 if wave.tier is Tier.NANO else 1]
    if price.instance_type != wave.instance_type:
        raise ValueError("portfolio wave selected wrong pricing evidence")
    return price


def _validate_active_manifest(
    *,
    plan: PortfolioPlan,
    wave: Wave,
    slot: PlannedRunSlot,
    path: Path,
) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"portfolio manifest must be a regular file: {path}")
    manifest = load_manifest(path)
    validate_materialized_manifest(plan=plan, wave=wave, slot=slot, manifest=manifest)
    if manifest.run_id != slot.run_id:
        raise ValueError("portfolio manifest run ID differs from logical slot")
    return manifest.seal_sha256()


def stage_portfolio_wave(
    *,
    destination: Path,
    plan: PortfolioPlan,
    wave: Wave,
    manifest_paths_by_model_id: Mapping[str, Path],
    wave_input_s3_prefix: str,
    wave_output_s3_prefix: str,
) -> VerifiedPortfolioWave:
    """Stage two fixed-node campaign bundles without using the g5-only wave helper."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"portfolio wave destination already exists: {destination}")
    _s3_prefix(wave_input_s3_prefix, field_name="wave input prefix")
    _s3_prefix(wave_output_s3_prefix, field_name="wave output prefix")
    expected_models = {slot.model_id for slot in wave.active_slots}
    if set(manifest_paths_by_model_id) != expected_models:
        raise ValueError("manifest mapping must exactly cover active wave models")
    manifests_by_slot: dict[int, tuple[Path, str]] = {}
    for slot in wave.active_slots:
        path = manifest_paths_by_model_id[slot.model_id]
        manifests_by_slot[slot.slot] = (
            path,
            _validate_active_manifest(plan=plan, wave=wave, slot=slot, path=path),
        )
    for node in (0, 1):
        active = sorted(
            (slot for slot in wave.active_slots if slot.node == node),
            key=lambda slot: slot.gpu,
        )
        if [slot.gpu for slot in active] != list(range(len(active))):
            raise ValueError(
                "reviewed campaign index requires active GPUs to be a fixed prefix; "
                "do not repartition around holes"
            )
    price = _price(plan, wave)
    pricing = price.campaign_reference()
    hardware = campaign_hardware_profile(wave.hardware_profile)
    if (
        hardware.instance_type,
        hardware.accelerator,
    ) != (wave.instance_type, wave.accelerator):
        raise ValueError("campaign adapter failed to resolve accelerator labels")
    wave_cost = _wave_cost(plan, wave)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        campaign_bindings: list[WaveCampaignBinding] = []
        campaigns: list[VerifiedCampaignBundle] = []
        for node in (0, 1):
            node_slots = sorted(
                (slot for slot in wave.active_slots if slot.node == node),
                key=lambda slot: slot.gpu,
            )
            campaign_id = f"campaign_{wave.wave_id.removeprefix('wave_')}_node_{node}"
            input_prefix = wave_input_s3_prefix.rstrip("/") + f"/node-{node}/"
            output_prefix = wave_output_s3_prefix.rstrip("/") + f"/node-{node}/"
            campaign = stage_campaign_bundle(
                destination=temporary / "campaigns" / campaign_id,
                campaign_id=campaign_id,
                created_at=plan.created_at,
                ordered_manifest_paths=[manifests_by_slot[slot.slot][0] for slot in node_slots],
                hardware=hardware,
                pricing=pricing,
                input_s3_prefix=input_prefix,
                campaign_output_prefix=output_prefix,
            )
            campaigns.append(campaign)
            campaign_bindings.append(
                WaveCampaignBinding(
                    ordinal=node,  # type: ignore[arg-type]
                    campaign_id=campaign_id,
                    bundle_path=f"campaigns/{campaign_id}",
                    campaign_index_sha256=campaign.index_sha256,
                    campaign_protocol_sha256=campaign.index.protocol_sha256,
                    input_s3_prefix=input_prefix,
                    output_s3_prefix=output_prefix,
                    run_ids=tuple(slot.run_id for slot in node_slots),
                )
            )
        by_slot_cost = {item.slot: item for item in wave_cost.slots}
        slot_bindings = tuple(
            PortfolioSlotBinding(
                slot=slot.slot,
                node=slot.node,
                gpu=slot.gpu,
                state=slot.state,
                run_id=slot.run_id if isinstance(slot, PlannedRunSlot) else None,
                manifest_sha256=(
                    manifests_by_slot[slot.slot][1] if isinstance(slot, PlannedRunSlot) else None
                ),
                allocated_ceiling_microusd=by_slot_cost[slot.slot].allocated_ceiling_microusd,
            )
            for slot in wave.slots
        )
        provisional = PortfolioWaveIndex.model_construct(
            wave_id=wave.wave_id,
            wave_matrix_sha256=wave.matrix_sha256,
            plan_sha256=plan.plan_sha256,
            hardware=hardware,
            pricing=pricing,
            campaigns=(campaign_bindings[0], campaign_bindings[1]),
            slots=slot_bindings,
            aggregate_ceiling_microusd=wave_cost.aggregate_ceiling_microusd,
            index_sha256="0" * 64,
        )
        index = PortfolioWaveIndex.model_validate(
            {
                **provisional.model_dump(mode="python"),
                "index_sha256": _portfolio_wave_index_hash(provisional),
            }
        )
        (temporary / PORTFOLIO_WAVE_INDEX_FILENAME).write_bytes(index.canonical_bytes())
        (temporary / PORTFOLIO_WAVE_SHA256_FILENAME).write_text(
            index.index_sha256 + "\n",
            encoding="ascii",
        )
        verified = verify_portfolio_wave(
            temporary,
            plan=plan,
            wave=wave,
            expected_index_sha256=index.index_sha256,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_portfolio_wave(
        destination,
        plan=plan,
        wave=wave,
        expected_index_sha256=verified.index_sha256,
    )


def verify_portfolio_wave(
    root: Path,
    *,
    plan: PortfolioPlan,
    wave: Wave,
    expected_index_sha256: str | None = None,
) -> VerifiedPortfolioWave:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"portfolio wave root must be a directory: {root}")
    index_path = root / PORTFOLIO_WAVE_INDEX_FILENAME
    sidecar_path = root / PORTFOLIO_WAVE_SHA256_FILENAME
    if index_path.is_symlink() or sidecar_path.is_symlink():
        raise ValueError("portfolio wave index files cannot be symlinks")
    raw = index_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("portfolio wave index must be a JSON object")
    index = PortfolioWaveIndex.model_validate(payload)
    if raw != index.canonical_bytes():
        raise ValueError("portfolio wave index is not RFC 8785 canonical JSON")
    if sidecar_path.read_bytes() != (index.index_sha256 + "\n").encode("ascii"):
        raise ValueError("portfolio wave index sidecar mismatch")
    if expected_index_sha256 is not None and index.index_sha256 != expected_index_sha256:
        raise ValueError("portfolio wave index SHA-256 mismatch")
    if (
        index.wave_id,
        index.wave_matrix_sha256,
        index.plan_sha256,
    ) != (wave.wave_id, wave.matrix_sha256, plan.plan_sha256):
        raise ValueError("portfolio campaign is not bound to the supplied plan and wave")
    campaigns: list[VerifiedCampaignBundle] = []
    for binding in index.campaigns:
        campaign = verify_campaign_bundle(
            root.joinpath(*PurePosixPath(binding.bundle_path).parts),
            expected_index_sha256=binding.campaign_index_sha256,
        )
        if (
            campaign.index.hardware,
            campaign.index.pricing,
            campaign.index.protocol_sha256,
            tuple(arm.run_id for arm in campaign.index.arms),
        ) != (
            index.hardware,
            index.pricing,
            binding.campaign_protocol_sha256,
            binding.run_ids,
        ):
            raise ValueError("portfolio child campaign binding mismatch")
        campaigns.append(campaign)
    manifest_by_run = {
        manifest.run_id: manifest for campaign in campaigns for manifest in campaign.manifests
    }
    for slot, binding in zip(wave.slots, index.slots, strict=True):
        if isinstance(slot, PlannedRunSlot):
            manifest = manifest_by_run.get(slot.run_id)
            if manifest is None or manifest.seal_sha256() != binding.manifest_sha256:
                raise ValueError("portfolio active slot manifest missing or changed")
            validate_materialized_manifest(
                plan=plan,
                wave=wave,
                slot=slot,
                manifest=manifest,
            )
        elif not isinstance(slot, NotStartedSlot) or binding.state != "not_started":
            raise ValueError("portfolio not-started slot identity changed")
    return VerifiedPortfolioWave(
        root=root.resolve(strict=True),
        index=index,
        index_sha256=index.index_sha256,
        campaigns=(campaigns[0], campaigns[1]),
    )


class SlotOutcome(FrozenModel):
    slot: NonNegativeSafeInt = Field(le=15)
    state: Literal["not_started", "failed", "succeeded"]
    error: StrictStr | None = None
    artifact_checksum_sha256: Sha256Hex | None = None
    proof_report_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _terminal(self) -> SlotOutcome:
        if self.state == "failed" and not self.error:
            raise ValueError("failed slot requires an error")
        if self.state == "succeeded" and (
            self.error is not None
            or self.artifact_checksum_sha256 is None
            or self.proof_report_sha256 is None
        ):
            raise ValueError("succeeded slot requires artifact and proof checksums")
        if self.state == "not_started" and any(
            value is not None
            for value in (
                self.error,
                self.artifact_checksum_sha256,
                self.proof_report_sha256,
            )
        ):
            raise ValueError("not-started slot cannot claim terminal evidence")
        return self


class ExecutionSlot(FrozenModel):
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    run_id: RunId | None
    state: Literal["not_started", "failed", "succeeded"]
    allocated_actual_cost_microusd: NonNegativeSafeInt
    error: StrictStr | None
    artifact_checksum_sha256: Sha256Hex | None
    proof_report_sha256: Sha256Hex | None


class WaveExecutionLedger(FrozenModel):
    schema_version: Literal["distillery.portfolio.execution_ledger.v1"] = (
        "distillery.portfolio.execution_ledger.v1"
    )
    portfolio_wave_index_sha256: Sha256Hex
    parent_actual_cost_microusd: tuple[
        NonNegativeSafeInt,
        NonNegativeSafeInt,
    ]
    aggregate_actual_cost_microusd: NonNegativeSafeInt
    slots: tuple[ExecutionSlot, ...] = Field(min_length=16, max_length=16)
    ledger_sha256: Sha256Hex

    @model_validator(mode="after")
    def _ledger(self) -> WaveExecutionLedger:
        if [slot.slot for slot in self.slots] != list(range(16)):
            raise ValueError("execution ledger cannot repartition logical slots")
        for node in (0, 1):
            if (
                sum(slot.allocated_actual_cost_microusd for slot in self.slots if slot.node == node)
                != self.parent_actual_cost_microusd[node]
            ):
                raise ValueError("execution ledger omits parent cost")
        if sum(self.parent_actual_cost_microusd) != self.aggregate_actual_cost_microusd:
            raise ValueError("execution ledger aggregate cost mismatch")
        if self.ledger_sha256 != _execution_ledger_hash(self):
            raise ValueError("execution ledger hash mismatch")
        return self


def _execution_ledger_hash(value: WaveExecutionLedger) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"ledger_sha256"}))


def build_execution_ledger(
    index: PortfolioWaveIndex,
    *,
    outcomes: Mapping[int, SlotOutcome],
    parent_actual_cost_microusd: tuple[int, int],
) -> WaveExecutionLedger:
    if set(outcomes) != set(range(16)):
        raise ValueError("execution outcomes must include all sixteen logical slots")
    allocations: dict[int, int] = {}
    for node in (0, 1):
        quotient, remainder = divmod(parent_actual_cost_microusd[node], 8)
        for gpu in range(8):
            allocations[node + 2 * gpu] = quotient + (1 if gpu < remainder else 0)
    slots: list[ExecutionSlot] = []
    for binding in index.slots:
        outcome = outcomes[binding.slot]
        if outcome.slot != binding.slot:
            raise ValueError("execution outcome slot key/field mismatch")
        if binding.state == "not_started" and outcome.state != "not_started":
            raise ValueError("reserved not-started slot cannot become a run")
        slots.append(
            ExecutionSlot(
                slot=binding.slot,
                node=binding.node,
                gpu=binding.gpu,
                run_id=binding.run_id,
                state=outcome.state,
                allocated_actual_cost_microusd=allocations[binding.slot],
                error=outcome.error,
                artifact_checksum_sha256=outcome.artifact_checksum_sha256,
                proof_report_sha256=outcome.proof_report_sha256,
            )
        )
    provisional = WaveExecutionLedger.model_construct(
        portfolio_wave_index_sha256=index.index_sha256,
        parent_actual_cost_microusd=parent_actual_cost_microusd,
        aggregate_actual_cost_microusd=sum(parent_actual_cost_microusd),
        slots=tuple(slots),
        ledger_sha256="0" * 64,
    )
    return WaveExecutionLedger.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "ledger_sha256": _execution_ledger_hash(provisional),
        }
    )


class ContainerStagingEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.container_staging.v1"] = (
        "distillery.portfolio.container_staging.v1"
    )
    runtime_image_digest: PrefixedSha256
    entrypoint: tuple[StrictStr, ...]
    module_sha256: FrozenDict[StrictStr, Sha256Hex]
    source_inventory_sha256: Sha256Hex
    source_inventory_size_bytes: PositiveSafeInt
    portfolio_task_filter_integration_sha256: Sha256Hex
    training_protocol_sha256: Sha256Hex
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _staged(self) -> ContainerStagingEvidence:
        if self.entrypoint != (
            CONTAINER_PYTHON,
            "-m",
            CAMPAIGN_ORCHESTRATOR_MODULE,
        ):
            raise ValueError("portfolio image has the wrong campaign entrypoint")
        missing = REQUIRED_IMAGE_MODULES - set(self.module_sha256)
        if missing:
            raise ValueError(f"portfolio image is missing staged modules: {sorted(missing)}")
        if self.evidence_sha256 != _container_staging_hash(self):
            raise ValueError("container staging evidence hash mismatch")
        return self

    def verify_source_inventory_bytes(self, value: bytes) -> None:
        if len(value) != self.source_inventory_size_bytes:
            raise ValueError("container source inventory byte length mismatch")
        if sha256_hex(value) != self.source_inventory_sha256:
            raise ValueError("container source inventory hash mismatch")


def _container_staging_hash(value: ContainerStagingEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def container_staging_evidence(**values: object) -> ContainerStagingEvidence:
    provisional = ContainerStagingEvidence.model_construct(**values, evidence_sha256="0" * 64)
    return ContainerStagingEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _container_staging_hash(provisional),
        }
    )


class PortfolioAwsEvidence(FrozenModel):
    aws_account_id: StrictStr = Field(pattern=r"^[0-9]{12}$")
    iam_role_arn: StrictStr = Field(pattern=r"^arn:aws:iam::[0-9]{12}:role/.+$")
    ecr_image_uri: StrictStr
    models_s3_uri: StrictStr
    dataset_bundle_s3_uri: StrictStr
    container: ContainerStagingEvidence

    @model_validator(mode="after")
    def _uris(self) -> PortfolioAwsEvidence:
        _s3_prefix(self.models_s3_uri, field_name="models URI")
        _s3_prefix(self.dataset_bundle_s3_uri, field_name="dataset bundle URI")
        return self


@dataclass(frozen=True, slots=True)
class PortfolioLaunchPlan:
    dry_run: bool
    wave_id: str
    portfolio_wave_index_sha256: str
    jobs: tuple[dict[str, object], dict[str, object]]
    aggregate_max_parent_cost_microusd: int


def build_portfolio_launch_plan(
    *,
    bundle: VerifiedPortfolioWave,
    plan: PortfolioPlan,
    wave: Wave,
    evidence: PortfolioAwsEvidence,
    volume_size_gb: int = 200,
) -> PortfolioLaunchPlan:
    """Build two requests only. This function has no AWS client or submit path."""
    verify_portfolio_wave(
        bundle.root,
        plan=plan,
        wave=wave,
        expected_index_sha256=bundle.index_sha256,
    )
    if volume_size_gb < 60:
        raise ValueError("portfolio campaign volume must be at least 60 GiB")
    runtime = plan.runtimes[tuple(Tier).index(wave.tier)]
    if (
        evidence.container.runtime_image_digest,
        evidence.container.training_protocol_sha256,
    ) != (runtime.runtime_image_digest, plan.protocol.protocol_sha256):
        raise ValueError("container evidence does not bind runtime image and portfolio protocol")
    image = parse_digest_pinned_ecr_image(evidence.ecr_image_uri)
    if image.digest != runtime.runtime_image_digest:
        raise ValueError("ECR image URI does not match the sealed runtime image")
    if image.account_id != evidence.aws_account_id:
        raise ValueError("ECR image and IAM account differ")
    if evidence.dataset_bundle_s3_uri.rstrip("/") != plan.dataset.uri.rstrip("/"):
        raise ValueError("AWS dataset channel is not the sealed portfolio bundle")
    jobs: list[dict[str, object]] = []
    max_parent_costs: list[int] = []
    for campaign in bundle.campaigns:
        index = campaign.index
        parent_cost = (
            index.pricing.hourly_price_microusd * index.max_runtime_seconds + 3599
        ) // 3600
        if parent_cost > plan.ceilings.per_wave_microusd:
            raise ValueError("portfolio parent job exceeds the sealed wave ceiling")
        if parent_cost > ACCOUNT_CEILING_MICROUSD:
            raise ValueError("portfolio parent job exceeds the $10k account ceiling")
        max_parent_costs.append(parent_cost)
        job_name = f"distillery-pf-{index.hardware.gpu_count}gpu-{campaign.index_sha256[:16]}"
        jobs.append(
            {
                "TrainingJobName": job_name,
                "AlgorithmSpecification": {
                    "TrainingImage": evidence.ecr_image_uri,
                    "TrainingInputMode": "File",
                    "ContainerEntrypoint": list(evidence.container.entrypoint),
                    "ContainerArguments": [
                        "--campaign-root",
                        "/opt/ml/input/data/campaign",
                        "--expected-index-sha256",
                        campaign.index_sha256,
                        "--dataset-dir",
                        "/opt/ml/input/data/dataset",
                        "--models-dir",
                        "/opt/ml/input/data/models",
                        "--output-root",
                        "/opt/ml/output/data",
                        "--model-root",
                        "/opt/ml/model",
                        "--runtime-root",
                        "/tmp/distillery-campaign",
                        "--python-executable",
                        CONTAINER_PYTHON,
                        "--timeout-seconds",
                        str(index.max_runtime_seconds),
                    ],
                },
                "RoleArn": evidence.iam_role_arn,
                "InputDataConfig": [
                    {
                        "ChannelName": "campaign",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": index.input_s3_prefix,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "InputMode": "File",
                    },
                    {
                        "ChannelName": "dataset",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": plan.dataset.uri,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "InputMode": "File",
                    },
                    {
                        "ChannelName": "models",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": evidence.models_s3_uri,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "InputMode": "File",
                    },
                ],
                "OutputDataConfig": {"S3OutputPath": index.campaign_output_prefix},
                "ResourceConfig": {
                    "InstanceType": index.hardware.instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": volume_size_gb,
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": index.max_runtime_seconds,
                },
                "HyperParameters": {
                    "portfolio_wave_index_sha256": bundle.index_sha256,
                    "campaign_index_sha256": campaign.index_sha256,
                    "portfolio_plan_sha256": plan.plan_sha256,
                    "container_staging_evidence_sha256": (evidence.container.evidence_sha256),
                    "portfolio_task_filter_integration_sha256": (
                        evidence.container.portfolio_task_filter_integration_sha256
                    ),
                    "max_parent_cost_microusd": str(parent_cost),
                },
                "EnableNetworkIsolation": True,
                "EnableManagedSpotTraining": False,
            }
        )
    aggregate = sum(max_parent_costs)
    if aggregate != bundle.index.aggregate_ceiling_microusd:
        raise ValueError("launch requests do not match sealed all-slot wave cost")
    if aggregate > plan.ceilings.per_wave_microusd:
        raise ValueError("portfolio launch wave exceeds its sealed ceiling")
    if aggregate > ACCOUNT_CEILING_MICROUSD:
        raise ValueError("portfolio launch wave exceeds the $10k account ceiling")
    return PortfolioLaunchPlan(
        dry_run=True,
        wave_id=wave.wave_id,
        portfolio_wave_index_sha256=bundle.index_sha256,
        jobs=(jobs[0], jobs[1]),
        aggregate_max_parent_cost_microusd=aggregate,
    )
