"""Container and SageMaker launch contracts (no AWS calls, no weight downloads)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.huge_backup.deadline import warm_time_arithmetic
from experiments.huge_backup.memory import expected_memory_briefing
from experiments.huge_backup.pins import HugeBackupEvidence
from experiments.huge_backup.profile import (
    DEFAULT_HUGE_BACKUP_PROFILE,
    INSTANCE_TYPE,
    HugeBackupTrainingProfile,
    assert_production_seal,
)
from experiments.huge_backup.protocol import assert_not_exact_logit_kd

CONFIRM_PHRASE = "I_CONFIRM_HUGE_BACKUP_WARM_LAUNCH"
ENTRYPOINT = "python -m experiments.huge_backup.train"


class HugeBackupLaunchContract(BaseModel):
    """Sealed launch request body. Building this must never call AWS."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.huge_backup.launch_contract.v1"] = (
        "distillery.huge_backup.launch_contract.v1"
    )
    training_job_name: str = Field(min_length=1, max_length=63)
    instance_type: Literal["ml.p4de.24xlarge"] = INSTANCE_TYPE
    instance_count: Literal[1] = 1
    world_size: Literal[8] = 8
    entrypoint: str = ENTRYPOINT
    mode: Literal["rehearsal", "warm"]
    max_runtime_seconds: int
    artifact_reserve_seconds: int
    hyperparameters: dict[str, str]
    input_channels: dict[str, str]
    output_path: str
    image_uri: str
    role_arn: str
    environment: dict[str, str]
    confirm_phrase_required: str = CONFIRM_PHRASE
    network_isolation: Literal[True] = True
    downloads_forbidden: Literal[True] = True
    distributed_strategy: Literal["ddp"] = "ddp"
    not_exact_logit_kd: Literal[True] = True

    @model_validator(mode="after")
    def _validate(self) -> HugeBackupLaunchContract:
        assert_not_exact_logit_kd(self.model_dump(mode="json"))
        if self.instance_count != 1:
            raise ValueError("huge_backup is single-node only")
        if "TRANSFORMERS_OFFLINE" not in self.environment:
            raise ValueError("TRANSFORMERS_OFFLINE must be sealed in environment")
        if self.environment.get("HF_HUB_OFFLINE") != "1":
            raise ValueError("HF_HUB_OFFLINE=1 required")
        return self


def build_launch_contract(
    *,
    evidence: HugeBackupEvidence,
    training_job_name: str,
    mode: Literal["rehearsal", "warm"],
    profile: HugeBackupTrainingProfile | None = None,
) -> HugeBackupLaunchContract:
    sealed = profile or DEFAULT_HUGE_BACKUP_PROFILE
    if mode == "warm":
        assert_production_seal(sealed)
    if not evidence.flash_attention_2_attested and sealed.flash_attention_2:
        raise ValueError("FlashAttention 2 requested but evidence does not attest it")
    arithmetic = warm_time_arithmetic()
    memory = expected_memory_briefing()
    return HugeBackupLaunchContract(
        training_job_name=training_job_name,
        mode=mode,
        max_runtime_seconds=sealed.max_runtime_seconds,
        artifact_reserve_seconds=sealed.artifact_reserve_seconds,
        hyperparameters={
            "mode": mode,
            "world_size": str(sealed.world_size),
            "profile": sealed.name,
            "protocol_objective": sealed.objective_mode,
        },
        input_channels={
            "manifest": evidence.dataset_s3_uri.rstrip("/") + "/manifest/",
            "dataset": evidence.dataset_s3_uri,
            "models": evidence.models_s3_uri,
            "teacher_responses": evidence.teacher_responses_s3_uri,
        },
        output_path=evidence.artifact_s3_prefix.rstrip("/") + f"/{training_job_name}/",
        image_uri=evidence.ecr_image_uri,
        role_arn=evidence.iam_role_arn,
        environment={
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_DEBUG": "WARN",
            "HUGE_BACKUP_INSTANCE_TYPE": INSTANCE_TYPE,
            "HUGE_BACKUP_WARM_WINDOW_SECONDS": str(arithmetic["training_window_seconds"]),
            "HUGE_BACKUP_EXPECTED_PEAK_GIB": f"{memory['expected_peak_gib']:.2f}",
        },
    )


def container_contract() -> dict[str, Any]:
    """Static container expectations checked by unit tests (not a live build)."""
    return {
        "schema_version": "distillery.huge_backup.container_contract.v1",
        "base_image_line": "pytorch/pytorch CUDA 12.4 torch 2.4.1",
        "python": "3.11",
        "entrypoint_module": "experiments.huge_backup.train",
        "required_packages": [
            "torch==2.4.1",
            "transformers==4.46.3",
            "peft==0.13.2",
            "accelerate==1.1.1",
            "safetensors==0.4.5",
        ],
        "optional_flash_attn": "flash-attn (attestation-gated)",
        "distributed": "torchrun --nproc_per_node=8",
        "network_isolation": True,
        "downloads_at_runtime": False,
        "aws_calls_from_contract_builder": False,
    }


def real_rehearsal_prerequisites() -> list[str]:
    return [
        "Operator-filled HugeBackupEvidence JSON with non-placeholder pins",
        "Local offline snapshots for Qwen2.5-14B-Instruct and tokenizer files",
        "Pre-materialized teacher_responses.json from Qwen2.5-32B-Instruct "
        "(generated before warm timer; provenance-hash sealed; no test labels)",
        "FlashAttention 2 compatibility attestation for torch 2.4.x + CUDA",
        "ml.p4de.24xlarge quota (1×) and gabriel-cli non-root IAM role",
        "Digest-pinned training image matching containers/training ML contract",
        "Mandatory rehearsal pass: load -> 3 optimizer steps -> save -> reload "
        "with median step ≤ 8s and peak memory ≤ 85% of 80 GiB",
        f"Exact confirmation phrase {CONFIRM_PHRASE!r} before any CreateTrainingJob",
    ]
