"""Measured-memory authorization for 72B QLoRA on 8×A100-80GB."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.evidence import (
    PREFIXED_SHA256_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
)
from experiments.qwen72b_fallback.pins import MODEL_ID, REVISION

QWEN25_72B_PARAMETER_COUNT = 72_706_203_648
A100_80GB_VRAM_BYTES = 80 * 1024**3
SAFE_PEAK_BYTES = int(A100_80GB_VRAM_BYTES * 0.85)
WORLD_SIZE = 8


class PrecisionMode(StrEnum):
    QLORA_NF4_BF16 = "qlora_nf4_bf16"
    BF16_LORA = "bf16_lora"


class DistributedStrategy(StrEnum):
    DDP = "ddp"
    FSDP2 = "fsdp2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"


class AttentionBackend(StrEnum):
    SDPA_MATH = "sdpa_math"


@dataclass(frozen=True, slots=True)
class PlanningEstimate:
    mode: PrecisionMode
    strategy: DistributedStrategy
    estimated_peak_bytes: int
    estimated_peak_gib: float
    authorizes_execution: Literal[False] = False


def estimate_for_planning_only(
    *,
    precision_mode: PrecisionMode,
    model_parameter_count: int = QWEN25_72B_PARAMETER_COUNT,
    max_length: int = 1024,
    microbatch: int = 1,
    lora_rank: int = 16,
) -> PlanningEstimate:
    """Non-authoritative estimate retained only for QLoRA/BF16 comparison."""
    bytes_per_parameter = 0.55 if precision_mode is PrecisionMode.QLORA_NF4_BF16 else 2.0
    base = int(math.ceil(model_parameter_count * bytes_per_parameter))
    activation_proxy = microbatch * max_length * 8192 * 4 * 2
    lora_parameters = 2 * lora_rank * 8192 * 7 * 80
    optimizer_proxy = lora_parameters * 4
    total = base + activation_proxy + optimizer_proxy + (2 * 1024**3)
    return PlanningEstimate(
        mode=precision_mode,
        strategy=DistributedStrategy.DDP,
        estimated_peak_bytes=total,
        estimated_peak_gib=total / (1024**3),
    )


class Qwen72BMemoryProbeEvidence(HashBoundEvidence):
    """Target-device measurement required before DDP rehearsal or training."""

    schema_version: Literal["distillery.qwen72b_fallback.memory_probe.v2"] = (
        "distillery.qwen72b_fallback.memory_probe.v2"
    )
    source: Literal[VerificationSource.TARGET_DEVICE] = VerificationSource.TARGET_DEVICE
    probe_id: str = Field(min_length=1)
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    revision: str = Field(pattern=REVISION_PATTERN)
    model_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_image_uri: str = Field(
        pattern=(
            r"^225989358036\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
            r"distillery-training@sha256:[0-9a-f]{64}$"
        )
    )
    runtime_image_digest: str = Field(pattern=PREFIXED_SHA256_PATTERN)
    image_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    profile_sha256: str = Field(pattern=SHA256_PATTERN)
    instance_type: Literal["ml.p4de.24xlarge"] = "ml.p4de.24xlarge"
    world_size: Literal[8] = WORLD_SIZE
    device_names: tuple[
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
        Literal["NVIDIA A100-SXM4-80GB"],
    ]
    precision_mode: Literal[PrecisionMode.QLORA_NF4_BF16] = PrecisionMode.QLORA_NF4_BF16
    distributed_strategy: Literal[DistributedStrategy.DDP] = DistributedStrategy.DDP
    attention_backend: Literal[AttentionBackend.SDPA_MATH] = AttentionBackend.SDPA_MATH
    max_length: Literal[1024] = 1024
    microbatch: Literal[1] = 1
    lora_rank: Literal[16] = 16
    per_rank_peak_memory_bytes: tuple[int, int, int, int, int, int, int, int]
    per_rank_capacity_bytes: tuple[int, int, int, int, int, int, int, int]
    measured_batch_shapes: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    model_load_completed: Literal[True]
    forward_completed: Literal[True]
    backward_completed: Literal[True]
    optimizer_step_completed: Literal[True]
    all_rank_acknowledgements: tuple[
        Literal[True],
        Literal[True],
        Literal[True],
        Literal[True],
        Literal[True],
        Literal[True],
        Literal[True],
        Literal[True],
    ]
    peak_metric: Literal["torch.cuda.max_memory_reserved"] = "torch.cuda.max_memory_reserved"
    sampler_order_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_artifact_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _measurement_invariants(self) -> Qwen72BMemoryProbeEvidence:
        if self.revision != REVISION:
            raise ValueError("memory probe uses the wrong model revision")
        if not self.runtime_image_uri.endswith(f"@{self.runtime_image_digest}"):
            raise ValueError("memory probe image URI/digest mismatch")
        if any(
            capacity < 75 * 1024**3 or capacity > A100_80GB_VRAM_BYTES
            for capacity in self.per_rank_capacity_bytes
        ):
            raise ValueError("memory probe capacity is outside the A100-80GB range")
        if any(peak <= 0 for peak in self.per_rank_peak_memory_bytes):
            raise ValueError("memory probe peak must be positive on every rank")
        if any(
            peak > int(capacity * 0.85)
            for peak, capacity in zip(
                self.per_rank_peak_memory_bytes,
                self.per_rank_capacity_bytes,
                strict=True,
            )
        ):
            raise ValueError("memory probe exceeds the sealed 85% A100 safety threshold")
        if any(shape != (1, 1024) for shape in self.measured_batch_shapes):
            raise ValueError("memory probe must measure the exact fixed-shape profile batch")
        expected_artifact = memory_probe_measurement_sha256(
            profile_sha256=self.profile_sha256,
            device_names=self.device_names,
            peaks=self.per_rank_peak_memory_bytes,
            capacities=self.per_rank_capacity_bytes,
            shapes=self.measured_batch_shapes,
            sampler_order_sha256=self.sampler_order_sha256,
        )
        if self.probe_artifact_sha256 != expected_artifact:
            raise ValueError("memory probe measurement artifact hash mismatch")
        return self


def memory_probe_measurement_sha256(
    *,
    profile_sha256: str,
    device_names: tuple[str, ...],
    peaks: tuple[int, ...],
    capacities: tuple[int, ...],
    shapes: tuple[tuple[int, int], ...],
    sampler_order_sha256: str,
) -> str:
    return content_sha256(
        {
            "profile_sha256": profile_sha256,
            "peak_metric": "torch.cuda.max_memory_reserved",
            "sampler_order_sha256": sampler_order_sha256,
            "ranks": [
                {
                    "rank": rank,
                    "device_name": device,
                    "peak_memory_bytes": peak,
                    "capacity_bytes": capacity,
                    "batch_shape": list(shape),
                }
                for rank, (device, peak, capacity, shape) in enumerate(
                    zip(
                        device_names,
                        peaks,
                        capacities,
                        shapes,
                        strict=True,
                    )
                )
            ],
        }
    )


def require_measured_probe(
    probe: Qwen72BMemoryProbeEvidence | None,
    *,
    profile_sha256: str,
    model_identity_sha256: str,
    image_binding_sha256: str,
    runtime_image_digest: str,
) -> Qwen72BMemoryProbeEvidence:
    if probe is None:
        raise RuntimeError("DDP authorization requires a measured 72B QLoRA target-device probe")
    if probe.profile_sha256 != profile_sha256:
        raise RuntimeError("memory probe is bound to a different training profile")
    if probe.model_identity_sha256 != model_identity_sha256:
        raise RuntimeError("memory probe is bound to a different model identity")
    if probe.image_binding_sha256 != image_binding_sha256:
        raise RuntimeError("memory probe is bound to a different image binding")
    if probe.runtime_image_digest != runtime_image_digest:
        raise RuntimeError("memory probe is bound to a different image digest")
    return probe


def planning_comparison() -> dict[str, object]:
    qlora = estimate_for_planning_only(precision_mode=PrecisionMode.QLORA_NF4_BF16)
    bf16 = estimate_for_planning_only(precision_mode=PrecisionMode.BF16_LORA)
    return {
        "schema_version": "distillery.qwen72b_fallback.memory_comparison.v2",
        "authorizes_execution": False,
        "chosen_candidate": PrecisionMode.QLORA_NF4_BF16.value,
        "reason": "BF16 LoRA cannot fit DDP; QLoRA still requires a measured probe.",
        "qlora_estimated_peak_gib": qlora.estimated_peak_gib,
        "bf16_estimated_peak_gib": bf16.estimated_peak_gib,
        "mandatory_probe": True,
    }
