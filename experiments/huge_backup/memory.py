"""A100-80GB DDP memory gates for 14B BF16 LoRA (no FSDP/ZeRO by default)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Public parameter counts from Qwen2.5 model cards (approx; gating only).
QWEN25_14B_PARAMS = 14_700_000_000
QWEN25_7B_PARAMS = 7_615_616_000
QWEN25_32B_PARAMS = 32_500_000_000
A100_80GB_VRAM_BYTES = 80 * 1024**3
# Leave headroom for CUDA context, activations, LoRA optimizer state.
A100_USABLE_FRACTION = 0.85
SAFE_PEAK_BYTES = int(A100_80GB_VRAM_BYTES * A100_USABLE_FRACTION)


class HugeBackupMemoryProbeEvidence(BaseModel):
    """Measured GPU probe evidence bound to exact model/runtime settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.huge_backup.memory_probe.v1"] = (
        "distillery.huge_backup.memory_probe.v1"
    )
    passed: bool
    precision_mode: Literal["bf16_lora"]
    device_type: Literal["NVIDIA A100-SXM4-80GB"]
    peak_memory_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    capacity_memory_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    headroom_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    safe_peak_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    probe_id: str = Field(min_length=1)
    student_model_id: str = Field(min_length=1)
    student_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_model_id: str = Field(min_length=1)
    teacher_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(ge=1)
    microbatch: int = Field(ge=1)
    world_size: int = Field(ge=8, le=8)
    distributed_strategy: Literal["ddp"] = "ddp"
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    instance_type: Literal["ml.p4de.24xlarge"]
    flash_attention_2_attested: bool

    @model_validator(mode="after")
    def _validate_measurement(self) -> HugeBackupMemoryProbeEvidence:
        if self.headroom_bytes != self.capacity_memory_bytes - self.peak_memory_bytes:
            raise ValueError("headroom_bytes must equal capacity_memory_bytes - peak")
        if self.peak_memory_bytes >= self.capacity_memory_bytes:
            raise ValueError("memory probe must retain positive measured headroom")
        if self.safe_peak_bytes != SAFE_PEAK_BYTES:
            raise ValueError(f"safe_peak_bytes must equal sealed {SAFE_PEAK_BYTES}")
        if self.capacity_memory_bytes != A100_80GB_VRAM_BYTES:
            raise ValueError("capacity must be exactly 80 GiB A100")
        if self.passed and self.peak_memory_bytes > self.safe_peak_bytes:
            raise ValueError("passed=true is incompatible with peak above safe threshold")
        return self


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    student_bytes: int
    activation_bytes: int
    lora_optimizer_bytes: int
    total_bytes: int
    device_bytes: int
    safe_peak_bytes: int
    fits_ddp: bool
    mode: Literal["bf16_lora"]
    strategy: Literal["ddp"]
    expected_peak_gb: float


def _bytes_for_params(n_params: int, bytes_per_param: float) -> int:
    return int(math.ceil(n_params * bytes_per_param))


def estimate_student_ddp_memory(
    *,
    max_length: int = 768,
    microbatch: int = 1,
    lora_rank: int = 16,
    student_params: int = QWEN25_14B_PARAMS,
    device_bytes: int = A100_80GB_VRAM_BYTES,
) -> MemoryEstimate:
    """
    Conservative per-GPU peak for 14B BF16 + LoRA under DDP.

    Teacher weights are offline and must not be loaded during the warm timer.
    """
    student_bytes = _bytes_for_params(student_params, 2.0)
    # Qwen2.5-14B: hidden=5120, layers=48 (card approximations for activation proxy).
    hidden = 5120
    layers = 48
    # Gradient checkpointing reduces activation footprint substantially.
    activation_bytes = microbatch * max_length * hidden * 4 * 2  # checkpointed proxy
    n_modules = 7
    lora_params = 2 * lora_rank * hidden * n_modules * layers
    lora_optimizer_bytes = lora_params * 2 * 2  # AdamW moments fp32
    cuda_context = 2 * 1024**3
    total = student_bytes + activation_bytes + lora_optimizer_bytes + cuda_context
    safe = int(device_bytes * A100_USABLE_FRACTION)
    return MemoryEstimate(
        student_bytes=student_bytes,
        activation_bytes=activation_bytes,
        lora_optimizer_bytes=lora_optimizer_bytes,
        total_bytes=total,
        device_bytes=device_bytes,
        safe_peak_bytes=safe,
        fits_ddp=total <= safe,
        mode="bf16_lora",
        strategy="ddp",
        expected_peak_gb=total / (1024**3),
    )


def assert_ddp_preferred(estimate: MemoryEstimate) -> None:
    if not estimate.fits_ddp:
        raise RuntimeError(
            "DDP memory estimate does not fit A100-80GB usable budget; "
            "only a measured probe may authorize FSDP/ZeRO, and this path "
            f"fails closed without it (total_bytes={estimate.total_bytes}, "
            f"safe_peak_bytes={estimate.safe_peak_bytes})"
        )


def peak_exceeds_safe_threshold(peak_memory_bytes: int) -> bool:
    return peak_memory_bytes > SAFE_PEAK_BYTES


def expected_memory_briefing() -> dict[str, float | int | bool | str]:
    estimate = estimate_student_ddp_memory()
    return {
        "student_model": "Qwen/Qwen2.5-14B-Instruct",
        "precision": "bf16_lora",
        "strategy": "ddp",
        "world_size": 8,
        "device": "NVIDIA A100-SXM4-80GB",
        "capacity_gib": 80,
        "student_weights_gib": estimate.student_bytes / (1024**3),
        "activation_proxy_gib": estimate.activation_bytes / (1024**3),
        "lora_optimizer_gib": estimate.lora_optimizer_bytes / (1024**3),
        "expected_peak_gib": estimate.expected_peak_gb,
        "safe_peak_gib": SAFE_PEAK_BYTES / (1024**3),
        "fits_ddp": estimate.fits_ddp,
        "fsdp_zero_default": False,
        "teacher_on_warm_timer": False,
    }
