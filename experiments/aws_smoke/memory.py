"""A10G/A100 memory gates for QLoRA vs BF16-LoRA emergency student loads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PrecisionMode = Literal["qlora_nf4", "bf16_lora"]

# Public parameter counts from Qwen2.5 model cards (approx; used only for gating).
QWEN25_05B_PARAMS = 494_000_000
QWEN25_15B_PARAMS = 1_540_000_000
A10G_VRAM_BYTES = 24 * 1024**3
A100_80GB_VRAM_BYTES = 80 * 1024**3
MAX_SUPPORTED_VRAM_BYTES = A100_80GB_VRAM_BYTES
# Leave headroom for CUDA context, activations, optimizer state on LoRA params.
A10G_USABLE_FRACTION = 0.82


class EmergencyMemoryProbeEvidence(BaseModel):
    """Measured GPU probe evidence bound to exact model/runtime settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.aws_smoke.memory_probe.v2"] = (
        "distillery.aws_smoke.memory_probe.v2"
    )
    passed: bool
    precision_mode: PrecisionMode
    device_type: Literal["NVIDIA A10G", "NVIDIA A100-SXM4-80GB"]
    peak_memory_bytes: int = Field(ge=1, le=MAX_SUPPORTED_VRAM_BYTES)
    capacity_memory_bytes: int = Field(ge=1, le=MAX_SUPPORTED_VRAM_BYTES)
    headroom_bytes: int = Field(ge=1, le=MAX_SUPPORTED_VRAM_BYTES)
    probe_id: str = Field(min_length=1)
    student_model_id: str = Field(min_length=1)
    student_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_model_id: str = Field(min_length=1)
    teacher_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(ge=1)
    max_completion: int = Field(ge=1)
    vocab_chunk_size: int = Field(ge=1)
    microbatch: int = Field(ge=1)
    grad_accumulation: int = Field(ge=1)
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    instance_type: Literal[
        "ml.g5.xlarge",
        "ml.g5.12xlarge",
        "ml.g5.48xlarge",
        "ml.p4de.24xlarge",
    ]

    @model_validator(mode="after")
    def _validate_measurement(self) -> EmergencyMemoryProbeEvidence:
        if self.headroom_bytes != self.capacity_memory_bytes - self.peak_memory_bytes:
            raise ValueError("headroom_bytes must equal capacity_memory_bytes - peak")
        if self.peak_memory_bytes >= self.capacity_memory_bytes:
            raise ValueError("memory probe must retain positive measured headroom")
        if self.instance_type == "ml.p4de.24xlarge":
            if self.device_type != "NVIDIA A100-SXM4-80GB":
                raise ValueError("p4de memory evidence requires NVIDIA A100-SXM4-80GB")
        elif self.device_type != "NVIDIA A10G":
            raise ValueError("g5 memory evidence requires NVIDIA A10G")
        if (
            self.device_type == "NVIDIA A10G"
            and self.capacity_memory_bytes > A10G_VRAM_BYTES
        ):
            raise ValueError("A10G capacity evidence cannot exceed 24 GiB")
        return self


# Compatibility name for the first emergency-path draft.
Bf16MemoryEvidence = EmergencyMemoryProbeEvidence


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    student_bytes: int
    teacher_bytes: int
    activation_bytes: int
    lora_optimizer_bytes: int
    total_bytes: int
    device_bytes: int
    fits: bool
    mode: PrecisionMode
    deviation_label: str | None


def _bytes_for_params(n_params: int, bytes_per_param: float) -> int:
    return int(math.ceil(n_params * bytes_per_param))


def estimate_emergency_memory(
    *,
    mode: PrecisionMode,
    max_length: int,
    microbatch: int,
    lora_rank: int,
    load_teacher: bool,
    student_params: int = QWEN25_05B_PARAMS,
    teacher_params: int = QWEN25_15B_PARAMS,
    device_bytes: int = A10G_VRAM_BYTES,
) -> MemoryEstimate:
    """
    Conservative peak VRAM estimate for the emergency profile.

    Not a substitute for a real dry-run probe. Used only to decide whether a
    BF16-LoRA deviation is memory-plausible when bitsandbytes/QLoRA is unreliable.
    """
    if mode == "qlora_nf4":
        # NF4 ~0.5 byte/param + small BF16 compute buffers.
        student_bytes = _bytes_for_params(student_params, 0.55)
        deviation = None
    elif mode == "bf16_lora":
        student_bytes = _bytes_for_params(student_params, 2.0)
        deviation = "DEVIATION:bf16_lora_no_bitsandbytes"
    else:
        raise ValueError(f"unknown precision mode: {mode}")

    teacher_bytes = _bytes_for_params(teacher_params, 2.0) if load_teacher else 0
    # Activation proxy: batch * seq * hidden(~896 for 0.5B) * layers(~24) * bytes
    hidden = 896
    layers = 24
    activation_bytes = microbatch * max_length * hidden * layers * 2
    # LoRA AdamW moments: ~2 * 2 bytes * (approx 2 * rank * hidden * n_modules)
    n_modules = 7
    lora_params = 2 * lora_rank * hidden * n_modules * layers
    lora_optimizer_bytes = lora_params * 2 * 2

    total = student_bytes + teacher_bytes + activation_bytes + lora_optimizer_bytes
    budget = int(device_bytes * A10G_USABLE_FRACTION)
    return MemoryEstimate(
        student_bytes=student_bytes,
        teacher_bytes=teacher_bytes,
        activation_bytes=activation_bytes,
        lora_optimizer_bytes=lora_optimizer_bytes,
        total_bytes=total,
        device_bytes=device_bytes,
        fits=total <= budget,
        mode=mode,
        deviation_label=deviation,
    )


def select_precision_mode(
    *,
    sealed_mode: PrecisionMode,
    nf4_kernel_probe_passed: bool,
    bf16_memory_evidence: EmergencyMemoryProbeEvidence | None,
    max_length: int,
    microbatch: int,
    lora_rank: int,
    load_teacher: bool,
) -> MemoryEstimate:
    """Enforce the sealed mode; never silently change protocol after a failed probe."""
    if sealed_mode == "qlora_nf4":
        if not nf4_kernel_probe_passed:
            raise RuntimeError(
                "sealed QLoRA mode requires a successful live bitsandbytes NF4 "
                "kernel probe; silent BF16 fallback is forbidden"
            )
        estimate = estimate_emergency_memory(
            mode="qlora_nf4",
            max_length=max_length,
            microbatch=microbatch,
            lora_rank=lora_rank,
            load_teacher=load_teacher,
        )
        if estimate.fits:
            return estimate
        raise RuntimeError(
            "QLoRA selected but memory estimate does not fit A10G usable budget; "
            f"total_bytes={estimate.total_bytes} "
            f"budget={int(A10G_VRAM_BYTES * A10G_USABLE_FRACTION)}"
        )
    if sealed_mode != "bf16_lora":
        raise ValueError(f"unknown sealed precision mode: {sealed_mode}")
    if bf16_memory_evidence is None:
        raise RuntimeError(
            "sealed BF16 LoRA deviation requires configuration-bound memory evidence"
        )
    if not bf16_memory_evidence.passed:
        raise RuntimeError("sealed BF16 LoRA memory evidence did not pass")
    if bf16_memory_evidence.precision_mode != "bf16_lora":
        raise RuntimeError("BF16 LoRA requires BF16-specific measured memory evidence")
    usable_budget = int(
        bf16_memory_evidence.capacity_memory_bytes * A10G_USABLE_FRACTION
    )
    if bf16_memory_evidence.peak_memory_bytes > usable_budget:
        raise RuntimeError(
            "sealed BF16 LoRA measured peak exceeds the GPU emergency budget: "
            f"peak={bf16_memory_evidence.peak_memory_bytes} budget={usable_budget}"
        )
    bf16 = estimate_emergency_memory(
        mode="bf16_lora",
        max_length=max_length,
        microbatch=microbatch,
        lora_rank=lora_rank,
        load_teacher=load_teacher,
        device_bytes=bf16_memory_evidence.capacity_memory_bytes,
    )
    if not bf16.fits:
        raise RuntimeError(
            "sealed BF16 LoRA static estimate does not fit the sealed GPU "
            f"(total_bytes={bf16.total_bytes}); measured evidence cannot override it"
        )
    return bf16


def validate_runtime_gpu_binding(
    evidence: EmergencyMemoryProbeEvidence,
    *,
    device_type: str,
    capacity_memory_bytes: int,
) -> None:
    if device_type != evidence.device_type:
        raise ValueError(
            "runtime GPU type differs from sealed memory evidence: "
            f"expected={evidence.device_type!r} actual={device_type!r}"
        )
    if capacity_memory_bytes != evidence.capacity_memory_bytes:
        raise ValueError(
            "runtime GPU capacity differs from sealed memory evidence: "
            f"expected={evidence.capacity_memory_bytes} actual={capacity_memory_bytes}"
        )
