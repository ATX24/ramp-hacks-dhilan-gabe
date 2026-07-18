"""Memory plan: 4-bit QLoRA vs BF16 LoRA on 8×A100-80GB (ml.p4de.24xlarge)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Parameter count inferred from safetensors total_size / 2 (bf16).
QWEN25_72B_PARAMS = 72_706_203_648
A100_80GB_VRAM_BYTES = 80 * 1024**3
A100_USABLE_FRACTION = 0.85
SAFE_PEAK_BYTES = int(A100_80GB_VRAM_BYTES * A100_USABLE_FRACTION)
HIDDEN = 8192
LAYERS = 80
N_LORA_MODULES = 7

PrecisionMode = Literal["qlora_4bit", "bf16_lora"]
DistributedStrategy = Literal["ddp", "fsdp2", "deepspeed_zero3"]


class Qwen72BMemoryProbeEvidence(BaseModel):
    """Optional measured probe; required only to authorize FSDP2/ZeRO-3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.qwen72b_fallback.memory_probe.v1"] = (
        "distillery.qwen72b_fallback.memory_probe.v1"
    )
    passed: bool
    precision_mode: PrecisionMode
    distributed_strategy: DistributedStrategy
    device_type: Literal["NVIDIA A100-SXM4-80GB"]
    peak_memory_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    capacity_memory_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    headroom_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    safe_peak_bytes: int = Field(ge=1, le=A100_80GB_VRAM_BYTES)
    probe_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(ge=1)
    microbatch: int = Field(ge=1)
    world_size: int = Field(ge=8, le=8)
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    instance_type: Literal["ml.p4de.24xlarge"]
    flash_attention_2_attested: bool

    @model_validator(mode="after")
    def _validate_measurement(self) -> Qwen72BMemoryProbeEvidence:
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
    base_weight_bytes: int
    activation_bytes: int
    lora_optimizer_bytes: int
    cuda_context_bytes: int
    total_bytes: int
    device_bytes: int
    safe_peak_bytes: int
    fits: bool
    mode: PrecisionMode
    strategy: DistributedStrategy
    expected_peak_gb: float
    requires_fsdp_or_zero: bool


def _bytes_for_params(n_params: int, bytes_per_param: float) -> int:
    return int(math.ceil(n_params * bytes_per_param))


def estimate_memory(
    *,
    precision_mode: PrecisionMode,
    max_length: int = 1024,
    microbatch: int = 1,
    lora_rank: int = 16,
    student_params: int = QWEN25_72B_PARAMS,
    device_bytes: int = A100_80GB_VRAM_BYTES,
    strategy: DistributedStrategy = "ddp",
) -> MemoryEstimate:
    """Conservative per-GPU peak. Teacher/tool trajectories are offline."""
    if precision_mode == "bf16_lora":
        base_weight_bytes = _bytes_for_params(student_params, 2.0)
    else:
        # NF4 QLoRA ~0.5 byte/param plus double-quant overhead proxy.
        base_weight_bytes = _bytes_for_params(student_params, 0.55)
    # Gradient checkpointing activation proxy.
    activation_bytes = microbatch * max_length * HIDDEN * 4 * 2
    lora_params = 2 * lora_rank * HIDDEN * N_LORA_MODULES * LAYERS
    lora_optimizer_bytes = lora_params * 2 * 2  # AdamW moments fp32
    cuda_context = 2 * 1024**3
    if strategy in {"fsdp2", "deepspeed_zero3"}:
        # Shard base weights across 8 ranks; keep a communication buffer proxy.
        base_weight_bytes = int(math.ceil(base_weight_bytes / 8)) + 4 * 1024**3
    total = base_weight_bytes + activation_bytes + lora_optimizer_bytes + cuda_context
    safe = int(device_bytes * A100_USABLE_FRACTION)
    fits = total <= safe
    requires_fsdp = precision_mode == "bf16_lora" and strategy == "ddp" and not fits
    return MemoryEstimate(
        base_weight_bytes=base_weight_bytes,
        activation_bytes=activation_bytes,
        lora_optimizer_bytes=lora_optimizer_bytes,
        cuda_context_bytes=cuda_context,
        total_bytes=total,
        device_bytes=device_bytes,
        safe_peak_bytes=safe,
        fits=fits,
        mode=precision_mode,
        strategy=strategy,
        expected_peak_gb=total / (1024**3),
        requires_fsdp_or_zero=requires_fsdp,
    )


def choose_precision_plan(
    *,
    max_length: int = 1024,
    microbatch: int = 1,
    lora_rank: int = 16,
    measured_probe: Qwen72BMemoryProbeEvidence | None = None,
) -> dict[str, Any]:
    """Choose QLoRA vs BF16 LoRA from memory/throughput, not novelty."""
    bf16_ddp = estimate_memory(
        precision_mode="bf16_lora",
        max_length=max_length,
        microbatch=microbatch,
        lora_rank=lora_rank,
        strategy="ddp",
    )
    qlora_ddp = estimate_memory(
        precision_mode="qlora_4bit",
        max_length=max_length,
        microbatch=microbatch,
        lora_rank=lora_rank,
        strategy="ddp",
    )
    bf16_fsdp = estimate_memory(
        precision_mode="bf16_lora",
        max_length=max_length,
        microbatch=microbatch,
        lora_rank=lora_rank,
        strategy="fsdp2",
    )

    if qlora_ddp.fits:
        chosen_mode: PrecisionMode = "qlora_4bit"
        chosen_strategy: DistributedStrategy = "ddp"
        rationale = (
            "BF16 LoRA under DDP exceeds A100-80GB usable budget for 72B "
            f"(~{bf16_ddp.expected_peak_gb:.1f} GiB peak). 4-bit QLoRA under DDP "
            f"fits (~{qlora_ddp.expected_peak_gb:.1f} GiB) with FlashAttention + "
            "gradient checkpointing, so FSDP2/ZeRO-3 are not required."
        )
    elif measured_probe is not None and measured_probe.passed:
        chosen_mode = measured_probe.precision_mode
        chosen_strategy = measured_probe.distributed_strategy
        rationale = (
            "Plan estimate did not fit; sealed measured probe authorizes "
            f"{chosen_mode}/{chosen_strategy}."
        )
    elif bf16_fsdp.fits:
        raise RuntimeError(
            "BF16 LoRA only fits with FSDP2/ZeRO sharding estimates, but no measured "
            "probe is sealed; refuse to enable FSDP2/DeepSpeed ZeRO-3 by novelty alone"
        )
    else:
        raise RuntimeError("no feasible 72B memory plan on ml.p4de.24xlarge")

    return {
        "schema_version": "distillery.qwen72b_fallback.memory_plan.v1",
        "chosen_precision_mode": chosen_mode,
        "chosen_distributed_strategy": chosen_strategy,
        "fsdp2_or_zero3_required": chosen_strategy != "ddp",
        "rationale": rationale,
        "estimates": {
            "bf16_lora_ddp": _estimate_dict(bf16_ddp),
            "qlora_4bit_ddp": _estimate_dict(qlora_ddp),
            "bf16_lora_fsdp2": _estimate_dict(bf16_fsdp),
        },
        "safe_peak_gib": SAFE_PEAK_BYTES / (1024**3),
        "device": "NVIDIA A100-SXM4-80GB",
        "world_size": 8,
        "teacher_or_tool_trajectories_on_warm_timer": False,
    }


def _estimate_dict(estimate: MemoryEstimate) -> dict[str, float | int | bool | str]:
    return {
        "mode": estimate.mode,
        "strategy": estimate.strategy,
        "base_weight_gib": estimate.base_weight_bytes / (1024**3),
        "activation_proxy_gib": estimate.activation_bytes / (1024**3),
        "lora_optimizer_gib": estimate.lora_optimizer_bytes / (1024**3),
        "expected_peak_gib": estimate.expected_peak_gb,
        "fits": estimate.fits,
        "requires_fsdp_or_zero": estimate.requires_fsdp_or_zero,
    }


def assert_strategy_authorized(
    *,
    strategy: DistributedStrategy,
    measured_probe: Qwen72BMemoryProbeEvidence | None,
) -> None:
    if strategy == "ddp":
        return
    if measured_probe is None or not measured_probe.passed:
        raise RuntimeError(
            f"{strategy} requires a passed measured memory probe; refusing closed"
        )
    if measured_probe.distributed_strategy != strategy:
        raise RuntimeError(
            "measured probe strategy mismatch: "
            f"requested={strategy} probe={measured_probe.distributed_strategy}"
        )
