"""A10G memory gates for QLoRA vs BF16-LoRA emergency student loads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

PrecisionMode = Literal["qlora_nf4", "bf16_lora"]

# Public parameter counts from Qwen2.5 model cards (approx; used only for gating).
QWEN25_05B_PARAMS = 494_000_000
QWEN25_15B_PARAMS = 1_540_000_000
A10G_VRAM_BYTES = 24 * 1024**3
# Leave headroom for CUDA context, activations, optimizer state on LoRA params.
A10G_USABLE_FRACTION = 0.82


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
    bitsandbytes_available: bool,
    bitsandbytes_reliable: bool,
    max_length: int,
    microbatch: int,
    lora_rank: int,
    load_teacher: bool,
) -> MemoryEstimate:
    """Prefer QLoRA; fall back to BF16 LoRA only when A10G estimate fits."""
    if bitsandbytes_available and bitsandbytes_reliable:
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

    bf16 = estimate_emergency_memory(
        mode="bf16_lora",
        max_length=max_length,
        microbatch=microbatch,
        lora_rank=lora_rank,
        load_teacher=load_teacher,
    )
    if not bf16.fits:
        raise RuntimeError(
            "bitsandbytes/QLoRA unavailable or unreliable, and BF16 LoRA memory "
            f"estimate does not fit A10G (total_bytes={bf16.total_bytes}). Failing loud."
        )
    return bf16
