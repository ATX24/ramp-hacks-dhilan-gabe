"""Loss/mask wiring for emergency arms (pure Python; reuses Distillery helpers)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from distillery.training.losses import (
    hard_cross_entropy,
    mixed_kd_ce_loss,
    validate_binary_mask,
    validate_target_ids,
)
from experiments.aws_smoke.profile import RunArm, arm_objective


def completion_position_mask(
    *,
    sequence_length: int,
    prompt_length: int,
) -> list[float]:
    """Binary mask: 1 on completion positions, 0 on prompt (and padding handled upstream)."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if prompt_length < 0 or prompt_length >= sequence_length:
        raise ValueError("prompt_length must be in [0, sequence_length)")
    return [0.0] * prompt_length + [1.0] * (sequence_length - prompt_length)


def build_labels_with_ignore(
    token_ids: Sequence[int],
    position_mask: Sequence[float],
    *,
    ignore_index: int = -100,
) -> list[int]:
    """Align labels with the completion mask using ignore_index on masked positions."""
    mask = validate_binary_mask(position_mask, expected_length=len(token_ids))
    labels: list[int] = []
    for token_id, active in zip(token_ids, mask, strict=True):
        labels.append(int(token_id) if active == 1.0 else ignore_index)
    validate_target_ids(
        labels,
        expected_length=len(token_ids),
        vocab_size=max(token_ids) + 1 if token_ids else 1,
        position_mask=mask,
        ignore_index=ignore_index,
        require_mask_label_alignment=True,
    )
    return labels


def compute_arm_loss(
    *,
    arm: RunArm,
    student_logits: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    position_mask: Sequence[float],
    teacher_logits: Sequence[Sequence[float]] | None = None,
    temperature: float = 2.0,
    kd_weight: float | None = None,
    hard_ce_weight: float | None = None,
    vocab_chunk_size: int | None = 4096,
) -> dict[str, float]:
    """
    Compute the emergency-arm objective.

    ``logit_kd`` requires full-vocabulary teacher logits and uses exact forward KL
    via Distillery ``mixed_kd_ce_loss`` (chunked when vocab_chunk_size is set).
    """
    objective = arm_objective(arm)
    mode = str(objective["mode"])
    if mode in {"oracle_sft", "sequence_kd", "ce_ablation"}:
        ce = hard_cross_entropy(
            student_logits,
            target_ids,
            position_mask=position_mask,
        )
        return {
            "loss": ce,
            "ce": ce,
            "kl": 0.0,
            "kd_term": 0.0,
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "temperature": 1.0,
        }
    if mode == "logit_kd":
        if teacher_logits is None:
            raise ValueError("logit_kd requires teacher_logits; refusing to fake KD")
        kd_w = float(kd_weight if kd_weight is not None else objective["kd_weight"])
        ce_w = float(
            hard_ce_weight if hard_ce_weight is not None else objective["hard_ce_weight"]
        )
        return mixed_kd_ce_loss(
            teacher_logits,
            student_logits,
            target_ids,
            temperature=temperature,
            kd_weight=kd_w,
            hard_ce_weight=ce_w,
            position_mask=position_mask,
            vocab_chunk_size=vocab_chunk_size,
            scale_kd_by_temperature_squared=True,
        )
    raise ValueError(f"unsupported arm mode: {mode}")


def assert_special_token_maps_compatible(
    teacher_map: dict[str, int],
    student_map: dict[str, int],
) -> None:
    """Require identical special-token evidence for honest same-tokenizer KD."""
    if teacher_map != student_map:
        raise ValueError(
            "teacher/student special_token_maps differ; refusing logit KD without "
            "same-tokenizer evidence"
        )


def loss_contract_summary(arm: RunArm) -> dict[str, Any]:
    objective = arm_objective(arm)
    return {
        "arm": arm,
        "mode": objective["mode"],
        "signal": objective["signal"],
        "requires_teacher_logits": arm == "logit_kd",
        "requires_teacher_sequences": arm == "sequence_kd",
        "fake_kd_allowed": False,
    }
