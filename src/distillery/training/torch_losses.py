"""Optional torch kernels for logit.v1 (import only when ML extras are installed).

This module intentionally performs a top-level ``import torch``. It must not be
imported from ``distillery.training`` package ``__init__`` so the base package
remains importable without ML extras. Callers opt in explicitly:

    from distillery.training.torch_losses import forward_kl_chunked_torch
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool):
        raise ValueError("temperature must be finite and > 0")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    return value


def _validate_logits(logits: torch.Tensor, *, name: str) -> None:
    if logits.ndim < 2:
        raise ValueError(f"{name} must have at least position and vocabulary dimensions")
    if logits.numel() == 0 or logits.shape[-1] < 1:
        raise ValueError(f"{name} must be non-empty with vocabulary size >= 1")
    if not logits.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"{name} must contain only finite logits")


def _validated_position_mask(
    position_mask: torch.Tensor | None,
    *,
    expected_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if position_mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    if position_mask.shape != expected_shape:
        raise ValueError("position_mask shape must equal logits position dimensions")
    if position_mask.is_floating_point() and not bool(
        torch.isfinite(position_mask).all().item()
    ):
        raise ValueError("position_mask must be finite")
    if not bool(((position_mask == 0) | (position_mask == 1)).all().item()):
        raise ValueError("position_mask must be binary (0 or 1)")
    return position_mask.to(device=device, dtype=torch.bool)


def forward_kl_full_torch(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean KL(softmax(teacher/T) || softmax(student/T)) over masked positions."""
    checked_temperature = _validate_temperature(temperature)
    _validate_logits(teacher_logits, name="teacher_logits")
    _validate_logits(student_logits, name="student_logits")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits shapes must match")
    if teacher_logits.device != student_logits.device:
        raise ValueError("teacher and student logits must be on the same device")
    mask = _validated_position_mask(
        position_mask,
        expected_shape=teacher_logits.shape[:-1],
        device=teacher_logits.device,
    )
    if not bool(mask.any().item()):
        raise ValueError("position_mask must select at least one output position")
    t = teacher_logits / checked_temperature
    s = student_logits / checked_temperature
    log_p = F.log_softmax(t, dim=-1)
    log_q = F.log_softmax(s, dim=-1)
    p = log_p.exp()
    kl_per_pos = torch.sum(p * (log_p - log_q), dim=-1)
    weight = mask.to(dtype=kl_per_pos.dtype)
    denom = weight.sum()
    return (kl_per_pos * weight).sum() / denom


def forward_kl_chunked_torch(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float,
    vocab_chunk_size: int,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact full-vocab forward KL with vocabulary chunking (shape invariant)."""
    if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, int):
        raise ValueError("vocab_chunk_size must be an integer >= 1")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be >= 1")
    _validate_logits(teacher_logits, name="teacher_logits")
    _validate_logits(student_logits, name="student_logits")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits shapes must match")
    if teacher_logits.device != student_logits.device:
        raise ValueError("teacher and student logits must be on the same device")
    checked_temperature = _validate_temperature(temperature)
    mask = _validated_position_mask(
        position_mask,
        expected_shape=teacher_logits.shape[:-1],
        device=teacher_logits.device,
    )
    if not bool(mask.any().item()):
        raise ValueError("position_mask must select at least one output position")

    t = teacher_logits / checked_temperature
    s = student_logits / checked_temperature
    vocab = t.shape[-1]

    t_chunk_lse: list[torch.Tensor] = []
    s_chunk_lse: list[torch.Tensor] = []
    for start in range(0, vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab)
        t_chunk_lse.append(torch.logsumexp(t[..., start:end], dim=-1))
        s_chunk_lse.append(torch.logsumexp(s[..., start:end], dim=-1))
    t_log_z = torch.logsumexp(torch.stack(t_chunk_lse, dim=-1), dim=-1)
    s_log_z = torch.logsumexp(torch.stack(s_chunk_lse, dim=-1), dim=-1)

    kl_per_pos = torch.zeros(t.shape[:-1], dtype=t.dtype, device=t.device)
    for start in range(0, vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab)
        log_p = t[..., start:end] - t_log_z.unsqueeze(-1)
        log_q = s[..., start:end] - s_log_z.unsqueeze(-1)
        p = log_p.exp()
        kl_per_pos = kl_per_pos + torch.sum(p * (log_p - log_q), dim=-1)

    weight = mask.to(dtype=kl_per_pos.dtype)
    denom = weight.sum()
    return (kl_per_pos * weight).sum() / denom


def hard_cross_entropy_torch(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    position_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Completion-only CE with exact mask/ignore-index alignment."""
    _validate_logits(student_logits, name="student_logits")
    if target_ids.shape != student_logits.shape[:-1]:
        raise ValueError("target_ids shape must equal logits position dimensions")
    if target_ids.device != student_logits.device:
        raise ValueError("target_ids and student_logits must be on the same device")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if target_ids.dtype not in integer_dtypes:
        raise ValueError("target_ids must use an integer dtype")
    mask = _validated_position_mask(
        position_mask,
        expected_shape=target_ids.shape,
        device=student_logits.device,
    )
    ignored = target_ids == ignore_index
    if position_mask is None:
        mask = ~ignored
    elif not bool((ignored == ~mask).all().item()):
        raise ValueError(
            "target/mask alignment requires ignore_index exactly at masked positions"
        )
    if not bool(mask.any().item()):
        raise ValueError("at least one non-ignored output target is required")
    active_targets = target_ids[mask]
    vocab_size = student_logits.shape[-1]
    if not bool(((active_targets >= 0) & (active_targets < vocab_size)).all().item()):
        raise ValueError("active target id is outside the vocabulary")
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(target_ids.shape)
    weight = mask.to(dtype=ce.dtype)
    denom = weight.sum()
    return (ce * weight).sum() / denom
