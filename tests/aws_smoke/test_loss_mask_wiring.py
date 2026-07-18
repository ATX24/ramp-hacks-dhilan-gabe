"""Loss and completion-mask wiring for emergency arms."""

from __future__ import annotations

import math

import pytest

from experiments.aws_smoke.loss_wiring import (
    assert_special_token_maps_compatible,
    build_labels_with_ignore,
    completion_position_mask,
    compute_arm_loss,
    loss_contract_summary,
)


def test_completion_mask_and_labels_align() -> None:
    mask = completion_position_mask(sequence_length=6, prompt_length=3)
    assert mask == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    labels = build_labels_with_ignore([1, 2, 3, 4, 5, 6], mask)
    assert labels == [-100, -100, -100, 4, 5, 6]


def test_logit_kd_requires_teacher_logits() -> None:
    student = [[0.1, 0.2, 0.3], [0.0, 1.0, 0.0]]
    targets = [1, 1]
    mask = [1.0, 1.0]
    with pytest.raises(ValueError, match="refusing to fake KD"):
        compute_arm_loss(
            arm="logit_kd",
            student_logits=student,
            target_ids=targets,
            position_mask=mask,
            teacher_logits=None,
        )


def test_logit_kd_uses_mixed_forward_kl() -> None:
    teacher = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    student = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    targets = [0, 1]
    mask = [1.0, 1.0]
    result = compute_arm_loss(
        arm="logit_kd",
        student_logits=student,
        target_ids=targets,
        position_mask=mask,
        teacher_logits=teacher,
        temperature=2.0,
        kd_weight=0.7,
        hard_ce_weight=0.3,
        vocab_chunk_size=2,
    )
    assert result["kl"] == pytest.approx(0.0, abs=1e-6)
    assert result["loss"] > 0.0
    assert math.isfinite(result["loss"])


def test_ce_ablation_is_hard_ce_only() -> None:
    student = [[0.0, 5.0, 0.0]]
    result = compute_arm_loss(
        arm="ce_ablation",
        student_logits=student,
        target_ids=[1],
        position_mask=[1.0],
    )
    assert result["kd_weight"] == 0.0
    assert result["hard_ce_weight"] == 1.0
    assert result["kl"] == 0.0


def test_special_token_maps_must_match() -> None:
    assert_special_token_maps_compatible({"eos": 1}, {"eos": 1})
    with pytest.raises(ValueError, match="special_token_maps"):
        assert_special_token_maps_compatible({"eos": 1}, {"eos": 2})


def test_loss_contract_summary_flags() -> None:
    assert loss_contract_summary("logit_kd")["requires_teacher_logits"] is True
    assert loss_contract_summary("logit_kd")["fake_kd_allowed"] is False
    assert loss_contract_summary("sequence_kd")["requires_teacher_sequences"] is True
