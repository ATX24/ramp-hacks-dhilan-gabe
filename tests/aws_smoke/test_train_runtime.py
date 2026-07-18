"""Executable production-loss tests; skipped unless the full ML stack exists."""

from __future__ import annotations

import importlib.util

import pytest

_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "peft", "transformers", "numpy")
)
pytestmark = pytest.mark.skipif(
    not _RUNTIME_AVAILABLE,
    reason="full torch+peft+transformers runtime is not installed",
)

if _RUNTIME_AVAILABLE:
    import torch
    import torch.nn.functional as torch_functional

    from experiments.aws_smoke import train


class TinyChatTokenizer:
    chat_template = "tiny-template"

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert tokenize is True
        prompt = [1, 4, 2]
        if len(conversation) == 1:
            assert add_generation_prompt is True
            return prompt
        assert add_generation_prompt is False
        return prompt + [5, 6, 3]


def test_production_shift_mask_and_ce() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 1, 2]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    shifted_logits, shifted_labels, shifted_mask = train.shift_for_lm(
        logits,
        labels,
        mask,
    )
    assert shifted_logits.shape == (1, 2, 3)
    assert shifted_labels.tolist() == [[1, 2]]
    assert shifted_mask.tolist() == [[1.0, 1.0]]
    loss, parts = train.compute_torch_loss(
        arm="oracle_sft",
        student_logits=logits,
        labels=labels,
        completion_mask=mask,
        teacher_logits=None,
        temperature=2.0,
        kd_weight=0.0,
        hard_ce_weight=1.0,
        vocab_chunk=2,
    )
    assert torch.isfinite(loss)
    assert parts["kl"] == 0.0


def test_production_exact_full_vocab_kl_and_hard_ce_mix() -> None:
    student = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        requires_grad=True,
    )
    teacher = student.detach().clone()
    labels = torch.tensor([[-100, 0, 1]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    loss, parts = train.compute_torch_loss(
        arm="logit_kd",
        student_logits=student,
        labels=labels,
        completion_mask=mask,
        teacher_logits=teacher,
        temperature=2.0,
        kd_weight=0.7,
        hard_ce_weight=0.3,
        vocab_chunk=2,
    )
    assert parts["kl"] == pytest.approx(0.0, abs=1e-7)
    assert parts["loss"] == pytest.approx(0.3 * parts["ce"], rel=1e-6)
    loss.backward()
    assert student.grad is not None


def test_production_tokenize_pair_uses_template_and_completion_cap() -> None:
    batch = train.tokenize_pair(
        TinyChatTokenizer(),
        "prompt",
        "target",
        max_length=8,
        max_completion=2,
    )
    assert batch["input_ids"].tolist() == [[1, 4, 2, 5, 3]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 5, 3]]
    assert batch["completion_mask"].tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0]]
    assert batch["token_evidence"].completion_truncated is True


def test_one_real_optimizer_step_updates_tiny_module() -> None:
    torch.manual_seed(17)
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    inputs = torch_functional.one_hot(
        torch.tensor([[0, 1, 2]]),
        num_classes=4,
    ).float()
    before = model.weight.detach().clone()
    logits = model(inputs)
    labels = torch.tensor([[-100, 1, 2]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    optimizer.zero_grad(set_to_none=True)
    loss, _ = train.compute_torch_loss(
        arm="ce_ablation",
        student_logits=logits,
        labels=labels,
        completion_mask=mask,
        teacher_logits=None,
        temperature=2.0,
        kd_weight=0.0,
        hard_ce_weight=1.0,
        vocab_chunk=2,
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())
