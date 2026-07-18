"""Memory plan choice and sealed training profile tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiments.qwen72b_fallback.channels import default_sm_channels
from experiments.qwen72b_fallback.cost import (
    FULL_RUN_HARD_CAP_USD,
    REHEARSAL_HARD_CAP_USD,
    exact_gross_cost_usd,
)
from experiments.qwen72b_fallback.memory import choose_precision_plan, estimate_memory
from experiments.qwen72b_fallback.profile import full_profile, rehearsal_profile
from experiments.qwen72b_fallback.protocol import (
    ProtocolClaimError,
    assert_not_distilled_student_claim,
    compute_protocol_hash,
    protocol_payload,
)


def test_qlora_chosen_over_bf16_lora_on_memory() -> None:
    plan = choose_precision_plan()
    bf16 = estimate_memory(precision_mode="bf16_lora")
    qlora = estimate_memory(precision_mode="qlora_4bit")
    assert bf16.fits is False
    assert qlora.fits is True
    assert plan["chosen_precision_mode"] == "qlora_4bit"
    assert plan["chosen_distributed_strategy"] == "ddp"
    assert plan["fsdp2_or_zero3_required"] is False
    assert plan["teacher_or_tool_trajectories_on_warm_timer"] is False


def test_rehearsal_profile_three_steps_under_100() -> None:
    profile = rehearsal_profile()
    assert profile.rehearsal_optimizer_steps == 3
    assert profile.max_updates == 3
    assert profile.precision_mode == "qlora_4bit"
    assert profile.is_distilled_student is False
    assert profile.model_role == "oracle_sft_adapted_fallback"
    assert profile.max_run_usd <= REHEARSAL_HARD_CAP_USD
    gross = exact_gross_cost_usd(
        hourly_usd=profile.hourly_usd,
        max_runtime_seconds=profile.max_runtime_seconds,
    )
    assert gross <= REHEARSAL_HARD_CAP_USD


def test_full_profile_30_to_90_under_500() -> None:
    profile = full_profile()
    assert 30 * 60 <= profile.max_runtime_seconds <= 90 * 60
    assert profile.max_runtime_seconds == 90 * 60
    assert profile.max_run_usd <= FULL_RUN_HARD_CAP_USD
    assert profile.train_examples == profile.max_updates * profile.global_batch


def test_profile_rejects_distilled_student_flag() -> None:
    payload = rehearsal_profile().model_dump(mode="python")
    payload["is_distilled_student"] = True
    with pytest.raises(ValidationError, match="distilled-student"):
        type(rehearsal_profile()).model_validate(payload)


def test_protocol_rejects_distilled_student_claim() -> None:
    with pytest.raises(ProtocolClaimError, match="distilled student"):
        assert_not_distilled_student_claim({"note": "72b is a distilled student"})


def test_protocol_hash_stable() -> None:
    profile = rehearsal_profile()
    channels = default_sm_channels()
    kwargs = {
        "profile": profile,
        "oracle_corpus_sha256": "a" * 64,
        "sampler_order_sha256": "b" * 64,
        "channel_contract": channels.as_contract(),
        "flash_attention_attested": True,
        "trajectories_sha256": "c" * 64,
    }
    first = compute_protocol_hash(**kwargs)
    second = compute_protocol_hash(**kwargs)
    assert first == second
    payload = protocol_payload(**kwargs)
    assert payload["is_distilled_student"] is False
    assert payload["deployable_small_model"] == "TinyFable"
