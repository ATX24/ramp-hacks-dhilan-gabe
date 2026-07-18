"""Memory estimate, FlashAttention attestation, and packing tests."""

from __future__ import annotations

import pytest

from experiments.huge_backup.flash_attn import (
    FlashAttentionError,
    attest_flash_attention_2,
    attn_implementation_for,
)
from experiments.huge_backup.memory import (
    SAFE_PEAK_BYTES,
    assert_ddp_preferred,
    estimate_student_ddp_memory,
    expected_memory_briefing,
    peak_exceeds_safe_threshold,
)
from experiments.huge_backup.packing import PackingError, pack_completion_only


def test_expected_14b_ddp_memory_fits() -> None:
    estimate = estimate_student_ddp_memory()
    assert estimate.fits_ddp is True
    assert estimate.strategy == "ddp"
    assert estimate.student_bytes == 14_700_000_000 * 2
    briefing = expected_memory_briefing()
    assert briefing["fits_ddp"] is True
    assert briefing["fsdp_zero_default"] is False
    assert briefing["expected_peak_gib"] < briefing["safe_peak_gib"]
    assert_ddp_preferred(estimate)


def test_safe_peak_threshold() -> None:
    assert SAFE_PEAK_BYTES == int(80 * 1024**3 * 0.85)
    assert peak_exceeds_safe_threshold(SAFE_PEAK_BYTES) is False
    assert peak_exceeds_safe_threshold(SAFE_PEAK_BYTES + 1) is True


def test_flash_attention_requires_attestation() -> None:
    with pytest.raises(FlashAttentionError, match="probes were not supplied"):
        attest_flash_attention_2(requested=True)
    ok = attest_flash_attention_2(
        requested=True,
        torch_version="2.4.1",
        cuda_available=True,
        flash_attn_importable=True,
    )
    assert ok.attested is True
    assert attn_implementation_for(ok) == "flash_attention_2"
    with pytest.raises(FlashAttentionError, match="2.4"):
        attest_flash_attention_2(
            requested=True,
            torch_version="2.3.0",
            cuda_available=True,
            flash_attn_importable=True,
        )


def test_packed_completion_only_masks_prompt() -> None:
    packed = pack_completion_only([1, 2, 3], [4, 5], max_length=16)
    assert packed.labels == [-100, -100, -100, 4, 5]
    assert packed.completion_mask == [0.0, 0.0, 0.0, 1.0, 1.0]
    with pytest.raises(PackingError):
        pack_completion_only([], [1], max_length=8)
