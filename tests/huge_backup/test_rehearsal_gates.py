"""Mandatory rehearsal gates: timing, memory, save/reload failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.huge_backup.memory import SAFE_PEAK_BYTES
from experiments.huge_backup.profile import HugeBackupTrainingProfile
from experiments.huge_backup.protocol import compute_protocol_hash
from experiments.huge_backup.rehearsal import RehearsalFailed, run_rehearsal
from tests.huge_backup.fakes import TinyLinearModel, TinyOptimizer, tiny_step


def _protocol(mini_profile: HugeBackupTrainingProfile) -> tuple[str, dict]:
    channel_contract = {"mode": "offline_file", "network": "disabled"}
    protocol_hash = compute_protocol_hash(
        profile=mini_profile,
        teacher_responses_sha256="a" * 64,
        sampler_order_sha256="b" * 64,
        channel_contract=channel_contract,
        flash_attention_attested=True,
    )
    return protocol_hash, channel_contract


def test_rehearsal_passes_with_fast_steps(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    protocol_hash, channel_contract = _protocol(mini_profile)
    clock_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3])

    result = run_rehearsal(
        output_root=tmp_path / "out",
        model=TinyLinearModel(),
        optimizer_factory=lambda model: TinyOptimizer(model),
        step_fn=tiny_step,
        protocol_hash=protocol_hash,
        profile=mini_profile,
        peak_memory_bytes=SAFE_PEAK_BYTES // 2,
        channel_contract=channel_contract,
        teacher_responses_sha256="a" * 64,
        sampler_order_sha256="b" * 64,
        flash_attention_attested=True,
        clock=lambda: next(clock_values),
    )
    assert result.passed is True
    assert result.fallback_plan is None
    assert Path(result.adapter_dir, "adapter_config.json").is_file()


def test_median_step_over_8s_emits_7b_fallback(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    protocol_hash, channel_contract = _protocol(mini_profile)
    # Each step reports 9 seconds.
    clock_values = iter([0.0, 9.0, 9.0, 18.0, 18.0, 27.0])

    with pytest.raises(RehearsalFailed, match="exceeds 8.0s") as excinfo:
        run_rehearsal(
            output_root=tmp_path / "out",
            model=TinyLinearModel(),
            optimizer_factory=lambda model: TinyOptimizer(model),
            step_fn=tiny_step,
            protocol_hash=protocol_hash,
            profile=mini_profile,
            peak_memory_bytes=SAFE_PEAK_BYTES // 2,
            channel_contract=channel_contract,
            teacher_responses_sha256="a" * 64,
            sampler_order_sha256="b" * 64,
            flash_attention_attested=True,
            clock=lambda: next(clock_values),
        )
    plan = excinfo.value.fallback_plan
    assert plan["fallback_student_model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert plan["protocol_hash"] != protocol_hash
    assert plan["failed_protocol_hash"] == protocol_hash


def test_peak_memory_fail_closed(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    protocol_hash, channel_contract = _protocol(mini_profile)
    clock_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3])
    with pytest.raises(RehearsalFailed, match="peak memory") as excinfo:
        run_rehearsal(
            output_root=tmp_path / "out",
            model=TinyLinearModel(),
            optimizer_factory=lambda model: TinyOptimizer(model),
            step_fn=tiny_step,
            protocol_hash=protocol_hash,
            profile=mini_profile,
            peak_memory_bytes=SAFE_PEAK_BYTES + 1,
            channel_contract=channel_contract,
            teacher_responses_sha256="a" * 64,
            sampler_order_sha256="b" * 64,
            flash_attention_attested=True,
            clock=lambda: next(clock_values),
        )
    assert excinfo.value.fallback_plan["schema_version"].endswith("fallback_plan.v1")


def test_save_failure_fail_closed(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    protocol_hash, channel_contract = _protocol(mini_profile)
    clock_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3])
    with pytest.raises(RehearsalFailed, match="adapter save failed"):
        run_rehearsal(
            output_root=tmp_path / "out",
            model=TinyLinearModel(),
            optimizer_factory=lambda model: TinyOptimizer(model),
            step_fn=tiny_step,
            protocol_hash=protocol_hash,
            profile=mini_profile,
            peak_memory_bytes=SAFE_PEAK_BYTES // 2,
            channel_contract=channel_contract,
            teacher_responses_sha256="a" * 64,
            sampler_order_sha256="b" * 64,
            flash_attention_attested=True,
            clock=lambda: next(clock_values),
            save_should_fail=True,
        )


def test_reload_failure_fail_closed(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    protocol_hash, channel_contract = _protocol(mini_profile)
    clock_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3])
    with pytest.raises(RehearsalFailed, match="adapter reload failed"):
        run_rehearsal(
            output_root=tmp_path / "out",
            model=TinyLinearModel(),
            optimizer_factory=lambda model: TinyOptimizer(model),
            step_fn=tiny_step,
            protocol_hash=protocol_hash,
            profile=mini_profile,
            peak_memory_bytes=SAFE_PEAK_BYTES // 2,
            channel_contract=channel_contract,
            teacher_responses_sha256="a" * 64,
            sampler_order_sha256="b" * 64,
            flash_attention_attested=True,
            clock=lambda: next(clock_values),
            reload_should_fail=True,
        )
