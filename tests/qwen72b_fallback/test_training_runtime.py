from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from experiments.qwen72b_fallback.artifacts import validate_real_adapter
from experiments.qwen72b_fallback.ddp import (
    DistributedContext,
    RankFailure,
    initialize_distributed,
    require_equal_shapes,
    run_synchronized_phase,
)
from experiments.qwen72b_fallback.deadline import RunPhase, time_arithmetic
from experiments.qwen72b_fallback.launch import build_training_request
from experiments.qwen72b_fallback.memory import (
    estimate_for_planning_only,
    planning_comparison,
)
from experiments.qwen72b_fallback.packing import (
    collate_fixed_shape,
    pack_completion_only,
)
from experiments.qwen72b_fallback.profile import (
    Qwen72BTrainingProfile,
    full_profile,
    rehearsal_profile,
)
from experiments.qwen72b_fallback.readiness import ExecutionAction
from experiments.qwen72b_fallback.train import verify_image_runtime

ROOT = Path(__file__).resolve().parents[2]


def test_flash_attention_claim_removed_and_sdpa_math_is_sealed() -> None:
    profile = rehearsal_profile()
    assert profile.flash_attention_2 is False
    assert profile.attention_backend.value == "sdpa_math"
    assert planning_comparison()["authorizes_execution"] is False
    payload = profile.model_dump(mode="json")
    payload["flash_attention_2"] = True
    with pytest.raises(ValidationError):
        Qwen72BTrainingProfile.model_validate(payload)
    dockerfile = (ROOT / "containers" / "training" / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'distillery.qwen72b.flash_attention_2="false"' in dockerfile
    assert "flash-attn" not in lock
    assert "model_parameter_count" in inspect.signature(estimate_for_planning_only).parameters
    assert "student_params" not in inspect.signature(estimate_for_planning_only).parameters


def test_rehearsal_deadline_reserves_every_required_phase() -> None:
    arithmetic = time_arithmetic("rehearsal")
    phases = arithmetic["phase_seconds"]
    assert phases["channel_verify"] > 0
    assert phases["model_load"] > 0
    assert phases["train_steps"] > 0
    assert phases["adapter_save"] > 0
    assert phases["adapter_reload"] > 0
    assert phases["cleanup_artifacts"] > 0
    assert phases["shutdown"] > 0
    offsets = arithmetic["phase_deadline_offsets"]
    assert offsets[RunPhase.TRAIN_STEPS.value] < offsets[RunPhase.ADAPTER_SAVE.value]
    assert offsets[RunPhase.ADAPTER_RELOAD.value] < offsets[RunPhase.SHUTDOWN.value]
    assert arithmetic["max_runtime_seconds"] == 3600


def test_fixed_shape_collation_pads_completion_only_rows() -> None:
    short = pack_completion_only([1, 2], [3], max_length=8)
    long = pack_completion_only([1, 2, 3], [4, 5], max_length=8)
    batch = collate_fixed_shape(
        [short, long],
        max_length=8,
        pad_token_id=0,
    )
    assert {len(row) for row in batch["input_ids"]} == {8}
    assert {len(row) for row in batch["labels"]} == {8}
    assert batch["labels"][0][:2] == [-100, -100]
    assert batch["labels"][0][2] == 3


class ShapeMismatchDist:
    def all_gather_object(self, output: list[Any], value: Any) -> None:
        output[:] = [value, (2, 1024), *([value] * 6)]


class RankDeathDist:
    def all_gather_object(self, _output: list[Any], _value: Any) -> None:
        raise RuntimeError("NCCL timeout after rank 3 death")


def _context(dist: Any) -> DistributedContext:
    return DistributedContext(
        rank=0,
        world_size=8,
        local_rank=0,
        device_index=0,
        timeout_seconds=120,
        torch=object(),
        dist=dist,
    )


def test_shape_mismatch_fails_before_forward() -> None:
    with pytest.raises(RankFailure, match="shape mismatch"):
        require_equal_shapes(_context(ShapeMismatchDist()), (1, 1024))


def test_rank_death_collective_is_bounded_and_fails_loudly() -> None:
    with pytest.raises(RankFailure, match="rank may have died"):
        run_synchronized_phase(
            _context(RankDeathDist()),
            "optimizer_step",
            lambda: True,
        )


def test_each_ddp_child_must_see_one_logical_gpu(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class Cuda:
        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def set_device(index: int) -> None:
            calls["device"] = index

    class Dist:
        @staticmethod
        def init_process_group(**kwargs: Any) -> None:
            calls["init"] = kwargs

    fake_torch = type("Torch", (), {"cuda": Cuda(), "distributed": Dist()})()
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    context = initialize_distributed(torch_module=fake_torch)
    assert context.rank == 3
    assert calls["device"] == 0
    assert calls["init"]["timeout"].total_seconds() == 120
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="exactly one physical GPU"):
        initialize_distributed(torch_module=fake_torch)


def test_train_path_never_uses_device_map_auto() -> None:
    source = (ROOT / "experiments" / "qwen72b_fallback" / "train.py").read_text(encoding="utf-8")
    assert 'device_map="auto"' not in source
    assert 'device_map={"": 0}' in source
    assert "DistributedDataParallel" in source
    assert "save_pretrained" in source
    assert "PeftModel.from_pretrained" in source


def test_wrong_runtime_image_is_rejected_before_training(
    authorization_factory,
    tmp_path: Path,
) -> None:
    profile = rehearsal_profile()
    authorization = authorization_factory(
        action=ExecutionAction.REHEARSAL,
        profile=profile,
        launch_name="qwen72b-rehearsal-image",
    )
    version = tmp_path / "VERSION.json"
    version.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        verify_image_runtime(
            authorization,
            runtime_image_digest="sha256:" + ("f" * 64),
            version_path=version,
        )


def test_memory_probe_authorization_cannot_switch_target_profile(
    authorization_factory,
) -> None:
    target = rehearsal_profile()
    authorization = authorization_factory(
        action=ExecutionAction.MEMORY_PROBE,
        profile=target,
        launch_name="qwen72b-probe-profile-binding",
    )
    with pytest.raises(ValueError, match="profile differs"):
        build_training_request(
            authorization=authorization,
            profile=full_profile(),
            input_prefix="qwen72b/inputs/qwen72b-probe-profile-binding",
            mode="memory_probe",
        )


def test_corrupt_adapter_bytes_cannot_seal_success(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fake adapter bytes")
    with pytest.raises(RuntimeError, match="safetensors validation failed"):
        validate_real_adapter(adapter)
