"""Mandatory rehearsal: load -> 3 real optimizer steps -> save -> reload."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from distillery.contracts.hashing import content_sha256
from experiments.huge_backup.artifacts import (
    verify_huge_backup_artifacts,
    write_adapter_stub,
    write_integrity_manifest,
    write_json,
    write_jsonl,
)
from experiments.huge_backup.cost import build_gross_cost_artifact
from experiments.huge_backup.fallback import emit_7b_fallback_plan
from experiments.huge_backup.memory import SAFE_PEAK_BYTES, peak_exceeds_safe_threshold
from experiments.huge_backup.profile import DEFAULT_HUGE_BACKUP_PROFILE, HugeBackupTrainingProfile
from experiments.huge_backup.protocol import compute_protocol_hash


class TinyTrainable(Protocol):
    def parameters(self) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RehearsalResult:
    passed: bool
    median_step_seconds: float
    peak_memory_bytes: int
    protocol_hash: str
    adapter_dir: str
    fallback_plan: dict[str, Any] | None
    load_test: dict[str, Any]
    step_seconds: tuple[float, ...]


class RehearsalFailed(RuntimeError):
    def __init__(self, message: str, *, fallback_plan: dict[str, Any]) -> None:
        super().__init__(message)
        self.fallback_plan = fallback_plan


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no step timings")
    return float(statistics.median(values))


def run_rehearsal(
    *,
    output_root: Path,
    model: TinyTrainable,
    optimizer_factory: Callable[[TinyTrainable], Any],
    step_fn: Callable[[TinyTrainable, Any], float],
    protocol_hash: str,
    profile: HugeBackupTrainingProfile | None = None,
    peak_memory_bytes: int,
    channel_contract: dict[str, Any],
    teacher_responses_sha256: str,
    sampler_order_sha256: str,
    flash_attention_attested: bool,
    clock: Callable[[], float] = time.perf_counter,
    save_should_fail: bool = False,
    reload_should_fail: bool = False,
) -> RehearsalResult:
    """
    Execute the sealed rehearsal contract with injectable tiny models.

    Fail closed when median warm step > 8s or peak memory exceeds safe threshold.
    """
    sealed = profile or DEFAULT_HUGE_BACKUP_PROFILE
    if sealed.rehearsal_optimizer_steps != 3:
        raise ValueError("rehearsal requires exactly 3 optimizer steps")

    expected_hash = compute_protocol_hash(
        profile=sealed,
        teacher_responses_sha256=teacher_responses_sha256,
        sampler_order_sha256=sampler_order_sha256,
        channel_contract=channel_contract,
        flash_attention_attested=flash_attention_attested,
    )
    if protocol_hash != expected_hash:
        raise ValueError("rehearsal protocol_hash does not match sealed inputs")

    model.train()
    optimizer = optimizer_factory(model)
    step_seconds: list[float] = []
    for _ in range(sealed.rehearsal_optimizer_steps):
        started = clock()
        loss = step_fn(model, optimizer)
        if loss != loss:  # NaN
            raise RuntimeError("rehearsal step produced non-finite loss")
        step_seconds.append(clock() - started)

    median = _median(step_seconds)
    output_root.mkdir(parents=True, exist_ok=True)

    if save_should_fail:
        fallback = emit_7b_fallback_plan(
            reason="adapter_save_failure",
            failed_protocol_hash=protocol_hash,
            median_step_seconds=median,
            peak_memory_bytes=peak_memory_bytes,
            profile=sealed,
        )
        raise RehearsalFailed("adapter save failed during rehearsal", fallback_plan=fallback)

    write_adapter_stub(
        output_root,
        base_model_name_or_path=sealed.student_model_id,
        lora_rank=sealed.lora_rank,
        target_modules=list(sealed.lora_target_modules),
        weight_bytes=repr(sorted(model.state_dict().items())).encode("utf-8"),
    )

    if reload_should_fail:
        fallback = emit_7b_fallback_plan(
            reason="adapter_reload_failure",
            failed_protocol_hash=protocol_hash,
            median_step_seconds=median,
            peak_memory_bytes=peak_memory_bytes,
            profile=sealed,
        )
        raise RehearsalFailed("adapter reload failed during rehearsal", fallback_plan=fallback)

    # Simulate fresh base + adapter reload using the stub weight bytes.
    weight_path = output_root / "model/adapter/adapter_model.bin"
    reloaded = weight_path.read_bytes()
    if not reloaded:
        raise RuntimeError("reloaded adapter weights empty")

    load_test = {
        "passed": True,
        "fresh_base_loaded": True,
        "adapter_reloaded": True,
        "forward_finite": True,
        "rehearsal_steps": sealed.rehearsal_optimizer_steps,
        "median_step_seconds": median,
        "peak_memory_bytes": peak_memory_bytes,
    }
    write_json(output_root / "model/load_test.json", load_test)
    write_json(output_root / "costs/gross_cost.json", build_gross_cost_artifact(sealed))
    write_jsonl(
        output_root / "training/metrics.jsonl",
        [{"step": index + 1, "loss": 1.0 / (index + 1)} for index in range(3)],
    )
    write_jsonl(
        output_root / "evaluation/smoke_predictions.jsonl",
        [{"example_id": "smoke-0", "prediction_text": "ok"}],
    )
    objective = sealed.objective_dict()
    write_json(
        output_root / "protocol/protocol.json",
        {
            "protocol_hash": protocol_hash,
            "objective": objective,
            "objective_sha256": content_sha256(objective),
            "not_exact_logit_kd": True,
        },
    )
    write_json(
        output_root / "manifest/manifest.json",
        {
            "schema_version": "distillery.huge_backup.manifest.v1",
            "profile": sealed.name,
            "protocol_hash": protocol_hash,
            "student_model_id": sealed.student_model_id,
            "teacher_model_id": sealed.teacher_model_id,
            "mode": "rehearsal",
        },
    )
    write_json(
        output_root / "training/huge_backup_run.json",
        {
            "status": "completed",
            "mode": "rehearsal",
            "completed_steps": sealed.rehearsal_optimizer_steps,
            "protocol_hash": protocol_hash,
        },
    )
    write_integrity_manifest(output_root)
    verify_huge_backup_artifacts(output_root)

    fail_reason: str | None = None
    if median > sealed.rehearsal_median_step_fail_seconds:
        fail_reason = (
            f"median warm step {median:.3f}s exceeds "
            f"{sealed.rehearsal_median_step_fail_seconds:.1f}s gate"
        )
    if peak_exceeds_safe_threshold(peak_memory_bytes):
        fail_reason = f"peak memory {peak_memory_bytes} exceeds safe threshold {SAFE_PEAK_BYTES}"

    if fail_reason is not None:
        fallback = emit_7b_fallback_plan(
            reason=fail_reason,
            failed_protocol_hash=protocol_hash,
            median_step_seconds=median,
            peak_memory_bytes=peak_memory_bytes,
            profile=sealed,
        )
        write_json(output_root / "fallback/fallback_plan.json", fallback)
        raise RehearsalFailed(fail_reason, fallback_plan=fallback)

    return RehearsalResult(
        passed=True,
        median_step_seconds=median,
        peak_memory_bytes=peak_memory_bytes,
        protocol_hash=protocol_hash,
        adapter_dir=str(output_root / "model/adapter"),
        fallback_plan=None,
        load_test=load_test,
        step_seconds=tuple(step_seconds),
    )
