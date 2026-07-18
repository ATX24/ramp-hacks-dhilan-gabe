"""7B fallback plan emission under a new protocol hash (fail-closed path)."""

from __future__ import annotations

from typing import Any

from distillery.contracts.hashing import content_sha256
from experiments.huge_backup.profile import (
    DEFAULT_HUGE_BACKUP_PROFILE,
    FALLBACK_STUDENT_MODEL_ID,
    HugeBackupTrainingProfile,
)

FALLBACK_SCHEMA = "distillery.huge_backup.fallback_plan.v1"


def emit_7b_fallback_plan(
    *,
    reason: str,
    failed_protocol_hash: str,
    median_step_seconds: float | None,
    peak_memory_bytes: int | None,
    profile: HugeBackupTrainingProfile | None = None,
) -> dict[str, Any]:
    """
    Build a sealed 7B fallback plan with a distinct protocol hash.

    The warm 14B path must fail closed before this plan is actionable.
    """
    base = profile or DEFAULT_HUGE_BACKUP_PROFILE
    plan = {
        "schema_version": FALLBACK_SCHEMA,
        "status": "fallback_required",
        "reason": reason,
        "failed_protocol_hash": failed_protocol_hash,
        "failed_student_model_id": base.student_model_id,
        "fallback_student_model_id": FALLBACK_STUDENT_MODEL_ID,
        "fallback_precision_mode": "bf16_lora",
        "fallback_distributed_strategy": "ddp",
        "fallback_world_size": base.world_size,
        "fallback_instance_type": base.instance_type,
        "fallback_lora_rank": base.lora_rank,
        "fallback_max_updates": base.max_updates,
        "fallback_train_examples": base.train_examples,
        "fallback_global_batch": base.global_batch,
        "fallback_max_length": base.max_length,
        "median_step_seconds": median_step_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "not_exact_logit_kd": True,
        "objective_mode": "offline_sequence_distillation",
    }
    protocol_hash = content_sha256(
        {
            "schema_version": FALLBACK_SCHEMA,
            "fallback_student_model_id": FALLBACK_STUDENT_MODEL_ID,
            "failed_protocol_hash": failed_protocol_hash,
            "reason": reason,
            "knobs": {
                "lora_rank": base.lora_rank,
                "max_updates": base.max_updates,
                "train_examples": base.train_examples,
                "global_batch": base.global_batch,
                "max_length": base.max_length,
                "world_size": base.world_size,
            },
        }
    )
    if protocol_hash == failed_protocol_hash:
        raise RuntimeError("fallback protocol hash collided with failed protocol hash")
    plan["protocol_hash"] = protocol_hash
    return plan
