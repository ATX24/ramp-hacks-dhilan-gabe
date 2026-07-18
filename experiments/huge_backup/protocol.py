"""Protocol hashing and anti-logit-KD claim enforcement for huge_backup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.contracts.hashing import content_sha256
from experiments.huge_backup import FORBIDDEN_OBJECTIVE_CLAIMS, HUGE_BACKUP_PROFILE_NAME
from experiments.huge_backup.profile import HugeBackupTrainingProfile

PROTOCOL_SCHEMA_VERSION = "distillery.huge_backup.protocol.v1"


class ProtocolClaimError(ValueError):
    pass


def _normalize_claim_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


_ALLOWED_NEGATION_TEXTS = frozenset(
    {
        "not exact logit kd",
        "not_exact_logit_kd",
    }
)


def assert_not_exact_logit_kd(payload: Mapping[str, Any] | str) -> None:
    """Fail closed if any sealed text claims exact logit KD."""
    if isinstance(payload, str):
        texts = [payload]
    else:
        texts = []
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                for key, value in item.items():
                    # Explicit negation flags are required seals, not claims.
                    if _normalize_claim_text(str(key)) in _ALLOWED_NEGATION_TEXTS:
                        continue
                    stack.append(key)
                    stack.append(value)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, str):
                texts.append(item)
    for text in texts:
        normalized = _normalize_claim_text(text)
        if normalized in _ALLOWED_NEGATION_TEXTS:
            continue
        for forbidden in FORBIDDEN_OBJECTIVE_CLAIMS:
            needle = _normalize_claim_text(forbidden)
            if needle in normalized:
                raise ProtocolClaimError(
                    "huge_backup must not claim exact logit KD; "
                    f"forbidden phrase {forbidden!r} found in {text!r}"
                )


def protocol_payload(
    *,
    profile: HugeBackupTrainingProfile,
    teacher_responses_sha256: str,
    sampler_order_sha256: str,
    channel_contract: Mapping[str, Any],
    flash_attention_attested: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "profile_name": HUGE_BACKUP_PROFILE_NAME,
        "objective": profile.objective_dict(),
        "train_examples": profile.train_examples,
        "max_updates": profile.max_updates,
        "global_batch": profile.global_batch,
        "microbatch": profile.microbatch,
        "world_size": profile.world_size,
        "max_length": profile.max_length,
        "lora_rank": profile.lora_rank,
        "lora_target_modules": list(profile.lora_target_modules),
        "precision_mode": profile.precision_mode,
        "distributed_strategy": profile.distributed_strategy,
        "flash_attention_2": profile.flash_attention_2,
        "flash_attention_attested": flash_attention_attested,
        "gradient_checkpointing": profile.gradient_checkpointing,
        "packed_completion_only": profile.packed_completion_only,
        "student_model_id": profile.student_model_id,
        "teacher_model_id": profile.teacher_model_id,
        "teacher_responses_sha256": teacher_responses_sha256,
        "sampler_order_sha256": sampler_order_sha256,
        "channel_contract": dict(channel_contract),
        "max_runtime_seconds": profile.max_runtime_seconds,
        "artifact_reserve_seconds": profile.artifact_reserve_seconds,
        "instance_type": profile.instance_type,
        "hourly_usd": profile.hourly_usd,
        "price_source": profile.price_source,
    }
    assert_not_exact_logit_kd(payload)
    return payload


def compute_protocol_hash(
    *,
    profile: HugeBackupTrainingProfile,
    teacher_responses_sha256: str,
    sampler_order_sha256: str,
    channel_contract: Mapping[str, Any],
    flash_attention_attested: bool,
) -> str:
    return content_sha256(
        protocol_payload(
            profile=profile,
            teacher_responses_sha256=teacher_responses_sha256,
            sampler_order_sha256=sampler_order_sha256,
            channel_contract=channel_contract,
            flash_attention_attested=flash_attention_attested,
        )
    )


def assert_protocol_deterministic(**kwargs: Any) -> str:
    first = compute_protocol_hash(**kwargs)
    second = compute_protocol_hash(**kwargs)
    if first != second:
        raise RuntimeError("huge_backup protocol hash is non-deterministic")
    return first
