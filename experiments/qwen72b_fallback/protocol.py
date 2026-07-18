"""Protocol hashing and anti-distilled-student claim enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback import (
    FALLBACK_ROLE_NAME,
    FORBIDDEN_STUDENT_CLAIMS,
    QWEN72B_PROFILE_NAME,
)
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile

PROTOCOL_SCHEMA_VERSION = "distillery.qwen72b_fallback.protocol.v1"


class ProtocolClaimError(ValueError):
    pass


def _normalize_claim_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def assert_not_distilled_student_claim(payload: Mapping[str, Any] | str) -> None:
    if isinstance(payload, str):
        texts = [payload]
    else:
        texts = []
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                stack.extend(item.values())
                stack.extend(item.keys())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, str):
                texts.append(item)
    for text in texts:
        normalized = _normalize_claim_text(text)
        for forbidden in FORBIDDEN_STUDENT_CLAIMS:
            if _normalize_claim_text(forbidden) in normalized:
                raise ProtocolClaimError(
                    "72B adapted fallback must not be called a distilled student "
                    f"without a separately identified larger teacher; forbidden "
                    f"phrase {forbidden!r} found in {text!r}"
                )


def protocol_payload(
    *,
    profile: Qwen72BTrainingProfile,
    oracle_corpus_sha256: str,
    sampler_order_sha256: str,
    channel_contract: Mapping[str, Any],
    flash_attention_attested: bool,
    trajectories_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "profile_name": QWEN72B_PROFILE_NAME,
        "model_role": FALLBACK_ROLE_NAME,
        "is_distilled_student": False,
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
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "oracle_corpus_sha256": oracle_corpus_sha256,
        "trajectories_sha256": trajectories_sha256,
        "sampler_order_sha256": sampler_order_sha256,
        "channel_contract": dict(channel_contract),
        "max_runtime_seconds": profile.max_runtime_seconds,
        "artifact_reserve_seconds": profile.artifact_reserve_seconds,
        "instance_type": profile.instance_type,
        "hourly_usd": profile.hourly_usd,
        "price_source": profile.price_source,
        "hard_cap_usd": profile.hard_cap_usd,
        "deployable_small_model": "TinyFable",
    }
    assert_not_distilled_student_claim(payload)
    return payload


def compute_protocol_hash(**kwargs: Any) -> str:
    return content_sha256(protocol_payload(**kwargs))
