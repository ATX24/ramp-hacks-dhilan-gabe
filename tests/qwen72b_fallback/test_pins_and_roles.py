"""Identity, license, and dual-role tests for the 72B fallback workstream."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiments.qwen72b_fallback.pins import (
    CHAT_TEMPLATE_SHA256,
    MODEL_ID,
    REVISION,
    TOKENIZER_SHA256,
    RoleBinding,
    fallback_role_binding,
    load_weight_inventory,
    sealed_identity,
    teacher_role_binding,
)


def test_sealed_identity_matches_inventory() -> None:
    identity = sealed_identity()
    inventory = load_weight_inventory()
    assert identity.model_id == MODEL_ID
    assert identity.revision == REVISION
    assert identity.revision == inventory["revision"]
    assert identity.tokenizer_sha256 == TOKENIZER_SHA256
    assert identity.chat_template_sha256 == CHAT_TEMPLATE_SHA256
    assert identity.inventory_sha256 == inventory["inventory_sha256"]
    assert inventory["n_safetensors_shards"] == 37
    assert inventory["qwen_family_tokenizer_compatible"] is True


def test_license_is_qwen_not_apache() -> None:
    identity = sealed_identity()
    assert "qwen-license" in identity.license_disposition.lower()
    assert "apache-2.0" not in identity.license_disposition.lower()


def test_teacher_and_fallback_roles_share_snapshot_but_differ() -> None:
    teacher = teacher_role_binding()
    fallback = fallback_role_binding()
    assert teacher.identity.revision == fallback.identity.revision
    assert teacher.role == "teacher"
    assert fallback.role == "oracle_sft_adapted_fallback"
    assert teacher.may_be_called_distilled_student is False
    assert fallback.may_be_called_distilled_student is False
    assert "tinyfable" in teacher.supervision_source.lower()
    assert "synthetic" in fallback.supervision_source.lower()


def test_fallback_rejects_larger_teacher_claim() -> None:
    with pytest.raises(ValidationError, match="larger teacher"):
        RoleBinding(
            role="oracle_sft_adapted_fallback",
            identity=sealed_identity(),
            supervision_source="synthetic finance via a larger teacher",
            notes="should fail",
        )
