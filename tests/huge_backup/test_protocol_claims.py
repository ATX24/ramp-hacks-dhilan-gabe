"""Anti logit-KD claim tests and protocol hash stability."""

from __future__ import annotations

import pytest

from experiments.huge_backup.profile import HugeBackupTrainingProfile
from experiments.huge_backup.protocol import (
    ProtocolClaimError,
    assert_not_exact_logit_kd,
    assert_protocol_deterministic,
    compute_protocol_hash,
)


def test_rejects_exact_logit_kd_claims() -> None:
    with pytest.raises(ProtocolClaimError, match="exact logit KD"):
        assert_not_exact_logit_kd({"notes": "we run exact logit KD here"})
    with pytest.raises(ProtocolClaimError):
        assert_not_exact_logit_kd("full_vocab_kl online teacher logits")


def test_protocol_hash_stable(mini_profile: HugeBackupTrainingProfile) -> None:
    kwargs = {
        "profile": mini_profile,
        "teacher_responses_sha256": "a" * 64,
        "sampler_order_sha256": "b" * 64,
        "channel_contract": {"mode": "offline_file", "network": "disabled"},
        "flash_attention_attested": True,
    }
    first = compute_protocol_hash(**kwargs)
    second = assert_protocol_deterministic(**kwargs)
    assert first == second
    assert len(first) == 64


def test_objective_declares_not_logit_kd() -> None:
    objective = HugeBackupTrainingProfile().objective_dict()
    assert objective["mode"] == "offline_sequence_distillation"
    assert objective["not_exact_logit_kd"] is True
    assert objective["teacher_runtime"] == "offline_pre_materialized_only"
    assert_not_exact_logit_kd(objective)
