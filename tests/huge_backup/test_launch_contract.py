"""Container/launch contracts must not call AWS or download weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.huge_backup.launch import (
    CONFIRM_PHRASE,
    build_launch_contract,
    container_contract,
    real_rehearsal_prerequisites,
)
from experiments.huge_backup.pins import HugeBackupEvidence
from experiments.huge_backup.profile import DEFAULT_HUGE_BACKUP_PROFILE, assert_production_seal


def test_launch_contract_offline(valid_evidence: HugeBackupEvidence) -> None:
    contract = build_launch_contract(
        evidence=valid_evidence,
        training_job_name="huge-backup-rehearsal-001",
        mode="rehearsal",
    )
    assert contract.instance_type == "ml.p4de.24xlarge"
    assert contract.world_size == 8
    assert contract.network_isolation is True
    assert contract.downloads_forbidden is True
    assert contract.not_exact_logit_kd is True
    assert contract.environment["HF_HUB_OFFLINE"] == "1"
    assert contract.confirm_phrase_required == CONFIRM_PHRASE


def test_warm_launch_requires_production_seal(valid_evidence: HugeBackupEvidence) -> None:
    assert_production_seal(DEFAULT_HUGE_BACKUP_PROFILE)
    contract = build_launch_contract(
        evidence=valid_evidence,
        training_job_name="huge-backup-warm-001",
        mode="warm",
        profile=DEFAULT_HUGE_BACKUP_PROFILE,
    )
    assert contract.max_runtime_seconds == 1800
    assert contract.artifact_reserve_seconds == 300


def test_flash_attn_must_be_attested(valid_evidence: HugeBackupEvidence) -> None:
    evidence = valid_evidence.model_copy(update={"flash_attention_2_attested": False})
    with pytest.raises(ValueError, match="FlashAttention 2"):
        build_launch_contract(
            evidence=evidence,
            training_job_name="huge-backup-rehearsal-002",
            mode="rehearsal",
        )


def test_container_contract_file_matches_module() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "containers"
        / "huge_backup"
        / "container_contract.json"
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    modular = container_contract()
    assert on_disk["entrypoint_module"] == modular["entrypoint_module"]
    assert on_disk["downloads_at_runtime"] is False
    assert on_disk["aws_calls_from_contract_builder"] is False
    assert on_disk["not_exact_logit_kd"] is True


def test_real_rehearsal_prerequisites_are_explicit() -> None:
    prereqs = real_rehearsal_prerequisites()
    assert any("Qwen2.5-14B-Instruct" in item for item in prereqs)
    assert any("Qwen2.5-32B-Instruct" in item for item in prereqs)
    assert any("median step" in item for item in prereqs)
    assert any(CONFIRM_PHRASE in item for item in prereqs)
