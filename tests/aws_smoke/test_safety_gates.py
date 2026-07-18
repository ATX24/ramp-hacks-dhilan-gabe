"""Safety gates: non-root, gabriel-cli, evidence placeholders."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from experiments.aws_smoke.pins import EmergencyEvidence, evidence_schema_template, load_evidence
from experiments.aws_smoke.safety import (
    CallerIdentity,
    enforce_safety_gates,
    evaluate_safety_gates,
    is_root_identity,
)


def test_evidence_template_cannot_pass_gates(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence_schema_template()), encoding="utf-8")
    with pytest.raises((ValidationError, ValueError)):
        load_evidence(path)


def test_root_identity_rejected(valid_evidence: EmergencyEvidence) -> None:
    root = CallerIdentity(
        account=valid_evidence.aws_account_id,
        arn=f"arn:aws:iam::{valid_evidence.aws_account_id}:root",
        user_id="root",
    )
    assert is_root_identity(root)
    result = evaluate_safety_gates(
        profile="gabriel-cli",
        confirm="I_CONFIRM_SAGEMAKER_SUBMIT",
        evidence=valid_evidence,
        identity=root,
        dry_run=False,
    )
    assert result.ok is False
    assert "root_identity_forbidden" in result.violations


def test_wrong_profile_rejected(valid_evidence: EmergencyEvidence) -> None:
    identity = CallerIdentity(
        account=valid_evidence.aws_account_id,
        arn=f"arn:aws:iam::{valid_evidence.aws_account_id}:user/gabriel-cli",
        user_id="AIDATEST",
    )
    result = evaluate_safety_gates(
        profile="default",
        confirm="I_CONFIRM_SAGEMAKER_SUBMIT",
        evidence=valid_evidence,
        identity=identity,
        dry_run=False,
    )
    assert "profile_must_be_gabriel-cli" in result.violations


def test_enforce_passes_for_non_root(valid_evidence: EmergencyEvidence) -> None:
    identity = CallerIdentity(
        account=valid_evidence.aws_account_id,
        arn=f"arn:aws:iam::{valid_evidence.aws_account_id}:user/gabriel-cli",
        user_id="AIDATEST",
    )
    resolved = enforce_safety_gates(
        profile="gabriel-cli",
        confirm="I_CONFIRM_SAGEMAKER_SUBMIT",
        evidence=valid_evidence,
        identity_provider=lambda: identity,
        dry_run=False,
    )
    assert resolved is not None
    assert resolved.arn.endswith("gabriel-cli")


def test_tokenizer_mismatch_rejected_in_evidence(valid_evidence: EmergencyEvidence) -> None:
    payload = valid_evidence.model_dump(mode="json")
    payload["teacher_tokenizer_sha256"] = "9" * 64
    with pytest.raises(ValidationError):
        EmergencyEvidence.model_validate(payload)
