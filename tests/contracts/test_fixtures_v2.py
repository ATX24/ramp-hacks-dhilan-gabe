"""Sealed finance_world_v2 fixture integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import FinanceTaskEnvelope, MerchantTaggingOutput

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "finance_world_v2"


def test_finance_world_v2_fixture_manifest_hashes() -> None:
    manifest = json.loads((FIXTURE_DIR / "fixture_manifest.json").read_text(encoding="utf-8"))
    golden = (FIXTURE_DIR / "golden.jsonl").read_text(encoding="utf-8")
    oracle = (FIXTURE_DIR / "oracle_expected.json").read_text(encoding="utf-8")
    assert sha256_hex(golden.encode()) == manifest["files"]["golden.jsonl"]["sha256"]
    assert sha256_hex(oracle.encode()) == manifest["files"]["oracle_expected.json"]["sha256"]
    assert "merchant_tagging" in manifest["coverage"]["tasks"]
    lines = [line for line in golden.splitlines() if line.strip()]
    assert len(lines) == manifest["record_count"]
    for line in lines:
        payload = json.loads(line)
        example_id = payload["example_id"]
        assert content_sha256(payload) == manifest["semantic_example_sha256"][example_id]
        assert sha256_hex((line + "\n").encode()) == manifest["raw_line_sha256"][example_id]
        if "negative_case" in payload.get("case_tags", []):
            with pytest.raises(ValidationError):
                FinanceTaskEnvelope.model_validate(payload)
            continue
        envelope = FinanceTaskEnvelope.model_validate(payload)
        assert envelope.schema_version == "finance_world.v2"
        if envelope.task.value == "merchant_tagging":
            MerchantTaggingOutput.model_validate(envelope.expected_output)
