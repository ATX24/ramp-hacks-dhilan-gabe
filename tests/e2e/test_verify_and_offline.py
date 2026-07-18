"""Artifact verification and offline fallback honesty."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_fallback import load_offline_bundle, write_stub_precomputed_layout
from verify_artifacts import (
    load_expected_from_fixture_manifest,
    verify_from_sha256sums,
    verify_tree,
)


def test_fixture_manifest_hashes(golden_jsonl: Path, fixture_manifest: Path) -> None:
    root = golden_jsonl.parent
    expected = load_expected_from_fixture_manifest(fixture_manifest)
    result = verify_tree(root, expected, allow_unexpected=True)
    assert result.ok, result.as_dict()
    assert result.checked >= 2


def test_stub_offline_verifies(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    report = write_stub_precomputed_layout(root)
    verify = verify_from_sha256sums(root)
    assert verify.ok, verify.as_dict()
    bundle = load_offline_bundle(root, report, require_verify=True)
    assert bundle.verified is True
    assert bundle.proof_status == "insufficient_evidence"
    payload = bundle.stage_payload()
    assert payload["mode"] == "precomputed_offline"
    assert "PRECOMPUTED" in payload["ui_label"]
    assert "pending" in payload["claim"].lower() or "projected" in payload["claim"].lower()


def test_offline_refuses_corrupt_checksum(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    report = write_stub_precomputed_layout(root)
    # Corrupt a tracked file after sums were written.
    (root / "report.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity failed"):
        load_offline_bundle(root, report, require_verify=True)
