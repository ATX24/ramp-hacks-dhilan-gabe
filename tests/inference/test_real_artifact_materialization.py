"""Focused checks for the exact real baseline serving registration."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from distillery_inference.bundle import ServingRegistryDocument

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER_PATH = ROOT / "scripts" / "inference" / "materialize_serving_bundle.py"
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_serving_bundle",
    MATERIALIZER_PATH,
)
assert MATERIALIZER_SPEC is not None and MATERIALIZER_SPEC.loader is not None
MATERIALIZER = importlib.util.module_from_spec(MATERIALIZER_SPEC)
MATERIALIZER_SPEC.loader.exec_module(MATERIALIZER)
build_registry = MATERIALIZER.build_registry
safe_extract = MATERIALIZER.safe_extract
safe_relative_path = MATERIALIZER.safe_relative_path
verify_sha256sums = MATERIALIZER.verify_sha256sums

BASELINE_RECORD = (
    ROOT / "infra" / "inference" / "baselines" / "aws-smoke-oracle-sft-79db97a954be.json"
)


def load_record() -> dict[str, object]:
    payload = json.loads(BASELINE_RECORD.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_real_baseline_record_is_exact_and_does_not_claim_improvement() -> None:
    record = load_record()
    assert record["training_job_name"] == "aws-smoke-oracle-sft-79db97a954be"
    assert (
        record["manifest"]["seal_sha256"]
        == "79db97a954be7bd3395cf6856ad2d1b74e3038d022d3aa0e8ff7a5f6b357edaf"
    )
    assert record["base"]["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert (
        record["adapter"]["weights_sha256"]
        == "0021b7d6cdfd86e6a255e9367bc53607055937f2cc249a418f030d3935b13bfc"
    )
    assert record["validation"]["base"]["primary_index"] == 0.0
    assert record["validation"]["adapter"]["primary_index"] == 0.0
    assert record["validation"]["primary_index_delta"] == 0.0
    assert record["validation"]["improvement_claimed"] is False
    assert record["adapter"]["proof_status"] == "insufficient_evidence"
    assert record["adapter"]["promotion_status"] == "not_promoted"


def test_real_registry_exposes_source_and_measured_metadata() -> None:
    record = load_record()
    base_digest = str(record["base"]["weights_sha256"])
    adapter_digest = str(record["adapter"]["weights_sha256"])
    payload = build_registry(
        record=record,
        base_checksums={"model.safetensors": base_digest},
        adapter_checksums={"adapter_model.safetensors": adapter_digest},
    )
    registry = ServingRegistryDocument.model_validate(payload)
    adapter = registry.artifacts[1]
    assert adapter.model_id == "model_oracle_sft"
    assert adapter.supported_tasks == ["variance_analysis", "cash_reconciliation"]
    assert adapter.source_provenance is not None
    assert adapter.source_provenance.training_job_name == record["training_job_name"]
    assert adapter.source_provenance.model_tar_sha256 == record["source"]["model_tar_sha256"]
    assert adapter.stats["validation_primary_index"] == 0.0
    assert adapter.stats["primary_index_delta"] == 0.0


def test_safe_extract_rejects_archive_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("adapter_model.safetensors")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract(archive_path, tmp_path / "output")


@pytest.mark.parametrize("relative", ["../outside", "/absolute", ""])
def test_materializer_rejects_unsafe_relative_paths(relative: str) -> None:
    with pytest.raises(ValueError, match="unsafe relative path"):
        safe_relative_path(relative)


def test_checksum_verification_fails_closed(tmp_path: Path) -> None:
    payload = b"real-adapter"
    artifact = tmp_path / "adapter_model.safetensors"
    artifact.write_bytes(payload)
    wrong = hashlib.sha256(b"different").hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{wrong}  adapter_model.safetensors\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum adapter_model.safetensors mismatch"):
        verify_sha256sums(tmp_path, sums)


def test_safe_extract_accepts_regular_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.tar.gz"
    payload = b"adapter"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("adapter_model.safetensors")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "output"
    safe_extract(archive_path, output)
    assert (output / "adapter_model.safetensors").read_bytes() == payload
