"""Trainer source, precision gates, and strict artifact success contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.artifacts import (
    verify_emergency_artifacts,
    write_emergency_integrity,
)
from experiments.aws_smoke.channels import discover_and_load_manifest
from experiments.aws_smoke.memory import (
    Bf16MemoryEvidence,
    estimate_emergency_memory,
    select_precision_mode,
    validate_runtime_gpu_binding,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import EmergencyTrainingProfile
from experiments.aws_smoke.readiness import assert_trainer_source_ready
from tests.aws_smoke.support import build_campaign


def test_trainer_source_has_real_ml_wiring() -> None:
    report = assert_trainer_source_ready()
    assert report["ok"] is True
    assert {"torch", "peft", "transformers"} <= set(report["modules"])


def test_qlora_requires_actual_nf4_probe() -> None:
    with pytest.raises(RuntimeError, match="kernel probe"):
        select_precision_mode(
            sealed_mode="qlora_nf4",
            nf4_kernel_probe_passed=False,
            bf16_memory_evidence=None,
            max_length=512,
            microbatch=1,
            lora_rank=8,
            load_teacher=True,
        )
    selected = select_precision_mode(
        sealed_mode="qlora_nf4",
        nf4_kernel_probe_passed=True,
        bf16_memory_evidence=None,
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    )
    assert selected.mode == "qlora_nf4"


def test_bf16_requires_sealed_measured_memory_evidence() -> None:
    with pytest.raises(RuntimeError, match="memory evidence"):
        select_precision_mode(
            sealed_mode="bf16_lora",
            nf4_kernel_probe_passed=False,
            bf16_memory_evidence=None,
            max_length=512,
            microbatch=1,
            lora_rank=8,
            load_teacher=True,
        )
    capacity = 24 * 1024**3
    peak = 8 * 1024**3
    evidence = Bf16MemoryEvidence(
        passed=True,
        precision_mode="bf16_lora",
        device_type="NVIDIA A10G",
        peak_memory_bytes=peak,
        capacity_memory_bytes=capacity,
        headroom_bytes=capacity - peak,
        probe_id="real-a10g-probe-001",
        student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision="e" * 40,
        teacher_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        teacher_revision="f" * 40,
        max_length=512,
        max_completion=128,
        vocab_chunk_size=4096,
        microbatch=1,
        grad_accumulation=1,
        runtime_image_digest="sha256:" + ("a" * 64),
        instance_type="ml.g5.xlarge",
    )
    selected = select_precision_mode(
        sealed_mode="bf16_lora",
        nf4_kernel_probe_passed=False,
        bf16_memory_evidence=evidence,
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    )
    assert selected.mode == "bf16_lora"
    assert selected.deviation_label == "DEVIATION:bf16_lora_no_bitsandbytes"
    validate_runtime_gpu_binding(
        evidence,
        device_type="NVIDIA A10G",
        capacity_memory_bytes=capacity,
    )
    with pytest.raises(ValueError, match="capacity"):
        validate_runtime_gpu_binding(
            evidence,
            device_type="NVIDIA A10G",
            capacity_memory_bytes=capacity - 1,
        )
    profile = EmergencyTrainingProfile(
        precision_mode="bf16_lora",
        memory_probe_evidence=evidence,
    )
    assert profile.precision_mode == "bf16_lora"
    assert estimate_emergency_memory(
        mode="bf16_lora",
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    ).fits


def _write_valid_artifacts(
    root: Path,
    *,
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path / "campaign", valid_evidence)
    manifest = discover_and_load_manifest(paths["oracle_sft"].parent)
    files = {
        "manifest.json": json.dumps(manifest.model_dump(mode="json")) + "\n",
        "training/metrics.jsonl": json.dumps(
            {"step": 1, "loss": 1.0, "ce": 1.0, "kl": 0.0}
        )
        + "\n",
        "evaluation/predictions.jsonl": json.dumps(
            {"example_id": "ex_test01", "prediction_text": "{}"}
        )
        + "\n",
        "model/adapter/adapter_config.json": json.dumps(
            {
                "peft_type": "LORA",
                "target_modules": ["q_proj"],
                "r": 8,
            }
        )
        + "\n",
        "model/adapter/tokenizer_config.json": '{"chat_template":"pinned"}\n',
        "model/adapter/tokenizer.json": '{"version":"1.0"}\n',
        "model/tokenizer_evidence.json": json.dumps(
            {"compatible": True, "student": {}, "teacher": {}}
        )
        + "\n",
        "model/chat_template.txt": "pinned-template\n",
        "model/load_test.json": json.dumps(
            {
                "passed": True,
                "fresh_base_loaded": True,
                "adapter_reloaded": True,
                "forward_finite": True,
            }
        )
        + "\n",
        "costs/gross_cost.json": json.dumps(
            {"gross_cost_usd": 0.01, "max_run_usd": 0.36}
        )
        + "\n",
        "training/emergency_run.json": json.dumps(
            {"status": "completed", "completed_steps": 8}
        )
        + "\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "model/adapter/adapter_model.safetensors").write_bytes(b"real-weights")


def test_artifact_verifier_requires_weights_reload_and_full_checksums(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_valid_artifacts(root, valid_evidence=valid_evidence, tmp_path=tmp_path)
    write_emergency_integrity(root)
    report = verify_emergency_artifacts(root)
    assert report["ok"] is True
    assert report["preferred_safetensors"] is True
    assert "model/load_test.json" in report["checked"]
    assert "evaluation/predictions.jsonl" in report["checked"]
    assert "costs/gross_cost.json" in report["checked"]

    (root / "unchecksummed.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        verify_emergency_artifacts(root)


def test_config_only_artifacts_fail(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    root = tmp_path / "config-only"
    _write_valid_artifacts(root, valid_evidence=valid_evidence, tmp_path=tmp_path)
    (root / "model/adapter/adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        write_emergency_integrity(root)


def test_failed_reload_evidence_fails_success_contract(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    root = tmp_path / "bad-reload"
    _write_valid_artifacts(root, valid_evidence=valid_evidence, tmp_path=tmp_path)
    (root / "model/load_test.json").write_text(
        json.dumps(
            {
                "passed": False,
                "fresh_base_loaded": True,
                "adapter_reloaded": False,
                "forward_finite": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="load-test"):
        write_emergency_integrity(root)
