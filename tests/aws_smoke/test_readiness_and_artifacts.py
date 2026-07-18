"""Trainer source readiness and artifact checksum verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.artifacts import (
    verify_emergency_artifacts,
    write_emergency_integrity,
)
from experiments.aws_smoke.memory import estimate_emergency_memory, select_precision_mode
from experiments.aws_smoke.readiness import assert_trainer_source_ready


def test_trainer_source_has_real_ml_wiring() -> None:
    report = assert_trainer_source_ready()
    assert report["ok"] is True
    assert "torch" in report["modules"]
    assert "peft" in report["modules"]
    assert "transformers" in report["modules"]


def test_bf16_lora_fits_a10g_for_emergency_profile() -> None:
    estimate = estimate_emergency_memory(
        mode="bf16_lora",
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    )
    assert estimate.fits is True
    assert estimate.deviation_label == "DEVIATION:bf16_lora_no_bitsandbytes"

    selected = select_precision_mode(
        bitsandbytes_available=False,
        bitsandbytes_reliable=False,
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    )
    assert selected.mode == "bf16_lora"


def test_qlora_preferred_when_reliable() -> None:
    selected = select_precision_mode(
        bitsandbytes_available=True,
        bitsandbytes_reliable=True,
        max_length=512,
        microbatch=1,
        lora_rank=8,
        load_teacher=True,
    )
    assert selected.mode == "qlora_nf4"
    assert selected.deviation_label is None


def test_artifact_verifier(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "training" / "final").mkdir(parents=True)
    (root / "model" / "adapter").mkdir(parents=True)
    (root / "integrity").mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "training" / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 1.0}) + "\n",
        encoding="utf-8",
    )
    (root / "training" / "final" / "adapter_config.json").write_text(
        '{"r": 8}\n',
        encoding="utf-8",
    )
    (root / "training" / "emergency_run.json").write_text(
        json.dumps({"arm": "oracle_sft", "completed_steps": 8}) + "\n",
        encoding="utf-8",
    )
    write_emergency_integrity(root)
    report = verify_emergency_artifacts(root)
    assert report["ok"] is True
    assert report["count"] >= 4


def test_artifact_verifier_fails_loud_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_emergency_artifacts(tmp_path / "missing")
