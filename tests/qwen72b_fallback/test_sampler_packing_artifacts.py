"""Sampler, packing, synthetic corpus, and sealed artifact tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qwen72b_fallback.artifacts import (
    artifact_bundle_sha256,
    seal_run_artifacts,
)
from experiments.qwen72b_fallback.ddp import (
    FailureBus,
    FakeProcessGroup,
    RankLogWriter,
    RankSafeArtifactPlan,
    assert_rank_safe_reload,
    run_with_failure_propagation,
)
from experiments.qwen72b_fallback.flash_attn import (
    FlashAttentionError,
    attest_flash_attention_2,
    attn_implementation_for,
)
from experiments.qwen72b_fallback.packing import pack_completion_only
from experiments.qwen72b_fallback.sampler import build_sampler_plan
from experiments.qwen72b_fallback.synthetic_finance import (
    corpus_sha256,
    precompute_trajectory_stub,
    rehearsal_corpus,
)


def test_deterministic_sampler_divisible_by_world_size() -> None:
    ids = [f"ex_{i:03d}" for i in range(24)]
    plan_a = build_sampler_plan(ids, world_size=8, seed=17, expected_count=24)
    plan_b = build_sampler_plan(ids, world_size=8, seed=17, expected_count=24)
    assert plan_a.order_sha256 == plan_b.order_sha256
    assert len(plan_a.rank_ids(0)) == 3
    assert plan_a.rank_ids(0) == plan_b.rank_ids(0)


def test_completion_only_masks_prompt() -> None:
    packed = pack_completion_only([1, 2, 3, 4], [5, 6], max_length=8)
    assert packed.labels[:4] == [-100, -100, -100, -100]
    assert packed.labels[4:] == [5, 6]
    assert packed.completion_mask == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_synthetic_rehearsal_corpus() -> None:
    rows = rehearsal_corpus()
    assert len(rows) == 24
    assert all(row.synthetic for row in rows)
    assert all(not row.contains_customer_data for row in rows)
    digest = corpus_sha256(rows)
    assert len(digest) == 64
    traj = precompute_trajectory_stub(rows)
    assert traj["included_in_warm_timer"] is False
    assert traj["corpus_sha256"] == digest


def test_flash_attention_requires_attestation() -> None:
    ok = attest_flash_attention_2(
        requested=True,
        torch_version="2.4.1",
        cuda_available=True,
        flash_attn_importable=True,
    )
    assert attn_implementation_for(ok) == "flash_attention_2"
    with pytest.raises(FlashAttentionError):
        attest_flash_attention_2(
            requested=True,
            torch_version="2.4.1",
            cuda_available=True,
            flash_attn_importable=False,
        )


def test_rank_safe_save_reload_and_artifacts(tmp_path: Path) -> None:
    plan = RankSafeArtifactPlan(
        adapter_dir=tmp_path / "adapter",
        reload_probe_path=tmp_path / "reload.ok",
        manifest_path=tmp_path / "manifest.json",
    )
    assert plan.may_write(0) is True
    assert plan.may_write(1) is False
    assert_rank_safe_reload(
        rank=0,
        plan=plan,
        adapter_files_present=True,
        reload_ok=True,
    )
    with pytest.raises(RuntimeError, match="reload probe failed"):
        assert_rank_safe_reload(
            rank=0,
            plan=plan,
            adapter_files_present=True,
            reload_ok=False,
        )

    checksums = seal_run_artifacts(
        tmp_path / "artifacts",
        adapter_config={"r": 16},
        adapter_blob=b"fake-adapter",
        protocol={"is_distilled_student": False},
        memory_plan={"chosen_precision_mode": "qlora_4bit"},
        gross_cost={"hard_cap_usd": 100},
        manifest={"profile": "rehearsal"},
    )
    assert "model/adapter/adapter_model.safetensors" in checksums
    assert (tmp_path / "artifacts" / "integrity" / "SHA256SUMS").is_file()
    assert len(artifact_bundle_sha256(checksums)) == 64

    group = FakeProcessGroup(rank=0, world_size=8)
    bus = FailureBus(rank=0, world_size=8)
    logger = RankLogWriter(tmp_path / "logs", rank=0)
    run_with_failure_propagation(
        group=group,
        bus=bus,
        logger=logger,
        body=lambda: None,
    )
    assert group.shutdown_calls == 1
    logger.close()
