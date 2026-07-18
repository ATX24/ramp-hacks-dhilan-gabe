"""Canonical channels, sampler seals, and scientific arm comparability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.channels import discover_and_load_manifest, discover_manifest
from experiments.aws_smoke.manifests import (
    assert_arms_comparable,
    assert_kd_ablation_matched,
    build_sampler_plan,
    manifest_arm,
    manifest_emergency_config,
    manifest_objective,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from tests.aws_smoke.support import build_campaign, build_tokenization_evidence


def test_generated_manifest_channel_round_trip(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    for arm, generated_path in paths.items():
        assert generated_path.name == "manifest.json"
        assert generated_path.parent.name == "manifest"
        parsed = discover_and_load_manifest(generated_path.parent)
        assert manifest_arm(parsed) == arm
        assert parsed.seal_sha256() == (
            generated_path.parent.parent / "manifest.sha256"
        ).read_text(encoding="utf-8").strip()


def test_manifest_channel_rejects_missing_and_ambiguous(tmp_path: Path) -> None:
    channel = tmp_path / "manifest"
    channel.mkdir()
    with pytest.raises(FileNotFoundError, match="exactly one"):
        discover_manifest(channel)
    (channel / "manifest.json").write_text("{}\n", encoding="utf-8")
    (channel / "manifest_backup.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        discover_manifest(channel)


def test_three_required_arms_share_initialization_and_order(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    manifests = {
        arm: discover_and_load_manifest(path.parent) for arm, path in paths.items()
    }
    assert_arms_comparable(manifests)
    fingerprints = {
        manifest.tags["InitializationFingerprint"]
        for manifest in manifests.values()
    }
    order_hashes = {manifest.sampler_order_hash for manifest in manifests.values()}
    assert len(fingerprints) == 1
    assert len(order_hashes) == 1


def test_ce_ablation_discloses_oracle_equivalence_and_matches_kd(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    ce = discover_and_load_manifest(paths["ce_ablation"].parent)
    kd = discover_and_load_manifest(paths["logit_kd"].parent)
    assert_kd_ablation_matched(ce, kd)
    objective = manifest_objective(ce)
    assert objective["distinct_training_signal"] is False
    assert objective["equivalent_to"] == "oracle_sft"
    assert objective["signal"] == "oracle_hard_ce"
    assert manifest_objective(kd)["teacher_runtime"].startswith("online_")


def test_sequence_kd_is_third_distinct_signal_only_with_response_evidence(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(
        tmp_path,
        valid_evidence,
        include_sequence_kd=True,
    )
    sequence = discover_and_load_manifest(paths["sequence_kd"].parent)
    objective = manifest_objective(sequence)
    assert objective["distinct_training_signal"] is True
    assert objective["hard_target_source"] == "pre_materialized_teacher"
    assert manifest_emergency_config(sequence)["teacher_responses_sha256"] == "9" * 64


def test_sampler_plan_uses_sealed_tokenizer_counts(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, rows = build_campaign(tmp_path, valid_evidence)
    manifest = discover_and_load_manifest(paths["oracle_sft"].parent)
    tokenization = build_tokenization_evidence(rows, valid_evidence).arm("oracle_sft")
    counts = {
        str(key): int(value)
        for key, value in manifest.training.completion_evidence.completion_token_counts.items()
    }
    prompt_counts = tokenization.prompt_token_counts
    total_counts = tokenization.total_token_counts
    record_hashes = tokenization.record_sha256
    plan = build_sampler_plan(
        example_ids=[str(row["example_id"]) for row in rows],
        tasks=[str(row["task"]) for row in rows],
        difficulties=[str(row["difficulty"]) for row in rows],
        completion_token_counts=counts,
        prompt_token_counts=prompt_counts,
        total_token_counts=total_counts,
        record_sha256=record_hashes,
        seed=manifest.training.seed,
        tokenizer_sha256=valid_evidence.student_tokenizer_sha256,
        microbatch_size=1,
    )
    assert plan.sampler_order_hash == manifest.sampler_order_hash
    index = json.loads(
        (tmp_path / "manifests/campaign_index.json").read_text(encoding="utf-8")
    )
    assert list(plan.order) == index["shared_sampler_order"]
    bad_counts = dict(counts)
    bad_counts[str(rows[0]["example_id"])] = 0
    with pytest.raises(ValueError):
        build_sampler_plan(
            example_ids=[str(row["example_id"]) for row in rows],
            tasks=[str(row["task"]) for row in rows],
            difficulties=[str(row["difficulty"]) for row in rows],
            completion_token_counts=bad_counts,
            prompt_token_counts=prompt_counts,
            total_token_counts=total_counts,
            record_sha256=record_hashes,
            seed=manifest.training.seed,
            tokenizer_sha256=valid_evidence.student_tokenizer_sha256,
            microbatch_size=1,
        )


def test_campaign_index_marks_only_distinct_signals(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    build_campaign(tmp_path, valid_evidence)
    index = json.loads(
        (tmp_path / "manifests/campaign_index.json").read_text(encoding="utf-8")
    )
    assert index["distinct_signal_count"] == 2
    assert index["arms"]["ce_ablation"]["equivalent_to"] == "oracle_sft"


def test_logit_manifest_requires_real_memory_probe_evidence(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    payload = valid_evidence.model_dump(mode="json")
    payload["memory_probe_evidence"] = None
    without_probe = EmergencyEvidence.model_validate(payload)
    with pytest.raises(ValueError, match="measured A10G memory probe"):
        build_campaign(tmp_path, without_probe)
