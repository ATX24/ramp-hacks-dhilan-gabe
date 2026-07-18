"""Manifest comparability and separate job/output prefixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.dataset_subset import materialize_emergency_subset
from experiments.aws_smoke.manifests import (
    assert_arms_comparable,
    build_emergency_manifest,
    job_name_for_arm,
    shared_sampler_order_hash,
    write_arm_manifests,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE, REQUIRED_ARMS


def _subset_inputs(tmp_path: Path) -> tuple[list[str], list[str], list[str], dict[str, str], str]:
    subset = materialize_emergency_subset(tmp_path / "subset")
    rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        [str(r["example_id"]) for r in rows],
        [str(r["task"]) for r in rows],
        [str(r["difficulty"]) for r in rows],
        subset.split_sha256,
        subset.content_sha256,
    )


def test_three_arms_share_student_pin_and_order(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    example_ids, tasks, difficulties, split_sha256, content_sha256 = _subset_inputs(tmp_path)
    order = shared_sampler_order_hash(
        example_ids=example_ids,
        tasks=tasks,
        difficulties=difficulties,
        seed=DEFAULT_EMERGENCY_PROFILE.seed,
        tokenizer_sha256=valid_evidence.student_tokenizer_sha256,
    )
    manifests = {
        arm: build_emergency_manifest(
            arm=arm,
            evidence=valid_evidence,
            dataset_id="ds_awssmoke01",
            dataset_uri=valid_evidence.dataset_s3_uri,
            dataset_sha256=content_sha256,
            split_sha256=split_sha256,
            sampler_order_hash=order,
        )
        for arm in REQUIRED_ARMS
    }
    assert_arms_comparable(manifests)
    revisions = {m.models.student.revision for m in manifests.values()}
    assert revisions == {valid_evidence.student_revision}
    orders = {m.sampler_order_hash for m in manifests.values()}
    assert orders == {order}


def test_arms_have_separate_job_names_and_output_prefixes(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    example_ids, tasks, difficulties, split_sha256, content_sha256 = _subset_inputs(tmp_path)
    paths = write_arm_manifests(
        output_dir=tmp_path / "manifests",
        evidence=valid_evidence,
        dataset_id="ds_awssmoke01",
        dataset_uri=valid_evidence.dataset_s3_uri,
        dataset_sha256=content_sha256,
        split_sha256=split_sha256,
        example_ids=example_ids,
        tasks=tasks,
        difficulties=difficulties,
    )
    names: set[str] = set()
    prefixes: set[str] = set()
    for arm, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        seal = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
        name = job_name_for_arm(arm, manifest_sha256=seal)
        names.add(name)
        prefixes.add(payload["output"]["prefix"])
        assert arm.replace("_", "-") in name
    assert len(names) == 3
    assert len(prefixes) == 3


def test_sequence_kd_requires_teacher_responses(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    example_ids, tasks, difficulties, split_sha256, content_sha256 = _subset_inputs(tmp_path)
    order = shared_sampler_order_hash(
        example_ids=example_ids,
        tasks=tasks,
        difficulties=difficulties,
        seed=17,
        tokenizer_sha256=valid_evidence.student_tokenizer_sha256,
    )
    with pytest.raises(ValueError, match="pre-materialized teacher responses"):
        build_emergency_manifest(
            arm="sequence_kd",
            evidence=valid_evidence,
            dataset_id="ds_awssmoke01",
            dataset_uri=valid_evidence.dataset_s3_uri,
            dataset_sha256=content_sha256,
            split_sha256=split_sha256,
            sampler_order_hash=order,
            include_sequence_kd_gate=True,
            teacher_responses_present=False,
        )


def test_subset_has_no_train_val_id_overlap(tmp_path: Path) -> None:
    subset = materialize_emergency_subset(tmp_path / "subset")
    train_ids = {
        json.loads(line)["example_id"]
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    val_ids = {
        json.loads(line)["example_id"]
        for line in subset.validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert not (train_ids & val_ids)
    assert 32 <= len(train_ids) <= 64
    assert len(val_ids) == 16
