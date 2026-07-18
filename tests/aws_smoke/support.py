"""Pure test builders for sealed campaign fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from experiments.aws_smoke.dataset_subset import materialize_emergency_subset
from experiments.aws_smoke.manifests import write_arm_manifests
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import REQUIRED_ARMS
from experiments.aws_smoke.tokenization import (
    ArmTokenizationEvidence,
    TokenizationEvidence,
    canonical_completion_records_sha256,
    completion_record_sha256,
)


def build_tokenization_evidence(
    rows: list[dict],
    evidence: EmergencyEvidence,
    *,
    include_sequence_kd: bool = False,
    source_file_sha256: str | None = None,
) -> TokenizationEvidence:
    # These are explicitly mocked tokenizer results for pure manifest tests.
    oracle_counts = {
        str(row["example_id"]): 8 + (index % 5)
        for index, row in enumerate(rows)
    }
    prompt_counts = {
        str(row["example_id"]): 20 + (index % 7)
        for index, row in enumerate(rows)
    }
    total_counts = {
        example_id: prompt_counts[example_id] + completion_count
        for example_id, completion_count in oracle_counts.items()
    }
    record_hashes = {
        str(row["example_id"]): content_sha256(row) for row in rows
    }
    oracle_completion_hashes = {
        str(row["example_id"]): completion_record_sha256(
            example_id=str(row["example_id"]),
            target_text=json.dumps(
                row["expected_output"],
                sort_keys=True,
                ensure_ascii=False,
            ),
            target_source="oracle",
        )
        for row in rows
    }
    source_hash = source_file_sha256 or content_sha256(rows)
    originals = dict(oracle_counts)
    oracle = ArmTokenizationEvidence(
        arm="oracle_sft",
        target_source="oracle",
        completion_token_counts=oracle_counts,
        prompt_token_counts=prompt_counts,
        total_token_counts=total_counts,
        record_sha256=record_hashes,
        source_file_sha256=source_hash,
        canonical_records_sha256=canonical_completion_records_sha256(
            oracle_completion_hashes
        ),
        completion_record_sha256=oracle_completion_hashes,
        original_completion_token_counts=originals,
    )
    arms = {
        "oracle_sft": oracle,
        "ce_ablation": oracle.model_copy(update={"arm": "ce_ablation"}),
        "logit_kd": oracle.model_copy(update={"arm": "logit_kd"}),
    }
    if include_sequence_kd:
        sequence_counts = {
            example_id: count + 2 for example_id, count in oracle_counts.items()
        }
        arms["sequence_kd"] = ArmTokenizationEvidence(
            arm="sequence_kd",
            target_source="pre_materialized_teacher",
            completion_token_counts=sequence_counts,
            prompt_token_counts=prompt_counts,
            total_token_counts={
                example_id: prompt_counts[example_id] + completion_count
                for example_id, completion_count in sequence_counts.items()
            },
            record_sha256=record_hashes,
            source_file_sha256="9" * 64,
            canonical_records_sha256=canonical_completion_records_sha256({
                example_id: completion_record_sha256(
                    example_id=example_id,
                    target_text="mock-teacher",
                    target_source="pre_materialized_teacher",
                )
                for example_id in sequence_counts
            }),
            completion_record_sha256={
                example_id: completion_record_sha256(
                    example_id=example_id,
                    target_text="mock-teacher",
                    target_source="pre_materialized_teacher",
                )
                for example_id in sequence_counts
            },
            original_completion_token_counts=sequence_counts,
            teacher_responses_sha256="9" * 64,
        )
    return TokenizationEvidence(
        student_tokenizer_sha256=evidence.student_tokenizer_sha256,
        student_chat_template_sha256=evidence.student_chat_template_sha256,
        student_special_token_map=evidence.student_special_token_map,
        max_length=512,
        max_completion=128,
        arms=arms,
    )


def build_campaign(
    tmp_path: Path,
    evidence: EmergencyEvidence,
    *,
    include_sequence_kd: bool = False,
) -> tuple[dict, object, list[dict]]:
    subset = materialize_emergency_subset(tmp_path / "subset")
    evidence_payload = evidence.model_dump(mode="json")
    evidence_payload["data_content_sha256"] = subset.content_sha256
    campaign_evidence = EmergencyEvidence.model_validate(evidence_payload)
    rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenization = build_tokenization_evidence(
        rows,
        campaign_evidence,
        include_sequence_kd=include_sequence_kd,
        source_file_sha256=subset.split_sha256["train"],
    )
    arms = REQUIRED_ARMS + (("sequence_kd",) if include_sequence_kd else ())
    paths = write_arm_manifests(
        output_dir=tmp_path / "manifests",
        evidence=campaign_evidence,
        dataset_id="ds_awssmoke01",
        dataset_uri=evidence.dataset_s3_uri,
        dataset_sha256=subset.content_sha256,
        split_sha256=subset.split_sha256,
        example_ids=[str(row["example_id"]) for row in rows],
        tasks=[str(row["task"]) for row in rows],
        difficulties=[str(row["difficulty"]) for row in rows],
        tokenization_evidence=tokenization,
        arms=arms,
    )
    return paths, subset, rows
