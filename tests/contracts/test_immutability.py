"""Frozen resource immutability and sealed-manifest hashing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.artifacts import ArtifactChecksums, ModelArtifact
from distillery.contracts.dataset import Dataset, SplitHashes, TaskDifficultyCounts
from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import (
    ManifestCompletionEvidence,
    ManifestCost,
    ManifestDatasetRef,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestQLoRAConfig,
    ManifestRecipe,
    ManifestRuntime,
    ManifestTraining,
    ManifestTrainingCapabilityEvidence,
    SealedRunManifest,
)
from distillery.contracts.recipes import (
    AUTO_SEQUENCE_RESPONSES_REASONS,
    AutoResolverInput,
)
from distillery.contracts.tasks import Difficulty, LabelSource, TaskId

HEX64 = "a" * 64
HEX64B = "b" * 64
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _ts() -> datetime:
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _dataset() -> Dataset:
    return Dataset(
        dataset_id="ds_finance_world_v1",
        content_sha256=HEX64,
        split_sha256=SplitHashes(train=HEX64, validation=HEX64B),
        uri="s3://bucket/datasets/ds_finance_world_v1/",
        provenance_summary="synthetic",
        task_difficulty_counts=TaskDifficultyCounts(
            by_task={
                TaskId.TRANSACTION_REVIEW: 1,
                TaskId.VARIANCE_ANALYSIS: 0,
                TaskId.CASH_RECONCILIATION: 0,
                TaskId.MERCHANT_TAGGING: 0,
            },
            by_difficulty={
                Difficulty.EASY: 1,
                Difficulty.MEDIUM: 0,
                Difficulty.HARD: 0,
            },
        ),
        example_count=1,
        created_at=_ts(),
        metadata={"nested": {"values": [1, 2]}},
    )


def _training(seed: int = 17) -> ManifestTraining:
    return ManifestTraining(
        seed=seed,
        max_steps=30,
        token_budget=0,
        max_length=512,
        qlora=ManifestQLoRAConfig(
            capability_evidence=ManifestTrainingCapabilityEvidence(
                auto_resolver_input=AutoResolverInput(usable_responses_exist=True)
            )
        ),
        completion_evidence=ManifestCompletionEvidence(
            source_file_sha256=HEX64B,
            canonical_records_sha256=HEX64,
            record_sha256={"ex_fixture_001": HEX64},
            provenance_sha256=HEX64B,
            completion_token_counts={"ex_fixture_001": 12},
            completion_tokenizer_sha256=HEX64,
            label_source_counts={LabelSource.ORACLE: 1},
            accepted_example_count=1,
        ),
    )


def _manifest() -> SealedRunManifest:
    return SealedRunManifest(
        run_id="run_seal_001",
        created_at=_ts(),
        dataset=ManifestDatasetRef(
            dataset_id="ds_finance_world_v1",
            uri="s3://bucket/datasets/ds_finance_world_v1/",
            sha256=HEX64,
            split_sha256={"train": HEX64, "validation": HEX64B},
        ),
        models=ManifestModels(
            teacher=ManifestModelSpec(
                id="Qwen/Qwen2.5-1.5B-Instruct",
                revision=REVISION_A,
                tokenizer_sha256=HEX64,
                chat_template_sha256=HEX64B,
            ),
            student=ManifestModelSpec(
                id="Qwen/Qwen2.5-0.5B-Instruct",
                revision=REVISION_B,
                tokenizer_sha256=HEX64,
                chat_template_sha256=HEX64B,
            ),
        ),
        recipe=ManifestRecipe(
            requested="auto",
            resolved="sequence.v1",
            resolver_reasons=AUTO_SEQUENCE_RESPONSES_REASONS,
        ),
        training=_training(),
        proof_protocol=ManifestProofProtocol(id="finance-proof.v1", sha256=HEX64),
        runtime=ManifestRuntime(
            backend="local",
            region="us-east-1",
            instance_type="ml.g5.xlarge",
            image_digest=f"sha256:{HEX64}",
        ),
        cost=ManifestCost(max_run_usd=25.0, estimate_low_usd=1.0, estimate_high_usd=5.0),
        output=ManifestOutput(prefix="s3://bucket/runs/run_seal_001/"),
        package_lock_hash=HEX64,
        source_revision="contracts-v1",
        sampler_order_hash=HEX64B,
    )


def test_dataset_is_frozen() -> None:
    ds = _dataset()
    with pytest.raises(ValidationError):
        ds.example_count = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        ds.metadata["nested"]["values"][0] = 99  # type: ignore[index]


def test_manifest_is_frozen_and_seal_changes_on_mutation() -> None:
    manifest = _manifest()
    h1 = manifest.seal_sha256()
    with pytest.raises(ValidationError):
        manifest.training = _training(23)  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.training.completion_evidence.completion_token_counts[  # type: ignore[index,union-attr]
            "ex_fixture_001"
        ] = 99
    with pytest.raises(TypeError):
        manifest.license_dispositions["teacher"] = "changed"  # type: ignore[index]
    assert manifest.seal_sha256() == h1
    # model_copy produces a new sealed address when content changes
    altered = manifest.model_copy(
        update={
            "training": _training(23),
            "run_id": "run_seal_002",
        }
    )
    h2 = altered.seal_sha256()
    assert h1 != h2
    assert h1 == content_sha256(manifest)


def test_ids_reject_bad_prefixes() -> None:
    with pytest.raises(ValidationError):
        Dataset(
            dataset_id="bad_id",
            content_sha256=HEX64,
            split_sha256=SplitHashes(train=HEX64, validation=HEX64B),
            uri="s3://x",
            provenance_summary="x",
            task_difficulty_counts=TaskDifficultyCounts(
                by_task={
                    TaskId.TRANSACTION_REVIEW: 1,
                    TaskId.VARIANCE_ANALYSIS: 0,
                    TaskId.CASH_RECONCILIATION: 0,
                    TaskId.MERCHANT_TAGGING: 0,
                },
                by_difficulty={
                    Difficulty.EASY: 1,
                    Difficulty.MEDIUM: 0,
                    Difficulty.HARD: 0,
                },
            ),
            example_count=1,
            created_at=_ts(),
        )


@pytest.mark.parametrize(
    "revision",
    [
        "short",
        "main",
        "A" * 40,
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
)
def test_model_revision_requires_lowercase_git_commit_sha(revision: str) -> None:
    with pytest.raises(ValidationError):
        ManifestModelSpec(
            id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=revision,
            tokenizer_sha256=HEX64,
            chat_template_sha256=HEX64B,
        )


def test_model_revision_accepts_lowercase_git_commit_sha() -> None:
    spec = ManifestModelSpec(
        id="Qwen/Qwen2.5-0.5B-Instruct",
        revision=REVISION_A,
        tokenizer_sha256=HEX64,
        chat_template_sha256=HEX64B,
    )
    assert spec.revision == REVISION_A


def test_model_artifact_also_requires_pinned_student_revision() -> None:
    with pytest.raises(ValidationError):
        ModelArtifact(
            artifact_id="art_unpinned_001",
            run_id="run_unpinned_001",
            student_base_id="Qwen/Qwen2.5-0.5B-Instruct",
            student_revision="main",
            adapter_uri="s3://bucket/adapter/",
            tokenizer_uri="s3://bucket/tokenizer/",
            chat_template_uri="s3://bucket/chat-template",
            license_record={"output_use": "allowed"},
            checksums=ArtifactChecksums(
                adapter_sha256=HEX64,
                tokenizer_sha256=HEX64,
                chat_template_sha256=HEX64,
            ),
            load_instructions="Load the adapter.",
            created_at=_ts(),
        )
