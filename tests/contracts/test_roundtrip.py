"""JSON round-trip and schema serializability for public resources."""

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
from distillery.contracts.proof import (
    PROOF_GATE_ORDER,
    ArmResult,
    ProofReport,
    ProofStatus,
    QualityGateResult,
)
from distillery.contracts.recipes import (
    AUTO_SEQUENCE_RESPONSES_REASONS,
    AutoResolverInput,
)
from distillery.contracts.run import DistillationRun
from distillery.contracts.states import RunState
from distillery.contracts.tasks import (
    CashReconciliationOutput,
    Difficulty,
    FinanceTaskEnvelope,
    LabelSource,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
)

HEX64 = "a" * 64
HEX64B = "b" * 64
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _ts() -> datetime:
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def test_dataset_roundtrip_and_hash() -> None:
    ds = Dataset(
        dataset_id="ds_finance_world_v1",
        content_sha256=HEX64,
        split_sha256=SplitHashes(train=HEX64, validation=HEX64B, test=HEX64),
        uri="s3://bucket/datasets/ds_finance_world_v1/",
        provenance_summary="synthetic finance_world.v1",
        task_difficulty_counts=TaskDifficultyCounts(
            by_task={
                TaskId.TRANSACTION_REVIEW: 5,
                TaskId.VARIANCE_ANALYSIS: 5,
                TaskId.MERCHANT_TAGGING: 0,
                TaskId.CASH_RECONCILIATION: 2,
            },
            by_difficulty={
                Difficulty.EASY: 3,
                Difficulty.MEDIUM: 5,
                Difficulty.HARD: 4,
            },
        ),
        example_count=12,
        created_at=_ts(),
    )
    data = ds.model_dump(mode="json")
    again = Dataset.model_validate(data)
    assert again == ds
    assert len(ds.resource_hash()) == 64
    schema = Dataset.model_json_schema()
    assert schema["title"] == "Dataset" or "dataset_id" in schema.get("properties", {})


def test_manifest_seal_is_stable() -> None:
    manifest = SealedRunManifest(
        run_id="run_demo_001",
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
        training=ManifestTraining(
            seed=17,
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
        ),
        proof_protocol=ManifestProofProtocol(id="finance-proof.v1", sha256=HEX64),
        runtime=ManifestRuntime(
            backend="local",
            region="us-east-1",
            instance_type="ml.g5.xlarge",
            image_digest=f"sha256:{HEX64}",
        ),
        cost=ManifestCost(max_run_usd=25.0, estimate_low_usd=1.0, estimate_high_usd=5.0),
        output=ManifestOutput(prefix="s3://bucket/runs/run_demo_001/"),
        package_lock_hash=HEX64,
        source_revision="contracts-v1",
        sampler_order_hash=HEX64B,
    )
    h1 = manifest.seal_sha256()
    h2 = SealedRunManifest.model_validate(manifest.model_dump(mode="json")).seal_sha256()
    assert h1 == h2
    assert h1 == content_sha256(manifest)


def test_run_artifact_proof_roundtrip() -> None:
    run = DistillationRun(
        run_id="run_demo_001",
        dataset_id="ds_finance_world_v1",
        state=RunState.QUEUED,
        manifest_sha256=HEX64,
        requested_recipe="auto",
        created_at=_ts(),
        updated_at=_ts(),
    )
    art = ModelArtifact(
        artifact_id="art_tinyfable_001",
        run_id="run_demo_001",
        student_base_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=REVISION_B,
        adapter_uri="s3://bucket/runs/run_demo_001/model/adapter/",
        tokenizer_uri="s3://bucket/runs/run_demo_001/model/tokenizer/",
        chat_template_uri="s3://bucket/runs/run_demo_001/model/chat_template.txt",
        license_record={"spdx": "Apache-2.0", "output_use": "allowed"},
        checksums=ArtifactChecksums(
            adapter_sha256=HEX64,
            tokenizer_sha256=HEX64,
            chat_template_sha256=HEX64,
        ),
        load_instructions="Load adapter with peft on pinned student revision.",
        created_at=_ts(),
    )
    report = ProofReport(
        report_id="prf_demo_001",
        run_ids=("run_demo_001",),
        protocol_id="finance-proof.v1",
        protocol_sha256=HEX64,
        proof_status=ProofStatus.INSUFFICIENT_EVIDENCE,
        first_failed_gate="pilot_teacher",
        unevaluated_gates=PROOF_GATE_ORDER[1:],
        arm_results=(ArmResult(arm_id="rules", primary_index=0.8),),
        quality_gates=tuple(
            QualityGateResult(
                gate_id=gate_id,
                passed=False if index == 0 else None,
                evaluated=index == 0,
                detail="missing pilot evidence" if index == 0 else "not evaluated",
            )
            for index, gate_id in enumerate(PROOF_GATE_ORDER)
        ),
        created_at=_ts(),
    )
    assert DistillationRun.model_validate(run.model_dump(mode="json")) == run
    assert ModelArtifact.model_validate(art.model_dump(mode="json")) == art
    assert ProofReport.model_validate(report.model_dump(mode="json")) == report


def test_task_output_roundtrips_from_golden(
    golden_records: list[dict],
    oracle_expected: dict,
) -> None:
    for record in golden_records:
        env = FinanceTaskEnvelope.model_validate(record)
        again = FinanceTaskEnvelope.model_validate(env.model_dump(mode="json"))
        assert again.example_id == env.example_id
        tags = set(env.case_tags)
        out = env.expected_output
        if env.task is TaskId.TRANSACTION_REVIEW:
            TransactionReviewOutput.model_validate(out)
        elif env.task is TaskId.VARIANCE_ANALYSIS:
            VarianceAnalysisOutput.model_validate(out)
        elif env.task is TaskId.CASH_RECONCILIATION:
            CashReconciliationOutput.model_validate(out)
        if "unbalanced_journal" in tags:
            with pytest.raises(ValidationError):
                TransactionReviewOutput.model_validate(
                    oracle_expected[env.example_id]["sample_invalid_prediction"]
                )
        elif "sign_inversion" in tags:
            with pytest.raises(ValidationError):
                VarianceAnalysisOutput.model_validate(
                    oracle_expected[env.example_id]["sample_invalid_prediction"]
                )
