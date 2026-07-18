"""Shared fake-only teacher fixtures. No model or AWS calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime

import pytest

from distillery.contracts.tasks import Difficulty, SplitName, TaskId
from distillery.teachers.policy import PROJECT_PROVIDER_POLICY
from distillery.teachers.types import (
    AttestationSource,
    AuthorizationProvider,
    DerivedArtifactDisposition,
    GenerationSettings,
    HumanApproverAttestation,
    IntendedUse,
    LicenseStatus,
    OutputRetention,
    OutputStorage,
    RedistributionScope,
    ReviewStatus,
    StudentModelFamily,
    TeacherModelFamily,
    TeacherModelRef,
    TeacherProvider,
    TeacherRecipe,
    TeacherRequest,
    ToolSpec,
    WrittenAuthorizationEvidence,
    seal_license_disposition,
    seal_output_use_policy,
    seal_written_authorization,
)

TODAY = date(2026, 7, 18)
CLAUDE_MODEL_ID = "anthropic.claude-fable-5-20260701-v1:0"
CLAUDE_PROFILE_ID = "us.anthropic.claude-fable-5"
NOVA_MODEL_ID = "amazon.nova-pro-v1:0"
QWEN_TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture
def claude_model() -> TeacherModelRef:
    return TeacherModelRef(
        provider=TeacherProvider.BEDROCK,
        family=TeacherModelFamily.CLAUDE,
        model_id=CLAUDE_MODEL_ID,
        inference_profile_id=CLAUDE_PROFILE_ID,
    )


@pytest.fixture
def nova_model() -> TeacherModelRef:
    return TeacherModelRef(
        provider=TeacherProvider.BEDROCK,
        family=TeacherModelFamily.NOVA,
        model_id=NOVA_MODEL_ID,
    )


@pytest.fixture
def qwen_model() -> TeacherModelRef:
    return TeacherModelRef(
        provider=TeacherProvider.LOCAL,
        family=TeacherModelFamily.QWEN,
        model_id=QWEN_TEACHER_ID,
        revision="a" * 40,
    )


@pytest.fixture
def valid_output() -> dict[str, object]:
    return {
        "schema_version": "merchant_tagging.v1",
        "task": "merchant_tagging",
        "merchant_id": "merchant_1",
        "merchant_name": "Acme Cloud",
        "spend_category": "cloud",
        "tags": ["infrastructure", "recurring"],
        "confidence": 0.9,
    }


@pytest.fixture
def authorization_factory() -> Callable[..., WrittenAuthorizationEvidence]:
    def factory(
        *,
        covered_models: Sequence[str] = (CLAUDE_MODEL_ID,),
        covered_anthropic_models: Sequence[str] = (),
        uses: Sequence[IntendedUse] = (IntendedUse.SEQUENCE_KD,),
        student_ids: Sequence[str] = (QWEN_STUDENT_ID,),
        student_families: Sequence[StudentModelFamily] = (StudentModelFamily.QWEN,),
        storage: Sequence[OutputStorage] = (OutputStorage.IMMUTABLE_CACHE,),
        redistribution: RedistributionScope = RedistributionScope.INTERNAL_ONLY,
        effective_date: date = date(2026, 7, 1),
        expiration_date: date = date(2026, 8, 1),
    ) -> WrittenAuthorizationEvidence:
        return seal_written_authorization(
            provider=AuthorizationProvider.ANTHROPIC,
            covered_bedrock_model_ids=covered_models,
            covered_anthropic_model_ids=covered_anthropic_models,
            permitted_intended_uses=uses,
            permitted_student_model_ids=student_ids,
            permitted_student_families=student_families,
            permitted_storage=storage,
            derived_weight_redistribution_scope=redistribution,
            effective_date=effective_date,
            expiration_date=expiration_date,
            external_reference="provided_out_of_band",
            approver_attestation=HumanApproverAttestation(
                source=AttestationSource.PROVIDED_OUT_OF_BAND,
                approver="human-approver-attestation",
                attested_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
            ),
        )

    return factory


@pytest.fixture
def request_factory() -> Callable[..., TeacherRequest]:
    def factory(
        *,
        model: TeacherModelRef,
        intended_use: IntendedUse = IntendedUse.EVALUATION,
        written_authorization: WrittenAuthorizationEvidence | None = None,
        review_status: ReviewStatus = ReviewStatus.ALLOWED,
        license_status: LicenseStatus = LicenseStatus.APPROVED,
        license_id: str | None = None,
        recipe: TeacherRecipe | None = None,
        target_family: StudentModelFamily | None = None,
        student_model_id: str | None = None,
        retention: OutputRetention | None = None,
        output_storage: OutputStorage | None = None,
        artifact_disposition: DerivedArtifactDisposition | None = None,
        tools: tuple[ToolSpec, ...] = (),
        allow_tool_use: bool = False,
        split: SplitName = SplitName.TRAIN,
        input_payload: dict[str, object] | None = None,
        max_tokens: int = 64,
        seed: int | None = 7,
        additional_model_fields: dict[str, object] | None = None,
    ) -> TeacherRequest:
        derives = intended_use.derives_weights
        if recipe is None and derives:
            recipe = (
                TeacherRecipe.LOGIT_V1
                if intended_use is IntendedUse.LOGIT_KD
                else TeacherRecipe.SEQUENCE_V1
            )
        if derives:
            target_family = target_family or StudentModelFamily.QWEN
            student_model_id = student_model_id or QWEN_STUDENT_ID
            retention = retention or OutputRetention.RETAINED
            output_storage = output_storage or OutputStorage.IMMUTABLE_CACHE
            artifact_disposition = artifact_disposition or DerivedArtifactDisposition.INTERNAL_ONLY
        else:
            retention = retention or OutputRetention.NON_RETAINED
            output_storage = output_storage or OutputStorage.EPHEMERAL
            artifact_disposition = artifact_disposition or DerivedArtifactDisposition.NOT_APPLICABLE
        if license_id is None and model.family is TeacherModelFamily.QWEN:
            license_id = "Apache-2.0"

        output_policy = seal_output_use_policy(
            record_id="reviewed-output-use",
            record_version="2026-07-18.1",
            status=review_status,
            intended_use=intended_use,
            retention=retention,
            reviewer="human-reviewer",
        )
        return TeacherRequest(
            example_id="example_1",
            task=TaskId.MERCHANT_TAGGING,
            difficulty=Difficulty.MEDIUM,
            split=split,
            input=input_payload
            or {
                "descriptor": "ACME CLOUD",
                "mcc": "5734",
                "amount_minor": 12500,
                "currency": "USD",
                "memo": "monthly service",
            },
            recipe=recipe,
            intended_use=intended_use,
            target_family=target_family,
            student_model_id=student_model_id,
            output_storage=output_storage,
            derived_artifact_disposition=artifact_disposition,
            model=model,
            settings=GenerationSettings(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop_sequences=("END",),
                seed=seed,
                do_sample=False,
                system_prompt="Return one JSON object matching the task schema.",
                additional_model_fields=(
                    {"performanceConfig": "standard"}
                    if additional_model_fields is None
                    else additional_model_fields
                ),
            ),
            output_use_policy=output_policy,
            license_disposition=seal_license_disposition(
                model_id=model.model_id,
                status=license_status,
                license_id=license_id,
                evidence_version="2026-07-18.1",
            ),
            provider_policy=PROJECT_PROVIDER_POLICY,
            written_authorization=written_authorization,
            tools=tools,
            allow_tool_use=allow_tool_use,
        )

    return factory
