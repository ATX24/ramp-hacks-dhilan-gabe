from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from experiments.qwen72b_fallback.cost import (
    P4DE_HOURLY_USD,
    P4DE_PRICE_SOURCE,
    REHEARSAL_HARD_CAP_USD,
    TRANSFER_HARD_CAP_USD,
    TRANSFER_HOURLY_USD,
    TRANSFER_PRICE_SOURCE,
    CostAction,
    seal_cost_evidence,
)
from experiments.qwen72b_fallback.finance_world_targets import rehearsal_corpus
from experiments.qwen72b_fallback.license_policy import verify_license_artifacts
from experiments.qwen72b_fallback.memory import (
    A100_80GB_VRAM_BYTES,
    Qwen72BMemoryProbeEvidence,
    memory_probe_measurement_sha256,
)
from experiments.qwen72b_fallback.pins import REVISION, sealed_identity
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile
from experiments.qwen72b_fallback.readiness import (
    ConflictEvidence,
    EcrImageEvidence,
    ExecutionAction,
    ExecutionAuthorization,
    IamScopeEvidence,
    LocalPolicyEvidence,
    ReviewClearanceEvidence,
    S3SnapshotEvidence,
    evaluate_readiness,
    required_confirmation,
)
from experiments.qwen72b_fallback.tokenizer_compat import (
    TokenizerCompatibilityEvidence,
    TokenizerPairEvidence,
    load_target_registry,
    seal_compatibility,
)

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIGEST = "sha256:" + ("3" * 64)
IMAGE_URI = f"225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training@{IMAGE_DIGEST}"


def local_policy() -> LocalPolicyEvidence:
    return LocalPolicyEvidence.seal(
        identity=sealed_identity(),
        license=verify_license_artifacts(ROOT),
        execution_bindings_bytes_sha256="4" * 64,
    )


def image_evidence() -> EcrImageEvidence:
    return EcrImageEvidence.seal(
        account_id="225989358036",
        region="us-east-1",
        repository="distillery-training",
        image_digest=IMAGE_DIGEST,
        image_uri=IMAGE_URI,
        image_binding_sha256="5" * 64,
        source_revision="a" * 40,
        package_lock_sha256="6" * 64,
        source_tree_sha256="7" * 64,
        qwen72b_trainer_packaged=True,
        attention_backend="sdpa_math",
        flash_attention_2_packaged=False,
    )


def memory_probe(
    profile: Qwen72BTrainingProfile,
    *,
    local: LocalPolicyEvidence | None = None,
    image: EcrImageEvidence | None = None,
) -> Qwen72BMemoryProbeEvidence:
    local = local or local_policy()
    image = image or image_evidence()
    peak = 60 * 1024**3
    names = tuple("NVIDIA A100-SXM4-80GB" for _ in range(8))
    peaks = tuple(peak for _ in range(8))
    capacities = tuple(A100_80GB_VRAM_BYTES for _ in range(8))
    shapes = tuple((1, 1024) for _ in range(8))
    sampler_hash = "8" * 64
    return Qwen72BMemoryProbeEvidence.seal(
        probe_id="qwen72b-probe-test",
        revision=REVISION,
        model_identity_sha256=local.identity.evidence_sha256,
        runtime_image_uri=image.image_uri,
        runtime_image_digest=image.image_digest,
        image_binding_sha256=image.image_binding_sha256,
        profile_sha256=profile.profile_sha256,
        device_names=names,
        per_rank_peak_memory_bytes=peaks,
        per_rank_capacity_bytes=capacities,
        measured_batch_shapes=shapes,
        model_load_completed=True,
        forward_completed=True,
        backward_completed=True,
        optimizer_step_completed=True,
        all_rank_acknowledgements=tuple(True for _ in range(8)),
        sampler_order_sha256=sampler_hash,
        probe_artifact_sha256=memory_probe_measurement_sha256(
            profile_sha256=profile.profile_sha256,
            device_names=names,
            peaks=peaks,
            capacities=capacities,
            shapes=shapes,
            sampler_order_sha256=sampler_hash,
        ),
    )


def tokenizer_evidence() -> TokenizerCompatibilityEvidence:
    registry = load_target_registry()
    pairs = tuple(
        TokenizerPairEvidence.seal(
            teacher_revision=REVISION,
            target_model_id=target.model_id,
            target_revision=target.revision,
            teacher_file_sha256=dict(registry.tokenizer_file_sha256),
            target_file_sha256=dict(registry.tokenizer_file_sha256),
            teacher_tokenizer_sha256=(
                "8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000"
            ),
            target_tokenizer_sha256=(
                "8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000"
            ),
            teacher_chat_template_sha256=registry.chat_template_sha256,
            target_chat_template_sha256=registry.chat_template_sha256,
            teacher_special_token_ids=dict(registry.special_token_ids),
            target_special_token_ids=dict(registry.special_token_ids),
        )
        for target in registry.targets
    )
    return seal_compatibility(pairs)


class FakeLiveVerifier:
    def __init__(
        self,
        *,
        action: ExecutionAction,
        profile: Qwen72BTrainingProfile | None,
    ) -> None:
        self.action = action
        self.profile = profile
        self.local = local_policy()
        self.image = image_evidence()
        self.finance = rehearsal_corpus() if profile is not None else None
        self.probe = (
            memory_probe(profile, local=self.local, image=self.image)
            if profile is not None
            else None
        )

    def verify_local_policy(self) -> LocalPolicyEvidence:
        return self.local

    def verify_reviews(self) -> ReviewClearanceEvidence:
        return ReviewClearanceEvidence.seal(
            review_packet_sha256=("1" * 64, "2" * 64),
            execution_bindings_bytes_sha256="4" * 64,
        )

    def verify_iam(self, action: ExecutionAction) -> IamScopeEvidence:
        materialize = action is ExecutionAction.MATERIALIZE
        return IamScopeEvidence.seal(
            action=action,
            caller_account_id="225989358036",
            caller_arn="arn:aws:iam::225989358036:user/test",
            role_arn=(
                "arn:aws:iam::225989358036:role/distillery-qwen72b-transfer"
                if materialize
                else "arn:aws:iam::225989358036:role/distillery-sagemaker-training"
            ),
            role_id="AROAABCDEFGHIJKLMNOP",
            instance_profile_arn=(
                "arn:aws:iam::225989358036:instance-profile/distillery-qwen72b-transfer-test"
                if materialize
                else None
            ),
            transfer_ami_id="ami-" + ("a" * 17) if materialize else None,
            transfer_subnet_id="subnet-" + ("b" * 17) if materialize else None,
            transfer_security_group_id="sg-" + ("c" * 17) if materialize else None,
            policy_document_sha256=("9" * 64,),
            allowed_s3_resources=("arn:aws:s3:::distillery-225989358036-us-east-1/models/*",),
        )

    def verify_conflicts(
        self,
        action: ExecutionAction,
        launch_name: str,
    ) -> ConflictEvidence:
        return ConflictEvidence.seal(
            action=action,
            requested_launch_name=launch_name,
            active_p4de_jobs=(),
            active_g5_jobs=(),
            active_14b_or_32b_jobs=(),
            active_transfer_instance_ids=(),
            duplicate_launches=(),
            orphan_resource_ids=(),
        )

    def verify_cost(
        self,
        action: ExecutionAction,
        profile: Qwen72BTrainingProfile | None,
    ) -> Any:
        if action is ExecutionAction.MATERIALIZE:
            return seal_cost_evidence(
                action=CostAction.MATERIALIZE,
                instance_type="c5n.9xlarge",
                hourly_usd=TRANSFER_HOURLY_USD,
                price_source=TRANSFER_PRICE_SOURCE,
                max_runtime_seconds=3 * 3600,
                hard_cap_usd=TRANSFER_HARD_CAP_USD,
                active_resources=(),
            )
        assert profile is not None
        return seal_cost_evidence(
            action=(
                CostAction.MEMORY_PROBE
                if action is ExecutionAction.MEMORY_PROBE
                else CostAction.REHEARSAL
            ),
            instance_type=profile.instance_type,
            hourly_usd=P4DE_HOURLY_USD,
            price_source=P4DE_PRICE_SOURCE,
            max_runtime_seconds=profile.max_runtime_seconds,
            hard_cap_usd=REHEARSAL_HARD_CAP_USD,
            active_resources=(),
        )

    def verify_s3_snapshot(self) -> S3SnapshotEvidence:
        return S3SnapshotEvidence.seal(
            bucket="distillery-225989358036-us-east-1",
            prefix=("models/Qwen/Qwen2.5-72B-Instruct/495f39366efef23836d0cfae4fbe635880d2be31"),
            inventory_sha256="a" * 64,
            object_body_sha256={"config.json": "b" * 64},
            object_sizes={"config.json": 1},
            snapshot_manifest_body_sha256="c" * 64,
            sha256sums_body_sha256="d" * 64,
            materialization_manifest_body_sha256="e" * 64,
        )

    def verify_tokenizer_compatibility(self) -> TokenizerCompatibilityEvidence:
        return tokenizer_evidence()

    def verify_ecr_image(self) -> EcrImageEvidence:
        return self.image

    def verify_memory_probe(self, **_kwargs: Any) -> Qwen72BMemoryProbeEvidence:
        assert self.probe is not None
        return self.probe

    def verify_finance_world_data(
        self,
        _profile: Qwen72BTrainingProfile,
    ) -> Any:
        assert self.finance is not None
        return self.finance


def make_authorization(
    *,
    action: ExecutionAction,
    profile: Qwen72BTrainingProfile | None,
    launch_name: str,
) -> ExecutionAuthorization:
    verifier = FakeLiveVerifier(action=action, profile=profile)
    report = evaluate_readiness(
        verifier,
        action=action,
        launch_name=launch_name,
        profile=profile,
        typed_confirmation=required_confirmation(action, launch_name),
        now_unix_seconds=int(time.time()),
    )
    assert report.authorization is not None
    return report.authorization


@pytest.fixture
def authorization_factory():
    return make_authorization


@pytest.fixture
def live_verifier_factory():
    return FakeLiveVerifier
