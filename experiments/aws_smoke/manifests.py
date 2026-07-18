"""Strict emergency manifests using typed Distillery evidence fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import (
    AutoResolverInput,
    ManifestCompletionEvidence,
    ManifestCost,
    ManifestDatasetRef,
    ManifestMemoryDryRunEvidence,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestQLoRAConfig,
    ManifestRecipe,
    ManifestRuntime,
    ManifestSpecialTokenMapEvidence,
    ManifestTraining,
    ManifestTrainingCapabilityEvidence,
    SealedRunManifest,
    manifest_capability_binding_sha256,
    manifest_length_configuration_sha256,
    manifest_memory_dry_run_evidence_sha256,
    manifest_model_configuration_sha256,
    manifest_training_configuration_sha256,
)
from distillery.training.batching import (
    DEFAULT_FINANCE_MIXTURE,
    BatchPlan,
    MixtureSpec,
    SamplerExample,
    plan_batches,
)
from experiments.aws_smoke.channels import CANONICAL_MANIFEST_FILENAME
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import (
    DEFAULT_EMERGENCY_PROFILE,
    REQUIRED_ARMS,
    EmergencyTrainingProfile,
    RunArm,
    arm_objective,
    arm_recipe_resolved,
)
from experiments.aws_smoke.tokenization import (
    ArmTokenizationEvidence,
    TokenizationEvidence,
)

# Matches experiments.aws_smoke.dataset_subset emergency task mixture. Difficulty
# LRA is not independently enforced because joint corpus cells can diverge at N=6.
EMERGENCY_FINANCE_MIXTURE = MixtureSpec(
    task_weights={
        "variance_analysis": 0.8,
        "cash_reconciliation": 0.2,
    },
    difficulty_weights=dict(DEFAULT_FINANCE_MIXTURE.difficulty_weights),
)


def run_id_for_arm(arm: RunArm, *, campaign_slug: str = "awssmoke") -> str:
    return f"run_{campaign_slug}-{arm.replace('_', '-')}"


def job_name_for_arm(arm: RunArm, *, manifest_sha256: str) -> str:
    base = f"aws-smoke-{arm.replace('_', '-')}-{manifest_sha256[:12]}"
    return base[:63].rstrip("-")


def canonical_manifest_path(output_dir: Path, arm: RunArm) -> Path:
    return output_dir / arm / "manifest" / CANONICAL_MANIFEST_FILENAME


def build_sampler_plan(
    *,
    example_ids: list[str],
    tasks: list[str],
    difficulties: list[str],
    completion_token_counts: Mapping[str, int],
    prompt_token_counts: Mapping[str, int],
    total_token_counts: Mapping[str, int],
    record_sha256: Mapping[str, str],
    seed: int,
    tokenizer_sha256: str,
    microbatch_size: int,
) -> BatchPlan:
    """Invoke the production sampler with actual tokenizer and record evidence."""
    if not (len(example_ids) == len(tasks) == len(difficulties)):
        raise ValueError("example_ids/tasks/difficulties length mismatch")
    for name, mapping in (
        ("completion_token_counts", completion_token_counts),
        ("prompt_token_counts", prompt_token_counts),
        ("total_token_counts", total_token_counts),
        ("record_sha256", record_sha256),
    ):
        if set(mapping) != set(example_ids):
            missing = sorted(set(example_ids) - set(mapping))
            extra = sorted(set(mapping) - set(example_ids))
            raise ValueError(
                f"{name} must exactly cover examples; missing={missing} extra={extra}"
            )
    examples = [
        SamplerExample(
            example_id=example_id,
            task=task,
            difficulty=difficulty,
            completion_tokens=int(completion_token_counts[example_id]),
            prompt_tokens=int(prompt_token_counts[example_id]),
            total_tokens=int(total_token_counts[example_id]),
            completion_token_source="student_tokenizer",
            completion_tokenizer_sha256=tokenizer_sha256,
            record_sha256=str(record_sha256[example_id]),
        )
        for example_id, task, difficulty in zip(
            example_ids,
            tasks,
            difficulties,
            strict=True,
        )
    ]
    return plan_batches(
        examples,
        seed=seed,
        microbatch_size=microbatch_size,
        mixture=EMERGENCY_FINANCE_MIXTURE,
        require_difficulty_lra=False,
    )


def completion_provenance_sha256(
    tokenization: ArmTokenizationEvidence,
) -> str:
    """Bind every non-wire token count and source record into one typed hash."""
    return content_sha256(
        {
            "target_source": tokenization.target_source,
            "prompt_token_counts": dict(sorted(tokenization.prompt_token_counts.items())),
            "total_token_counts": dict(sorted(tokenization.total_token_counts.items())),
            "original_completion_token_counts": dict(
                sorted(tokenization.original_completion_token_counts.items())
            ),
            "truncated_example_ids": sorted(tokenization.truncated_example_ids),
            "record_sha256": dict(sorted(tokenization.record_sha256.items())),
            "source_file_sha256": tokenization.source_file_sha256,
            "canonical_records_sha256": tokenization.canonical_records_sha256,
            "completion_record_sha256": dict(
                sorted(tokenization.completion_record_sha256.items())
            ),
            "teacher_responses_sha256": tokenization.teacher_responses_sha256,
        }
    )


def initialization_fingerprint(
    evidence: EmergencyEvidence,
    profile: EmergencyTrainingProfile,
) -> str:
    return content_sha256(
        {
            "student_model_id": evidence.student_model_id,
            "student_revision": evidence.student_revision,
            "student_model_config_sha256": evidence.student_model_config_sha256,
            "seed": profile.seed,
            "precision_mode": profile.precision_mode,
            "rank": profile.lora_rank,
            "alpha": profile.lora_alpha,
            "dropout": profile.lora_dropout,
            "target_modules": list(profile.lora_target_modules),
            "gradient_checkpointing": profile.gradient_checkpointing,
        }
    )


def _validate_memory_probe_input(
    *,
    evidence: EmergencyEvidence,
    profile: EmergencyTrainingProfile,
) -> None:
    measured = profile.memory_probe_evidence
    if measured is None:
        raise ValueError(
            "logit_kd requires non-placeholder measured A10G memory probe evidence"
        )
    expected = {
        "precision_mode": profile.precision_mode,
        "student_model_id": evidence.student_model_id,
        "student_revision": evidence.student_revision,
        "teacher_model_id": evidence.teacher_model_id,
        "teacher_revision": evidence.teacher_revision,
        "max_length": profile.max_length,
        "max_completion": profile.max_completion,
        "vocab_chunk_size": profile.vocab_chunk,
        "microbatch": profile.microbatch,
        "grad_accumulation": profile.grad_accumulation,
        "runtime_image_digest": evidence.image_digest,
        "instance_type": profile.instance_type,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(measured, key)}
        for key, value in expected.items()
        if getattr(measured, key) != value
    }
    if not measured.passed:
        mismatches["passed"] = {"expected": True, "actual": False}
    if mismatches:
        raise ValueError(f"memory probe input mismatch: {mismatches}")


def _wire_memory_evidence(
    *,
    provisional: SealedRunManifest,
    profile: EmergencyTrainingProfile,
) -> ManifestMemoryDryRunEvidence:
    measured = profile.memory_probe_evidence
    if measured is None:
        raise ValueError("memory probe evidence is required")
    payload: dict[str, Any] = {
        "schema_version": "distillery.memory_dry_run.v2",
        "passed": measured.passed,
        "binding_sha256": manifest_capability_binding_sha256(provisional),
        "training_config_sha256": manifest_training_configuration_sha256(provisional),
        "teacher_model_config_sha256": manifest_model_configuration_sha256(
            provisional.models.teacher
        ),
        "student_model_config_sha256": manifest_model_configuration_sha256(
            provisional.models.student
        ),
        "length_config_sha256": manifest_length_configuration_sha256(provisional),
        "runtime_image_digest": provisional.runtime.image_digest,
        "instance_type": provisional.runtime.instance_type,
        "recipe_id": "logit.v1",
        "teacher_model_id": provisional.models.teacher.id,
        "teacher_revision": provisional.models.teacher.revision,
        "student_model_id": provisional.models.student.id,
        "student_revision": provisional.models.student.revision,
        "max_length": provisional.training.max_length,
        "max_completion": provisional.training.qlora.max_completion,
        "vocab_chunk_size": provisional.training.qlora.vocab_chunk,
        "peak_memory_bytes": measured.peak_memory_bytes,
        "capacity_memory_bytes": measured.capacity_memory_bytes,
        "headroom_bytes": measured.headroom_bytes,
        "device_type": measured.device_type,
        "probe_id": measured.probe_id,
    }
    payload["evidence_sha256"] = manifest_memory_dry_run_evidence_sha256(payload)
    return ManifestMemoryDryRunEvidence.model_validate(payload)


def _emergency_config(
    *,
    arm: RunArm,
    profile: EmergencyTrainingProfile,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "profile": profile.name,
        "learning_rate": profile.learning_rate,
        "microbatch": profile.microbatch,
        "grad_accumulation": profile.grad_accumulation,
        "gradient_checkpointing": profile.gradient_checkpointing,
        "deterministic_algorithms": profile.deterministic_algorithms,
        "precision_mode": profile.precision_mode,
        "memory_probe_evidence": (
            profile.memory_probe_evidence.model_dump(mode="json")
            if profile.memory_probe_evidence is not None
            else None
        ),
        "artifact_reserve_seconds": profile.artifact_reserve_seconds,
        "shutdown_margin_seconds": profile.shutdown_margin_seconds,
        "max_runtime_seconds": profile.max_runtime_seconds,
        "model_channel_materialization": (
            "regular_file_copy_verified_snapshot_manifest_sha256_v1"
        ),
        "protocol_deviation": (
            "DEVIATION:bf16_lora_no_nf4"
            if profile.precision_mode == "bf16_lora"
            else None
        ),
    }


def manifest_arm(manifest: SealedRunManifest) -> RunArm:
    arm = manifest.tags.get("Arm")
    if arm not in {"oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"}:
        raise ValueError(f"manifest has invalid emergency arm tag: {arm!r}")
    return arm


def manifest_objective(manifest: SealedRunManifest) -> dict[str, Any]:
    arm = manifest_arm(manifest)
    objective = arm_objective(arm)
    objective["kd_weight"] = manifest.training.qlora.kd_weight
    objective["hard_ce_weight"] = manifest.training.qlora.hard_ce_weight
    if manifest.tags.get("ScientificRole") != objective["scientific_role"]:
        raise ValueError("manifest scientific-role tag does not match arm")
    return objective


def manifest_emergency_config(manifest: SealedRunManifest) -> dict[str, Any]:
    raw = manifest.tags.get("EmergencyConfig")
    if raw is None:
        raise ValueError("manifest lacks sealed EmergencyConfig tag")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("EmergencyConfig tag must decode to an object")
    expected_hash = manifest.tags.get("EmergencyConfigSha256")
    if expected_hash != content_sha256(payload):
        raise ValueError("EmergencyConfig tag hash mismatch")
    if payload.get("arm") != manifest_arm(manifest):
        raise ValueError("EmergencyConfig arm differs from manifest Arm")
    return payload


def build_emergency_manifest(
    *,
    arm: RunArm,
    evidence: EmergencyEvidence,
    dataset_id: str,
    dataset_uri: str,
    dataset_sha256: str,
    split_sha256: dict[str, str],
    sampler_plan: BatchPlan,
    tokenization: ArmTokenizationEvidence,
    profile: EmergencyTrainingProfile | None = None,
    created_at: datetime | None = None,
) -> SealedRunManifest:
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if tokenization.arm != arm:
        raise ValueError(f"tokenization evidence arm {tokenization.arm} != {arm}")
    if arm == "sequence_kd":
        if tokenization.target_source != "pre_materialized_teacher":
            raise ValueError("sequence_kd requires pre-materialized teacher targets")
        if tokenization.teacher_responses_sha256 is None:
            raise ValueError("sequence_kd requires sealed teacher-response records")
    elif tokenization.target_source != "oracle":
        raise ValueError(f"{arm} must use oracle hard targets")

    objective = arm_objective(arm, p)
    config = _emergency_config(arm=arm, profile=p)
    config["teacher_responses_sha256"] = tokenization.teacher_responses_sha256
    config["student_model_config_sha256"] = evidence.student_model_config_sha256
    config["teacher_model_config_sha256"] = evidence.teacher_model_config_sha256
    config_sha256 = content_sha256(config)
    init_fingerprint = initialization_fingerprint(evidence, p)
    if arm in {"ce_ablation", "logit_kd"}:
        _validate_memory_probe_input(evidence=evidence, profile=p)
    capability = ManifestTrainingCapabilityEvidence(
        special_token_maps=ManifestSpecialTokenMapEvidence(
            teacher=evidence.teacher_special_token_map,
            student=evidence.student_special_token_map,
        ),
        auto_resolver_input=AutoResolverInput(
            cheaper_baseline_satisfies_gate=False,
            usable_responses_exist=True,
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
            allowed_teacher_can_fill_within_ceiling=True,
        ),
    )
    qlora = ManifestQLoRAConfig(
        rank=p.lora_rank,
        alpha=p.lora_alpha,
        dropout=p.lora_dropout,
        target_modules=p.lora_target_modules,
        max_completion=p.max_completion,
        logit_temperature=p.logit_temperature,
        kd_weight=float(objective["kd_weight"]),
        hard_ce_weight=float(objective["hard_ce_weight"]),
        vocab_chunk=p.vocab_chunk,
        capability_evidence=capability,
    )
    completion_evidence = ManifestCompletionEvidence(
        source_file_sha256=tokenization.source_file_sha256,
        canonical_records_sha256=tokenization.canonical_records_sha256,
        record_sha256=tokenization.completion_record_sha256,
        provenance_sha256=completion_provenance_sha256(tokenization),
        completion_token_counts=dict(
            sorted(tokenization.completion_token_counts.items())
        ),
        completion_tokenizer_sha256=evidence.student_tokenizer_sha256,
        label_source_counts={
            "teacher" if arm == "sequence_kd" else "oracle": len(
                tokenization.completion_token_counts
            )
        },
        accepted_example_count=len(tokenization.completion_token_counts),
    )
    protocol_sha256 = content_sha256(
        {
            "config": config,
            "objective": objective,
            "sampler_order_hash": sampler_plan.sampler_order_hash,
            "completion_evidence": completion_evidence.model_dump(mode="json"),
        }
    )
    run_id = run_id_for_arm(arm)
    output_prefix = evidence.artifact_s3_prefix.rstrip("/") + f"/runs/{run_id}/"
    provisional = SealedRunManifest.model_construct(
        run_id=run_id,
        created_at=created_at or datetime(2026, 7, 18, 16, 0, tzinfo=UTC),
        dataset=ManifestDatasetRef(
            dataset_id=dataset_id,  # type: ignore[arg-type]
            uri=dataset_uri,
            sha256=dataset_sha256,
            split_sha256=split_sha256,  # type: ignore[arg-type]
        ),
        models=ManifestModels(
            teacher=ManifestModelSpec(
                id=evidence.teacher_model_id,
                revision=evidence.teacher_revision,
                tokenizer_sha256=evidence.teacher_tokenizer_sha256,
                chat_template_sha256=evidence.teacher_chat_template_sha256,
            ),
            student=ManifestModelSpec(
                id=evidence.student_model_id,
                revision=evidence.student_revision,
                tokenizer_sha256=evidence.student_tokenizer_sha256,
                chat_template_sha256=evidence.student_chat_template_sha256,
            ),
        ),
        recipe=ManifestRecipe(
            requested=arm_recipe_resolved(arm),  # type: ignore[arg-type]
            resolved=arm_recipe_resolved(arm),  # type: ignore[arg-type]
            resolver_reasons=("explicit_request",),
        ),
        training=ManifestTraining(
            seed=p.seed,
            max_steps=p.max_steps,
            token_budget=0,
            max_length=p.max_length,
            qlora=qlora,
            completion_evidence=completion_evidence,
        ),
        proof_protocol=ManifestProofProtocol(
            id=evidence.proof_protocol_id,
            sha256=evidence.proof_protocol_sha256,
        ),
        runtime=ManifestRuntime(
            backend="sagemaker",
            region=evidence.aws_region,
            instance_type=p.instance_type,
            image_digest=evidence.image_digest,  # type: ignore[arg-type]
        ),
        cost=ManifestCost(
            max_run_usd=p.max_run_usd,
            estimate_low_usd=p.estimate_low_usd,
            estimate_high_usd=p.estimate_high_usd,
        ),
        output=ManifestOutput(prefix=output_prefix),
        package_lock_hash=evidence.package_lock_hash,
        source_revision=evidence.source_revision,
        license_dispositions={
            evidence.student_model_id: evidence.license_disposition,
            evidence.teacher_model_id: evidence.license_disposition,
        },
        tags={
            "RunMode": "smoke",
            "EmergencyProfile": p.name,
            "Arm": arm,
            "ScientificRole": str(objective["scientific_role"]),
            "DistinctTrainingSignal": str(objective["distinct_training_signal"]).lower(),
            "EquivalentTo": str(objective["equivalent_to"] or "none"),
            "TeacherRuntime": str(objective["teacher_runtime"]),
            "HardTargetSource": str(objective["hard_target_source"]),
            "TreatmentOverhead": str(objective["treatment_overhead"]),
            "EnableNetworkIsolation": "true",
            "MaxRuntimeInSeconds": str(p.max_runtime_seconds),
            "QuotaInstanceCount": str(p.quota_instance_count),
            "HourlyUsd": str(p.hourly_usd),
            "InitializationFingerprint": init_fingerprint,
            "ProtocolDeviation": str(config["protocol_deviation"] or "none"),
            "EmergencyConfig": json.dumps(
                config,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "EmergencyConfigSha256": config_sha256,
            "TrainingProtocolSha256": protocol_sha256,
        },
        sampler_order_hash=sampler_plan.sampler_order_hash,
    )
    if arm in {"ce_ablation", "logit_kd"}:
        memory = _wire_memory_evidence(provisional=provisional, profile=p)
        final_capability = capability.model_copy(update={"memory_dry_run": memory})
        final_qlora = qlora.model_copy(
            update={"capability_evidence": final_capability}
        )
        final_training = provisional.training.model_copy(
            update={"qlora": final_qlora}
        )
        provisional = provisional.model_copy(update={"training": final_training})
    return SealedRunManifest.model_validate(provisional.model_dump(mode="json"))


def comparable_arm_payload(manifest: SealedRunManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    qlora = dict(payload["training"]["qlora"])
    qlora.pop("kd_weight", None)
    qlora.pop("hard_ce_weight", None)
    capability = dict(qlora.get("capability_evidence") or {})
    capability["memory_dry_run"] = None
    qlora["capability_evidence"] = capability
    payload["training"]["qlora"] = qlora
    payload["training"]["completion_evidence"] = None
    payload.pop("run_id")
    payload.pop("output")
    payload.pop("recipe")
    # Production sampler hashes bind actual completion lengths, which are a
    # sequence-KD treatment difference. write_arm_manifests separately proves
    # the resulting example-id order is identical across arms.
    payload.pop("sampler_order_hash")
    tags = dict(payload["tags"])
    for key in (
        "Arm",
        "ScientificRole",
        "DistinctTrainingSignal",
        "EquivalentTo",
        "TeacherRuntime",
        "HardTargetSource",
        "TreatmentOverhead",
        "EmergencyConfig",
        "EmergencyConfigSha256",
        "TrainingProtocolSha256",
    ):
        tags.pop(key, None)
    payload["tags"] = tags
    return payload


def assert_arms_comparable(manifests: dict[RunArm, SealedRunManifest]) -> None:
    if len(manifests) < 2:
        raise ValueError("need at least two manifests to compare")
    arms = sorted(manifests)
    baseline_arm = arms[0]
    baseline_hash = content_sha256(comparable_arm_payload(manifests[baseline_arm]))
    baseline_init = manifests[baseline_arm].tags["InitializationFingerprint"]
    for arm in arms[1:]:
        if content_sha256(comparable_arm_payload(manifests[arm])) != baseline_hash:
            raise ValueError(
                f"arm {arm} diverges from {baseline_arm} in shared initialization, "
                "dataset, sampler order, or training settings"
            )
        if manifests[arm].tags["InitializationFingerprint"] != baseline_init:
            raise ValueError("initialization fingerprint mismatch across arms")


def assert_kd_ablation_matched(
    ce_manifest: SealedRunManifest,
    kd_manifest: SealedRunManifest,
) -> None:
    if manifest_arm(ce_manifest) != "ce_ablation":
        raise ValueError("first manifest must be ce_ablation")
    if manifest_arm(kd_manifest) != "logit_kd":
        raise ValueError("second manifest must be logit_kd")
    if comparable_arm_payload(ce_manifest) != comparable_arm_payload(kd_manifest):
        raise ValueError("ce_ablation and logit_kd shared settings differ")
    if (
        ce_manifest.training.completion_evidence.model_dump(mode="json")
        != kd_manifest.training.completion_evidence.model_dump(mode="json")
    ):
        raise ValueError("ce_ablation and logit_kd hard-target evidence differs")
    ce_config = manifest_emergency_config(ce_manifest)
    kd_config = manifest_emergency_config(kd_manifest)
    ce_config.pop("arm")
    kd_config.pop("arm")
    if ce_config != kd_config:
        raise ValueError("ce_ablation and logit_kd runtime settings differ")
    if manifest_objective(ce_manifest)["equivalent_to"] != "oracle_sft":
        raise ValueError("ce_ablation must disclose oracle_sft equivalence")


def write_arm_manifests(
    *,
    output_dir: Path,
    evidence: EmergencyEvidence,
    dataset_id: str,
    dataset_uri: str,
    dataset_sha256: str,
    split_sha256: dict[str, str],
    example_ids: list[str],
    tasks: list[str],
    difficulties: list[str],
    tokenization_evidence: TokenizationEvidence,
    arms: tuple[RunArm, ...] = REQUIRED_ARMS,
    profile: EmergencyTrainingProfile | None = None,
) -> dict[RunArm, Path]:
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if evidence.data_content_sha256 != dataset_sha256:
        raise ValueError("dataset content hash does not match operator evidence")
    if p.memory_probe_evidence is None and evidence.memory_probe_evidence is not None:
        profile_payload = p.model_dump(mode="json")
        profile_payload["memory_probe_evidence"] = evidence.memory_probe_evidence.model_dump(
            mode="json"
        )
        p = EmergencyTrainingProfile.model_validate(profile_payload)
    if tokenization_evidence.student_tokenizer_sha256 != evidence.student_tokenizer_sha256:
        raise ValueError("tokenization tokenizer hash does not match run evidence")
    if (
        tokenization_evidence.student_chat_template_sha256
        != evidence.student_chat_template_sha256
    ):
        raise ValueError("tokenization chat-template hash does not match run evidence")
    if tokenization_evidence.student_special_token_map != evidence.student_special_token_map:
        raise ValueError("tokenization special-token map does not match run evidence")
    if tokenization_evidence.max_length != p.max_length:
        raise ValueError("tokenization max_length does not match profile")
    if tokenization_evidence.max_completion != p.max_completion:
        raise ValueError("tokenization max_completion does not match profile")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[RunArm, SealedRunManifest] = {}
    paths: dict[RunArm, Path] = {}
    plans: dict[RunArm, BatchPlan] = {}
    for arm in arms:
        arm_tokens = tokenization_evidence.arm(arm)
        plan = build_sampler_plan(
            example_ids=example_ids,
            tasks=tasks,
            difficulties=difficulties,
            completion_token_counts=arm_tokens.completion_token_counts,
            prompt_token_counts=arm_tokens.prompt_token_counts,
            total_token_counts=arm_tokens.total_token_counts,
            record_sha256=arm_tokens.record_sha256,
            seed=p.seed,
            tokenizer_sha256=evidence.student_tokenizer_sha256,
            microbatch_size=p.microbatch,
        )
        plans[arm] = plan
        manifest = build_emergency_manifest(
            arm=arm,
            evidence=evidence,
            dataset_id=dataset_id,
            dataset_uri=dataset_uri,
            dataset_sha256=dataset_sha256,
            split_sha256=split_sha256,
            sampler_plan=plan,
            tokenization=arm_tokens,
            profile=p,
        )
        manifests[arm] = manifest
        path = canonical_manifest_path(output_dir, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        arm_dir = path.parent.parent
        (arm_dir / "manifest.sha256").write_text(
            manifest.seal_sha256() + "\n",
            encoding="utf-8",
        )
        objective = manifest_objective(manifest)
        (arm_dir / "jobmeta.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "run_id": manifest.run_id,
                    "job_name": job_name_for_arm(
                        arm,
                        manifest_sha256=manifest.seal_sha256(),
                    ),
                    "manifest_sha256": manifest.seal_sha256(),
                    "manifest_channel_dir": str(path.parent),
                    "output_prefix": manifest.output.prefix,
                    "max_run_usd": manifest.cost.max_run_usd,
                    "max_runtime_seconds": p.max_runtime_seconds,
                    "scientific_role": objective["scientific_role"],
                    "distinct_training_signal": objective["distinct_training_signal"],
                    "equivalent_to": objective["equivalent_to"],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        paths[arm] = path

    first_arm = arms[0]
    if len(manifests) >= 2:
        assert_arms_comparable(manifests)
    baseline_order = plans[first_arm].order
    for arm, plan in plans.items():
        if plan.order != baseline_order:
            raise ValueError(f"arm {arm} produced a different trainer sample order")
    if {"ce_ablation", "logit_kd"} <= set(manifests):
        assert_kd_ablation_matched(
            manifests["ce_ablation"],
            manifests["logit_kd"],
        )
    index = {
        "schema_version": "distillery.aws_smoke.campaign.v2",
        "arms": {
            arm: {
                "manifest": str(paths[arm].relative_to(output_dir)),
                "manifest_sha256": manifests[arm].seal_sha256(),
                "run_id": manifests[arm].run_id,
                "job_name": job_name_for_arm(
                    arm,
                    manifest_sha256=manifests[arm].seal_sha256(),
                ),
                "output_prefix": manifests[arm].output.prefix,
                "scientific_role": manifest_objective(manifests[arm])[
                    "scientific_role"
                ],
                "distinct_training_signal": manifest_objective(manifests[arm])[
                    "distinct_training_signal"
                ],
                "equivalent_to": manifest_objective(manifests[arm])["equivalent_to"],
            }
            for arm in arms
        },
        "shared_student_revision": evidence.student_revision,
        "shared_initialization_fingerprint": initialization_fingerprint(evidence, p),
        "shared_sampler_order": list(plans[first_arm].order),
        "shared_sampler_order_hash": plans[first_arm].sampler_order_hash,
        "distinct_signal_count": sum(
            bool(manifest_objective(manifest)["distinct_training_signal"])
            for manifest in manifests.values()
        ),
        "control_disclosure": (
            "ce_ablation uses the same oracle targets and hard-CE objective as "
            "oracle_sft; it is a matched replication/control"
        ),
        "campaign_fingerprint": hashlib.sha256(
            (
                evidence.student_revision
                + initialization_fingerprint(evidence, p)
                + "|".join(arms)
            ).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "campaign_index.json").write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
