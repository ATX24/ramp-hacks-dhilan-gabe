"""Sealed emergency manifests: one arm / job / output prefix, shared student pin."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import (
    ManifestCost,
    ManifestDatasetRef,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestRecipe,
    ManifestRuntime,
    ManifestTraining,
    SealedRunManifest,
)
from distillery.training.batching import (
    DEFAULT_FINANCE_MIXTURE,
    SamplerExample,
    plan_batches,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import (
    DEFAULT_EMERGENCY_PROFILE,
    EmergencyTrainingProfile,
    RunArm,
    arm_objective,
    arm_recipe_resolved,
)


def run_id_for_arm(arm: RunArm, *, campaign_slug: str = "awssmoke") -> str:
    """Deterministic run_id body; keeps arms distinct and SageMaker-name friendly."""
    body = f"{campaign_slug}-{arm.replace('_', '-')}"
    return f"run_{body}"


def job_name_for_arm(arm: RunArm, *, manifest_sha256: str) -> str:
    """Separate Training Job name per arm (never shared across objectives)."""
    suffix = manifest_sha256[:12]
    base = f"aws-smoke-{arm.replace('_', '-')}-{suffix}"
    return base[:63].rstrip("-")


def shared_sampler_order_hash(
    *,
    example_ids: list[str],
    tasks: list[str],
    difficulties: list[str],
    seed: int,
    tokenizer_sha256: str,
) -> str:
    """Deterministic sampler hash shared by all arms for the same subset order."""
    if not (len(example_ids) == len(tasks) == len(difficulties)):
        raise ValueError("example_ids/tasks/difficulties length mismatch")
    examples = [
        SamplerExample(
            example_id=example_id,
            task=task,
            difficulty=difficulty,
            completion_tokens=32,
            completion_token_source="student_tokenizer",
            completion_tokenizer_sha256=tokenizer_sha256,
        )
        for example_id, task, difficulty in zip(
            example_ids, tasks, difficulties, strict=True
        )
    ]
    plan = plan_batches(
        examples,
        seed=seed,
        microbatch_size=1,
        mixture=DEFAULT_FINANCE_MIXTURE,
    )
    return plan.sampler_order_hash


def build_emergency_manifest(
    *,
    arm: RunArm,
    evidence: EmergencyEvidence,
    dataset_id: str,
    dataset_uri: str,
    dataset_sha256: str,
    split_sha256: dict[str, str],
    sampler_order_hash: str,
    profile: EmergencyTrainingProfile | None = None,
    created_at: datetime | None = None,
    include_sequence_kd_gate: bool = False,
    teacher_responses_present: bool = False,
) -> SealedRunManifest:
    """Build one sealed Distillery manifest for a single emergency arm."""
    if arm == "sequence_kd":
        if not include_sequence_kd_gate:
            raise ValueError("sequence_kd arm disabled unless explicitly gated")
        if not teacher_responses_present:
            raise ValueError(
                "sequence_kd requires pre-materialized teacher responses; refusing"
            )

    p = profile or DEFAULT_EMERGENCY_PROFILE
    objective = arm_objective(arm)
    resolved = arm_recipe_resolved(arm)
    run_id = run_id_for_arm(arm)
    output_prefix = (
        evidence.artifact_s3_prefix.rstrip("/") + f"/runs/{run_id}/"
    )

    qlora: dict[str, Any] = {
        "rank": p.lora_rank,
        "alpha": p.lora_alpha,
        "dropout": p.lora_dropout,
        "max_completion": p.max_completion,
        "logit_temperature": p.logit_temperature,
        "kd_weight": float(objective["kd_weight"]),
        "hard_ce_weight": float(objective["hard_ce_weight"]),
        "vocab_chunk": p.vocab_chunk,
        "learning_rate": p.learning_rate,
        "microbatch": p.microbatch,
        "grad_accumulation": p.grad_accumulation,
        "emergency_profile": p.name,
        "arm": arm,
        "objective": objective,
        "license_disposition": evidence.license_disposition,
        "output_use_disposition": evidence.output_use_disposition,
    }

    return SealedRunManifest(
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
            requested=resolved,  # type: ignore[arg-type]
            resolved=resolved,  # type: ignore[arg-type]
            resolver_reasons=("explicit_request",),
        ),
        training=ManifestTraining(
            seed=p.seed,
            max_steps=p.max_steps,
            token_budget=0,
            max_length=p.max_length,
            qlora=qlora,
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
            "EnableNetworkIsolation": "false",
            "NetworkIsolationReason": (
                "Pinned model snapshots are downloaded or mounted during the job"
            ),
            "MaxRuntimeInSeconds": str(p.max_runtime_seconds),
            "QuotaInstanceCount": str(p.quota_instance_count),
            "HourlyUsd": str(p.hourly_usd),
        },
        sampler_order_hash=sampler_order_hash,
    )


def comparable_arm_payload(manifest: SealedRunManifest) -> dict[str, Any]:
    """Fields that must match across arms for a fair emergency comparison."""
    payload = manifest.model_dump(mode="json")
    training = dict(payload["training"])
    qlora = dict(training.get("qlora") or {})
    for key in ("arm", "objective", "kd_weight", "hard_ce_weight"):
        qlora.pop(key, None)
    training["qlora"] = qlora
    payload["training"] = training
    payload.pop("run_id")
    payload.pop("output")
    payload.pop("recipe")
    tags = dict(payload.get("tags") or {})
    tags.pop("Arm", None)
    payload["tags"] = tags
    return payload


def assert_arms_comparable(manifests: dict[RunArm, SealedRunManifest]) -> None:
    """All arms start from the same pinned student revision/config/order."""
    if len(manifests) < 2:
        raise ValueError("need at least two manifests to compare")
    arms = sorted(manifests)
    baseline_arm = arms[0]
    baseline = comparable_arm_payload(manifests[baseline_arm])
    baseline_hash = content_sha256(baseline)
    for arm in arms[1:]:
        other = comparable_arm_payload(manifests[arm])
        other_hash = content_sha256(other)
        if other_hash != baseline_hash:
            raise ValueError(
                f"arm {arm} is not comparable to {baseline_arm}: "
                "student pin, dataset, seed, sampler order, or training config diverged"
            )
        baseline_student = manifests[baseline_arm].models.student.revision
        if manifests[arm].models.student.revision != baseline_student:
            raise ValueError("student revision mismatch across arms")
        baseline_order = manifests[baseline_arm].sampler_order_hash
        if manifests[arm].sampler_order_hash != baseline_order:
            raise ValueError("sampler_order_hash mismatch across arms")


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
    arms: tuple[RunArm, ...] = ("oracle_sft", "ce_ablation", "logit_kd"),
    profile: EmergencyTrainingProfile | None = None,
    teacher_responses_present: bool = False,
) -> dict[RunArm, Path]:
    p = profile or DEFAULT_EMERGENCY_PROFILE
    order_hash = shared_sampler_order_hash(
        example_ids=example_ids,
        tasks=tasks,
        difficulties=difficulties,
        seed=p.seed,
        tokenizer_sha256=evidence.student_tokenizer_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[RunArm, SealedRunManifest] = {}
    paths: dict[RunArm, Path] = {}
    for arm in arms:
        manifest = build_emergency_manifest(
            arm=arm,
            evidence=evidence,
            dataset_id=dataset_id,
            dataset_uri=dataset_uri,
            dataset_sha256=dataset_sha256,
            split_sha256=split_sha256,
            sampler_order_hash=order_hash,
            profile=p,
            include_sequence_kd_gate=arm == "sequence_kd",
            teacher_responses_present=teacher_responses_present,
        )
        manifests[arm] = manifest
        path = output_dir / f"manifest_{arm}.json"
        text = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        sha_path = output_dir / f"manifest_{arm}.sha256"
        sha_path.write_text(manifest.seal_sha256() + "\n", encoding="utf-8")
        meta = {
            "arm": arm,
            "run_id": manifest.run_id,
            "job_name": job_name_for_arm(arm, manifest_sha256=manifest.seal_sha256()),
            "manifest_sha256": manifest.seal_sha256(),
            "output_prefix": manifest.output.prefix,
            "max_run_usd": manifest.cost.max_run_usd,
            "max_runtime_seconds": p.max_runtime_seconds,
        }
        (output_dir / f"jobmeta_{arm}.json").write_text(
            json.dumps(meta, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        paths[arm] = path
    assert_arms_comparable(manifests)
    index = {
        "arms": {
            arm: {
                "manifest": paths[arm].name,
                "manifest_sha256": manifests[arm].seal_sha256(),
                "run_id": manifests[arm].run_id,
                "job_name": job_name_for_arm(
                    arm, manifest_sha256=manifests[arm].seal_sha256()
                ),
                "output_prefix": manifests[arm].output.prefix,
            }
            for arm in arms
        },
        "shared_student_revision": evidence.student_revision,
        "shared_sampler_order_hash": order_hash,
        "campaign_fingerprint": hashlib.sha256(
            order_hash.encode("utf-8") + evidence.student_revision.encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "campaign_index.json").write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
