"""Model registry/stats metadata from sealed proof/artifact files."""

from __future__ import annotations

from typing import Any

from distillery_inference.bundle import ArtifactManifest, LoadedBundle
from distillery_inference.schemas import (
    ModelEntry,
    ModelRegistryResponse,
    ModelStats,
    ServingInfo,
    TeacherStudentRef,
)


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def stats_from_artifact(artifact: ArtifactManifest) -> ModelStats:
    raw = artifact.stats if isinstance(artifact.stats, dict) else {}
    teacher = None
    student = None
    teacher_raw = raw.get("teacher")
    student_raw = raw.get("student")
    if isinstance(teacher_raw, dict) and teacher_raw.get("id") and teacher_raw.get("revision"):
        teacher = TeacherStudentRef(
            id=str(teacher_raw["id"]),
            revision=str(teacher_raw["revision"]),
        )
    if isinstance(student_raw, dict) and student_raw.get("id") and student_raw.get("revision"):
        student = TeacherStudentRef(
            id=str(student_raw["id"]),
            revision=str(student_raw["revision"]),
        )
    promotion = artifact.promotion_status
    if promotion not in {"promoted", "not_promoted", "unknown"}:
        promotion = "unknown"
    return ModelStats(
        advertised_parameter_count=_optional_int(raw.get("advertised_parameter_count")),
        adapter_parameter_count=_optional_int(raw.get("adapter_parameter_count")),
        compression_ratio=_optional_number(raw.get("compression_ratio")),
        recipe=_optional_str(artifact.recipe or raw.get("recipe")),
        teacher=teacher,
        student=student,
        seed=_optional_int(raw.get("seed")),
        data_hash=_optional_str(raw.get("data_hash")),
        manifest_hash=_optional_str(raw.get("manifest_hash")),
        artifact_hash=_optional_str(
            raw.get("artifact_hash")
            or artifact.checksums.get("adapter_model.safetensors")
            or next(iter(artifact.checksums.values()), None)
        ),
        training_duration_seconds=_optional_number(raw.get("training_duration_seconds")),
        training_cost_usd=_optional_number(raw.get("training_cost_usd")),
        iid_primary_index=_optional_number(raw.get("iid_primary_index")),
        iid_ci_low=_optional_number(raw.get("iid_ci_low")),
        iid_ci_high=_optional_number(raw.get("iid_ci_high")),
        ood_retention=_optional_number(raw.get("ood_retention")),
        ood_ci_low=_optional_number(raw.get("ood_ci_low")),
        ood_ci_high=_optional_number(raw.get("ood_ci_high")),
        proof_status=_optional_str(artifact.proof_status or raw.get("proof_status")),
        promotion_status=promotion,
    )


def build_model_registry_response(
    bundle: LoadedBundle,
    *,
    endpoint_id: str,
) -> ModelRegistryResponse:
    models: list[ModelEntry] = []
    for artifact in bundle.registry.artifacts:
        availability = "unavailable" if artifact.excluded else "live"
        reason = artifact.exclusion_reason if artifact.excluded else None
        models.append(
            ModelEntry(
                model_id=artifact.model_id,
                arm_id=artifact.arm_id,
                artifact_id=artifact.artifact_id,
                display_name=artifact.display_name,
                purpose=artifact.purpose,
                kind=artifact.kind,
                excluded=artifact.excluded,
                exclusion_reason=artifact.exclusion_reason,
                supported_tasks=list(artifact.supported_tasks),
                serving=ServingInfo(
                    availability=availability,
                    endpoint_id=None if artifact.excluded else endpoint_id,
                    artifact_id=None if artifact.excluded else artifact.artifact_id,
                    reason=reason,
                ),
                stats=stats_from_artifact(artifact),
            )
        )
    return ModelRegistryResponse(
        run_id=bundle.registry.run_id,
        dataset_id=bundle.registry.dataset_id,
        endpoint_id=endpoint_id,
        models=models,
    )
