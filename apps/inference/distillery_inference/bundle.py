"""Immutable model-bundle loading, checksum verification, and offline readiness."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.schemas import ArtifactKind, DemoModelArmId, FinanceTaskId

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

REGISTRY_FILENAME = "serving_registry.json"
SHA256SUMS_RELATIVE = "integrity/SHA256SUMS"


class BundleChecksums(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    files: dict[StrictStr, StrictStr] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def _validate_digests(cls, value: dict[str, str]) -> dict[str, str]:
        for relative_path, digest in value.items():
            _validate_relative_path(relative_path)
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"malformed digest for {relative_path}")
        return value


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["distillery.serving_artifact.v1"] = (
        "distillery.serving_artifact.v1"
    )
    artifact_id: StrictStr = Field(min_length=1)
    model_id: StrictStr = Field(min_length=1)
    arm_id: DemoModelArmId
    kind: ArtifactKind
    relative_path: StrictStr = Field(min_length=1)
    display_name: StrictStr = Field(min_length=1)
    purpose: StrictStr = Field(min_length=1)
    base_model_id: StrictStr = Field(min_length=1)
    base_revision: StrictStr = Field(min_length=1)
    tokenizer_revision: StrictStr = Field(min_length=1)
    supported_tasks: list[FinanceTaskId] = Field(min_length=1)
    checksums: dict[StrictStr, StrictStr] = Field(min_length=1)
    recipe: StrictStr | None = None
    proof_status: StrictStr | None = None
    promotion_status: Literal["promoted", "not_promoted", "unknown"] = "unknown"
    excluded: bool = False
    exclusion_reason: StrictStr | None = None
    stats: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_revision", "tokenizer_revision")
    @classmethod
    def _pinned_revision(cls, value: str) -> str:
        if _REVISION_RE.fullmatch(value) is None:
            raise ValueError("model/tokenizer revision must be a 40-char git sha")
        return value

    @field_validator("relative_path")
    @classmethod
    def _safe_relative(cls, value: str) -> str:
        _validate_relative_path(value)
        return value

    @field_validator("checksums")
    @classmethod
    def _checksum_shape(cls, value: dict[str, str]) -> dict[str, str]:
        for relative_path, digest in value.items():
            _validate_relative_path(relative_path)
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"malformed digest for {relative_path}")
        return value


class ServingRegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["distillery.serving_registry.v1"] = (
        "distillery.serving_registry.v1"
    )
    run_id: StrictStr = Field(min_length=1)
    dataset_id: StrictStr | None = None
    endpoint_id: StrictStr = Field(min_length=1)
    base_model_id: StrictStr = Field(min_length=1)
    base_revision: StrictStr = Field(min_length=1)
    tokenizer_revision: StrictStr = Field(min_length=1)
    base_relative_path: StrictStr = Field(min_length=1)
    artifacts: list[ArtifactManifest] = Field(min_length=1)

    @field_validator("base_revision", "tokenizer_revision")
    @classmethod
    def _pinned_revision(cls, value: str) -> str:
        if _REVISION_RE.fullmatch(value) is None:
            raise ValueError("base/tokenizer revision must be a 40-char git sha")
        return value

    @field_validator("base_relative_path")
    @classmethod
    def _safe_base_path(cls, value: str) -> str:
        _validate_relative_path(value)
        return value


class LoadedBundle:
    """Validated immutable bundle ready for offline serving."""

    def __init__(
        self,
        *,
        root: Path,
        registry: ServingRegistryDocument,
        artifacts_by_model: dict[str, ArtifactManifest],
        artifacts_by_id: dict[str, ArtifactManifest],
    ) -> None:
        self.root = root
        self.registry = registry
        self.artifacts_by_model = artifacts_by_model
        self.artifacts_by_id = artifacts_by_id

    def resolve_artifact(
        self,
        *,
        model_id: str,
        artifact_id: str,
    ) -> ArtifactManifest:
        artifact = self.artifacts_by_model.get(model_id)
        if artifact is None:
            raise InferenceError(
                InferenceErrorCode.MODEL_NOT_IN_REGISTRY,
                f"Model {model_id} is not present in the sealed serving registry.",
                http_status=404,
            )
        if artifact.artifact_id != artifact_id:
            raise InferenceError(
                InferenceErrorCode.ARTIFACT_ID_MISMATCH,
                (
                    f"Requested artifact_id {artifact_id} does not match registry "
                    f"artifact {artifact.artifact_id} for model {model_id}."
                ),
                http_status=409,
                details={
                    "model_id": model_id,
                    "requested_artifact_id": artifact_id,
                    "registry_artifact_id": artifact.artifact_id,
                },
            )
        if artifact.excluded:
            raise InferenceError(
                InferenceErrorCode.ARTIFACT_NOT_SERVABLE,
                artifact.exclusion_reason
                or f"Artifact {artifact_id} is excluded from serving.",
                http_status=409,
            )
        return artifact

    def artifact_path(self, artifact: ArtifactManifest) -> Path:
        return self.root.joinpath(*PurePosixPath(artifact.relative_path).parts)

    def base_path(self) -> Path:
        return self.root.joinpath(*PurePosixPath(self.registry.base_relative_path).parts)


def _validate_relative_path(relative_path: str) -> None:
    if not relative_path or "\x00" in relative_path or "\\" in relative_path:
        raise ValueError(f"unsafe relative path {relative_path!r}")
    pure = PurePosixPath(relative_path)
    parts = relative_path.split("/")
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or pure.as_posix() != relative_path
    ):
        raise ValueError(f"unsafe relative path {relative_path!r}")


def sha256_file(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            f"checksum path does not exist: {path}",
            http_status=500,
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            f"checksum path must be a regular non-symlink file: {path}",
            http_status=500,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, separator, relative_path = stripped.partition("  ")
        if not separator or not relative_path:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        relative_path = relative_path.strip()
        digest = digest.strip()
        _validate_relative_path(relative_path)
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"malformed digest on line {line_number}")
        if relative_path in result:
            raise ValueError(f"duplicate SHA256SUMS path: {relative_path}")
        result[relative_path] = digest
    return result


def verify_checksums(*, root: Path, entries: dict[str, str]) -> None:
    violations: list[str] = []
    try:
        root_resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            f"model bundle root missing: {root}",
            http_status=500,
        ) from exc
    for relative_path, expected in sorted(entries.items()):
        path = root_resolved.joinpath(*PurePosixPath(relative_path).parts)
        current = root_resolved
        symlink = False
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            if current.is_symlink():
                violations.append(f"symlink:{relative_path}")
                symlink = True
                break
        if symlink:
            continue
        if not path.is_file():
            violations.append(f"missing:{relative_path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            violations.append(f"mismatch:{relative_path}")
    if violations:
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            "Model bundle checksum verification failed.",
            http_status=500,
            details={"violations": violations},
        )


def load_serving_bundle(root: Path) -> LoadedBundle:
    """Load registry, validate manifests/tasks/revisions, and verify checksums."""
    if not root.is_dir():
        raise InferenceError(
            InferenceErrorCode.SERVING_NOT_READY,
            f"model bundle root is not a directory: {root}",
            http_status=503,
        )
    registry_path = root / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise InferenceError(
            InferenceErrorCode.SERVING_NOT_READY,
            f"serving registry missing at {registry_path}",
            http_status=503,
        )
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = ServingRegistryDocument.model_validate(registry_payload)

    sums_path = root / SHA256SUMS_RELATIVE
    if not sums_path.is_file():
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            f"integrity checksum file missing: {sums_path}",
            http_status=500,
        )
    checksum_entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    if REGISTRY_FILENAME not in checksum_entries:
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
            "SHA256SUMS must include serving_registry.json",
            http_status=500,
        )
    verify_checksums(root=root, entries=checksum_entries)

    artifacts_by_model: dict[str, ArtifactManifest] = {}
    artifacts_by_id: dict[str, ArtifactManifest] = {}
    for artifact in registry.artifacts:
        if artifact.model_id in artifacts_by_model:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"duplicate model_id in registry: {artifact.model_id}",
                http_status=500,
            )
        if artifact.artifact_id in artifacts_by_id:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"duplicate artifact_id in registry: {artifact.artifact_id}",
                http_status=500,
            )
        if artifact.base_model_id != registry.base_model_id:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"artifact {artifact.artifact_id} base_model_id mismatch",
                http_status=500,
            )
        if artifact.base_revision != registry.base_revision:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"artifact {artifact.artifact_id} base_revision mismatch",
                http_status=500,
            )
        if artifact.tokenizer_revision != registry.tokenizer_revision:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"artifact {artifact.artifact_id} tokenizer_revision mismatch",
                http_status=500,
            )
        for relative_path, digest in artifact.checksums.items():
            joined = f"{artifact.relative_path}/{relative_path}"
            expected = checksum_entries.get(joined)
            if expected is None:
                raise InferenceError(
                    InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
                    f"artifact checksum path missing from SHA256SUMS: {joined}",
                    http_status=500,
                )
            if expected != digest:
                raise InferenceError(
                    InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED,
                    f"artifact checksum digest mismatch for {joined}",
                    http_status=500,
                )
        artifacts_by_model[artifact.model_id] = artifact
        artifacts_by_id[artifact.artifact_id] = artifact

    base_marker = root.joinpath(*PurePosixPath(registry.base_relative_path).parts)
    if not base_marker.is_dir():
        raise InferenceError(
            InferenceErrorCode.SERVING_NOT_READY,
            f"base model directory missing: {base_marker}",
            http_status=503,
        )

    return LoadedBundle(
        root=root.resolve(strict=True),
        registry=registry,
        artifacts_by_model=artifacts_by_model,
        artifacts_by_id=artifacts_by_id,
    )
