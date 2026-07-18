"""Artifact layout, checksum files, and load-instruction generation."""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from distillery.contracts.hashing import sha256_hex
from distillery.recipes.base import require_pinned_revision

# Relative paths under a run output prefix (local directory or s3://.../runs/<run_id>/).
RUN_ARTIFACT_LAYOUT: dict[str, str] = {
    "manifest": "manifest.json",
    "manifest_sha256": "manifest.sha256",
    "materialization": "inputs/materialization.json",
    "synthesis_responses": "synthesis/responses.jsonl",
    "synthesis_provenance": "synthesis/provenance.json",
    "training_metrics": "training/metrics.jsonl",
    "training_final_adapter": "training/final/adapter_model.safetensors",
    "model_adapter_dir": "model/adapter",
    "model_merged_dir": "model/merged",
    "model_load_test": "model/load_test.json",
    "evaluation_predictions": "evaluation/predictions.jsonl",
    "evaluation_metrics": "evaluation/metrics.json",
    "systems_profile": "systems/profile.json",
    "gross_cost": "costs/gross_cost.json",
    "events_log": "logs/events.jsonl",
    "integrity_sums": "integrity/SHA256SUMS",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactFileSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required: bool = True
    description: str = ""


class ArtifactLayout(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    root_prefix: str
    files: tuple[ArtifactFileSpec, ...]

    def resolve(self, key_or_path: str) -> str:
        """Join root prefix with a layout key or relative path."""
        rel = RUN_ARTIFACT_LAYOUT.get(key_or_path, key_or_path)
        _validate_checksum_entry(rel, "0" * 64)
        root = self.root_prefix.rstrip("/")
        return f"{root}/{rel}"


def build_run_artifact_layout(*, run_id: str, root_prefix: str) -> ArtifactLayout:
    files = tuple(
        ArtifactFileSpec(
            relative_path=path,
            required=key
            not in {
                "model_merged_dir",
                "evaluation_predictions",
                "evaluation_metrics",
                "systems_profile",
                "gross_cost",
            },
            description=key,
        )
        for key, path in RUN_ARTIFACT_LAYOUT.items()
    )
    return ArtifactLayout(run_id=run_id, root_prefix=root_prefix, files=files)


def sha256_file(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise ValueError(f"checksum path does not exist: {path}") from None
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"checksum path must be a regular non-symlink file: {path}")
    data = path.read_bytes()
    return sha256_hex(data)


def _validate_checksum_entry(relative_path: str, digest: str) -> None:
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"malformed SHA-256 digest for {relative_path!r}")
    if not relative_path or "\x00" in relative_path or "\\" in relative_path:
        raise ValueError(f"unsafe checksum path {relative_path!r}")
    pure = PurePosixPath(relative_path)
    raw_parts = relative_path.split("/")
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or pure.as_posix() != relative_path
    ):
        raise ValueError(f"unsafe checksum path {relative_path!r}")


def write_sha256sums(
    entries: Mapping[str, str],
    *,
    destination: Path,
) -> Path:
    """
    Write a GNU-style SHA256SUMS file.

    Each line: ``<hex>  <relative_path>`` (two spaces).
    """
    for rel, digest in entries.items():
        _validate_checksum_entry(rel, digest)
    lines = [f"{digest}  {rel}" for rel, digest in sorted(entries.items())]
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + ("\n" if lines else "")
    destination.write_text(text, encoding="utf-8")
    return destination


def parse_sha256sums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, rel = stripped.partition("  ")
        if not rel:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        relative_path = rel.strip()
        digest = digest.strip()
        _validate_checksum_entry(relative_path, digest)
        if relative_path in result:
            raise ValueError(
                f"duplicate SHA256SUMS path on line {line_number}: {relative_path}"
            )
        result[relative_path] = digest
    return result


def verify_sha256sums(
    entries: Mapping[str, str],
    *,
    root: Path,
) -> tuple[str, ...]:
    """Verify regular files without allowing traversal, symlinks, or root escape."""
    violations: list[str] = []
    try:
        root_resolved = root.resolve(strict=True)
    except FileNotFoundError:
        return ("missing_root",)
    if not root_resolved.is_dir():
        return ("root_not_directory",)

    for rel, expected in sorted(entries.items()):
        try:
            _validate_checksum_entry(rel, expected)
        except ValueError as exc:
            violations.append(f"invalid_entry:{rel}:{exc}")
            continue
        path = root_resolved.joinpath(*PurePosixPath(rel).parts)
        current = root_resolved
        symlink_found = False
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                violations.append(f"symlink:{rel}")
                symlink_found = True
                break
        if symlink_found:
            continue
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            violations.append(f"missing:{rel}")
            continue
        if not resolved.is_relative_to(root_resolved):
            violations.append(f"escape:{rel}")
            continue
        try:
            mode = resolved.lstat().st_mode
        except FileNotFoundError:
            violations.append(f"missing:{rel}")
            continue
        if not stat.S_ISREG(mode):
            violations.append(f"non_regular:{rel}")
            continue
        try:
            actual = sha256_file(resolved)
        except ValueError:
            violations.append(f"non_regular:{rel}")
            continue
        if actual != expected:
            violations.append(f"mismatch:{rel}")
    return tuple(violations)


def generate_load_instructions(
    *,
    student_base_id: str,
    student_revision: str,
    adapter_uri: str,
    merged_uri: str | None = None,
    tokenizer_uri: str | None = None,
) -> str:
    """Human/machine load instructions for unmodified transformers + peft."""
    require_pinned_revision(student_revision, role="student")
    lines = [
        "# Distillery TinyFable load instructions",
        "# Requires pinned transformers/peft; do not use repository custom code.",
        "",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import PeftModel",
        "",
        f"base_id = {student_base_id!r}",
        f"revision = {student_revision!r}",
        f"adapter_uri = {adapter_uri!r}",
        "tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)",
    ]
    if tokenizer_uri:
        lines.append(f"# Optional tokenizer snapshot: {tokenizer_uri!r}")
    lines.extend(
        [
            "base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision)",
            "model = PeftModel.from_pretrained(base, adapter_uri)",
            "model.eval()",
        ]
    )
    if merged_uri:
        lines.extend(
            [
                "",
                "# Optional merged weights (no peft required):",
                f"# merged = AutoModelForCausalLM.from_pretrained({merged_uri!r})",
            ]
        )
    return "\n".join(lines) + "\n"


def write_load_test_document(
    destination: Path,
    *,
    student_base_id: str,
    student_revision: str,
    adapter_uri: str,
    checksums: Mapping[str, str],
    merged_uri: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "distillery.load_test.v1",
        "student_base_id": student_base_id,
        "student_revision": student_revision,
        "adapter_uri": adapter_uri,
        "merged_uri": merged_uri,
        "checksums": dict(checksums),
        "load_instructions": generate_load_instructions(
            student_base_id=student_base_id,
            student_revision=student_revision,
            adapter_uri=adapter_uri,
            merged_uri=merged_uri,
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def materialization_sidecar(
    *,
    accepted_example_ids: Iterable[str],
    rejected_example_ids: Iterable[str],
    label_source_counts: Mapping[str, int],
    recipe_id: str,
    sampler_order_hash: str,
    completion_token_counts: Mapping[str, int],
    completion_tokenizer_sha256: str,
    canonical_records_sha256: str,
    source_file_sha256: str,
    provenance_sha256: str,
    record_sha256: Mapping[str, str],
) -> dict[str, Any]:
    digest_fields = {
        "completion_tokenizer_sha256": completion_tokenizer_sha256,
        "canonical_records_sha256": canonical_records_sha256,
        "source_file_sha256": source_file_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if any(_SHA256_RE.fullmatch(digest) is None for digest in digest_fields.values()):
        raise ValueError("materialization digests must be lowercase SHA-256 hex")
    if any(_SHA256_RE.fullmatch(digest) is None for digest in record_sha256.values()):
        raise ValueError("record_sha256 values must be lowercase SHA-256 hex")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in completion_token_counts.values()
    ):
        raise ValueError("completion token counts must be positive integers")
    return {
        "schema_version": "distillery.materialization.v1",
        "mode": "validation_only",
        "executed": False,
        "execution_gate": "unimplemented_hard_stop",
        "recipe_id": recipe_id,
        "accepted_example_ids": list(accepted_example_ids),
        "rejected_example_ids": list(rejected_example_ids),
        "label_source_counts": dict(label_source_counts),
        "completion_token_counts": dict(completion_token_counts),
        "completion_token_count_source": "student_tokenizer",
        "completion_tokenizer_sha256": completion_tokenizer_sha256,
        "canonical_records_sha256": canonical_records_sha256,
        "source_file_sha256": source_file_sha256,
        "provenance_sha256": provenance_sha256,
        "record_sha256": dict(record_sha256),
        "sampler_order_hash": sampler_order_hash,
    }
