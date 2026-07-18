#!/usr/bin/env python3
"""Build one checksum-sealed serving bundle from verified real artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from distillery.contracts.manifest import SealedRunManifest

SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def safe_relative_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    return path


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != SHA256_LENGTH
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        safe_relative_path(relative)
        if relative in entries:
            raise ValueError(f"duplicate checksum path: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError(f"empty checksum file: {path}")
    return entries


def verify_sha256sums(root: Path, sums_path: Path) -> dict[str, str]:
    entries = parse_sha256sums(sums_path)
    for relative, expected in sorted(entries.items()):
        require_equal(
            f"checksum {relative}",
            sha256_file(root.joinpath(*PurePosixPath(relative).parts)),
            expected,
        )
    return entries


def verify_snapshot(base_dir: Path, record: dict[str, Any]) -> dict[str, str]:
    base = record["base"]
    snapshot_path = base_dir / "snapshot-manifest.json"
    require_equal(
        "snapshot manifest sha256",
        sha256_file(snapshot_path),
        base["snapshot_manifest_sha256"],
    )
    snapshot = load_json(snapshot_path)
    require_equal("base model id", snapshot.get("model_id"), base["model_id"])
    require_equal("base revision", snapshot.get("revision"), base["revision"])
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("snapshot manifest files must be a nonempty object")
    verified: dict[str, str] = {}
    for relative, evidence in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(evidence, dict):
            raise ValueError("invalid snapshot file evidence")
        expected_size = evidence.get("size")
        expected_digest = evidence.get("sha256")
        relative_path = safe_relative_path(relative)
        path = base_dir.joinpath(*relative_path.parts)
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            raise ValueError(f"snapshot size mismatch: {relative}")
        require_equal(
            f"snapshot checksum {relative}",
            sha256_file(path),
            expected_digest,
        )
        verified[relative] = str(expected_digest)
    require_equal(
        "base weights sha256",
        verified.get("model.safetensors"),
        base["weights_sha256"],
    )
    return verified


def copy_regular(source: Path, destination: Path) -> None:
    mode = source.lstat().st_mode
    if source.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"refusing non-regular source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.chmod(destination, 0o444)


def copy_tree_files(
    source_root: Path,
    destination_root: Path,
    relative_files: list[str],
) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for relative in sorted(relative_files):
        relative_path = safe_relative_path(relative)
        source = source_root.joinpath(*relative_path.parts)
        destination = destination_root.joinpath(*relative_path.parts)
        copy_regular(source, destination)
        checksums[relative] = sha256_file(destination)
    return checksums


def validate_real_evidence(
    *,
    record: dict[str, Any],
    manifest_path: Path,
    model_root: Path,
    output_root: Path,
    comparison_path: Path,
    base_predictions_path: Path,
    adapter_predictions_path: Path,
) -> None:
    manifest = SealedRunManifest.model_validate(load_json(manifest_path))
    require_equal(
        "manifest file sha256",
        sha256_file(manifest_path),
        record["manifest"]["file_sha256"],
    )
    require_equal(
        "manifest seal sha256",
        manifest.seal_sha256(),
        record["manifest"]["seal_sha256"],
    )
    require_equal("manifest run id", str(manifest.run_id), record["run_id"])
    require_equal(
        "manifest source revision",
        manifest.source_revision,
        record["manifest"]["source_revision"],
    )
    require_equal(
        "manifest training image",
        manifest.runtime.image_digest,
        record["manifest"]["training_image_digest"],
    )
    require_equal(
        "manifest student revision",
        manifest.models.student.revision,
        record["base"]["revision"],
    )
    require_equal(
        "manifest validation sha256",
        manifest.dataset.split_sha256["validation"],
        record["validation"]["split_sha256"],
    )
    require_equal(
        "record validation sha256",
        record["dataset"]["validation_sha256"],
        record["validation"]["split_sha256"],
    )
    require_equal(
        "manifest dataset sha256",
        manifest.dataset.sha256,
        record["dataset"]["content_sha256"],
    )

    verify_sha256sums(output_root, output_root / "integrity" / "SHA256SUMS")
    emergency = load_json(output_root / "training" / "emergency_run.json")
    require_equal("training status", emergency.get("status"), "completed")
    require_equal("training arm", emergency.get("arm"), "oracle_sft")
    require_equal(
        "training manifest sha256",
        emergency.get("manifest_sha256"),
        record["manifest"]["seal_sha256"],
    )
    require_equal(
        "training student revision",
        emergency.get("student_revision"),
        record["base"]["revision"],
    )
    require_equal(
        "training completed steps",
        emergency.get("completed_steps"),
        record["adapter"]["completed_steps"],
    )
    require_equal("adapter reload evidence", emergency.get("adapter_reload_passed"), True)
    load_test = load_json(output_root / "model" / "load_test.json")
    require_equal("fresh adapter load test", load_test.get("passed"), True)
    require_equal("fresh base load test", load_test.get("fresh_base_loaded"), True)
    tokenizer_evidence = load_json(output_root / "model" / "tokenizer_evidence.json")
    require_equal("tokenizer compatibility", tokenizer_evidence.get("compatible"), True)

    model_files = sorted(
        path.relative_to(model_root).as_posix() for path in model_root.rglob("*") if path.is_file()
    )
    output_adapter = output_root / "model" / "adapter"
    output_files = sorted(
        path.relative_to(output_adapter).as_posix()
        for path in output_adapter.rglob("*")
        if path.is_file()
    )
    require_equal("model/output adapter file set", model_files, output_files)
    for relative in model_files:
        require_equal(
            f"model/output adapter checksum {relative}",
            sha256_file(model_root / relative),
            sha256_file(output_adapter / relative),
        )
    require_equal(
        "adapter weights sha256",
        sha256_file(model_root / "adapter_model.safetensors"),
        record["adapter"]["weights_sha256"],
    )
    require_equal(
        "adapter config sha256",
        sha256_file(model_root / "adapter_config.json"),
        record["adapter"]["config_sha256"],
    )

    comparison = load_json(comparison_path)
    require_equal(
        "validation comparison sha256",
        sha256_file(comparison_path),
        record["validation"]["comparison_sha256"],
    )
    require_equal(
        "base predictions sha256",
        sha256_file(base_predictions_path),
        record["validation"]["base_predictions_sha256"],
    )
    require_equal(
        "adapter predictions sha256",
        sha256_file(adapter_predictions_path),
        record["validation"]["adapter_predictions_sha256"],
    )
    require_equal(
        "comparison validation sha256",
        comparison.get("validation_sha256"),
        record["validation"]["split_sha256"],
    )
    require_equal(
        "comparison examples",
        comparison.get("validation_examples"),
        record["validation"]["examples"],
    )
    require_equal(
        "comparison base weights",
        comparison["base"]["weights_sha256"],
        record["base"]["weights_sha256"],
    )
    require_equal(
        "comparison adapter weights",
        comparison["adapter"]["weights_sha256"],
        record["adapter"]["weights_sha256"],
    )
    require_equal(
        "comparison delta",
        comparison["comparison"]["primary_index_delta"],
        record["validation"]["primary_index_delta"],
    )
    require_equal(
        "comparison improvement claim",
        comparison["comparison"]["improvement_claimed"],
        False,
    )


def artifact_entry(
    *,
    record: dict[str, Any],
    arm_id: str,
    kind: str,
    relative_path: str,
    display_name: str,
    purpose: str,
    checksums: dict[str, str],
    stats: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    identity = record["base"] if arm_id == "student_base" else record["adapter"]
    return {
        "schema_version": "distillery.serving_artifact.v1",
        "artifact_id": identity["artifact_id"],
        "model_id": identity["serving_model_id"],
        "arm_id": arm_id,
        "kind": kind,
        "relative_path": relative_path,
        "display_name": display_name,
        "purpose": purpose,
        "base_model_id": record["base"]["model_id"],
        "base_revision": record["base"]["revision"],
        "tokenizer_revision": record["base"]["tokenizer_revision"],
        "supported_tasks": (
            ["transaction_review", "variance_analysis", "cash_reconciliation"]
            if arm_id == "student_base"
            else record["serving"]["supported_tasks"]
        ),
        "checksums": checksums,
        "recipe": None if arm_id == "student_base" else "sequence.v1",
        "proof_status": (None if arm_id == "student_base" else record["adapter"]["proof_status"]),
        "promotion_status": (
            "unknown" if arm_id == "student_base" else record["adapter"]["promotion_status"]
        ),
        "excluded": False,
        "exclusion_reason": None,
        "stats": stats,
        "source_provenance": provenance,
    }


def build_registry(
    *,
    record: dict[str, Any],
    base_checksums: dict[str, str],
    adapter_checksums: dict[str, str],
) -> dict[str, Any]:
    validation = record["validation"]
    shared_stats = {
        "student": {
            "id": record["base"]["model_id"],
            "revision": record["base"]["revision"],
        },
        "seed": validation["seed"],
        "data_hash": record["dataset"]["content_sha256"],
        "manifest_hash": record["manifest"]["seal_sha256"],
        "evaluation_scope": validation["scope"],
        "validation_examples": validation["examples"],
        "proof_status": "insufficient_evidence",
    }
    base_stats = {
        **shared_stats,
        "advertised_parameter_count": record["base"]["parameter_count"],
        "artifact_hash": record["base"]["weights_sha256"],
        "validation_primary_index": validation["base"]["primary_index"],
        "base_validation_primary_index": validation["base"]["primary_index"],
        "primary_index_delta": 0.0,
        "mean_latency_ms": validation["base"]["mean_latency_ms"],
        "p50_latency_ms": validation["base"]["p50_latency_ms"],
        "p95_latency_ms": validation["base"]["p95_latency_ms"],
    }
    adapter_stats = {
        **shared_stats,
        "advertised_parameter_count": record["base"]["parameter_count"],
        "adapter_parameter_count": record["adapter"]["parameter_count"],
        "recipe": "sequence.v1",
        "artifact_hash": record["adapter"]["weights_sha256"],
        "training_duration_seconds": record["adapter"]["training_duration_seconds"],
        "training_cost_usd": record["adapter"]["training_cost_usd"],
        "validation_primary_index": validation["adapter"]["primary_index"],
        "base_validation_primary_index": validation["base"]["primary_index"],
        "primary_index_delta": validation["primary_index_delta"],
        "mean_latency_ms": validation["adapter"]["mean_latency_ms"],
        "p50_latency_ms": validation["adapter"]["p50_latency_ms"],
        "p95_latency_ms": validation["adapter"]["p95_latency_ms"],
    }
    base_provenance = {
        "training_job_name": None,
        "source_uri": record["base"]["source_uri"],
        "manifest_sha256": None,
        "manifest_file_sha256": record["base"]["snapshot_manifest_sha256"],
        "model_tar_sha256": None,
        "output_tar_sha256": None,
        "weights_sha256": record["base"]["weights_sha256"],
        "base_revision": record["base"]["revision"],
        "training_image_digest": None,
        "validation_sha256": validation["split_sha256"],
    }
    adapter_provenance = {
        "training_job_name": record["training_job_name"],
        "source_uri": record["source"]["model_tar_uri"],
        "manifest_sha256": record["manifest"]["seal_sha256"],
        "manifest_file_sha256": record["manifest"]["file_sha256"],
        "model_tar_sha256": record["source"]["model_tar_sha256"],
        "output_tar_sha256": record["source"]["output_tar_sha256"],
        "weights_sha256": record["adapter"]["weights_sha256"],
        "base_revision": record["base"]["revision"],
        "training_image_digest": record["manifest"]["training_image_digest"],
        "validation_sha256": validation["split_sha256"],
    }
    return {
        "schema_version": "distillery.serving_registry.v1",
        "run_id": record["run_id"],
        "dataset_id": record["dataset_id"],
        "endpoint_id": record["endpoint_id"],
        "base_model_id": record["base"]["model_id"],
        "base_revision": record["base"]["revision"],
        "tokenizer_revision": record["base"]["tokenizer_revision"],
        "base_relative_path": "base",
        "artifacts": [
            artifact_entry(
                record=record,
                arm_id="student_base",
                kind="base",
                relative_path="base",
                display_name="Qwen2.5 0.5B base",
                purpose="Pinned untrained student comparator",
                checksums=base_checksums,
                stats=base_stats,
                provenance=base_provenance,
            ),
            artifact_entry(
                record=record,
                arm_id="oracle_sft",
                kind="peft_adapter",
                relative_path="adapters/oracle_sft",
                display_name="Oracle SFT baseline",
                purpose=(
                    "Real 8-step emergency hard-target baseline; measured, not proof-promoted"
                ),
                checksums=adapter_checksums,
                stats=adapter_stats,
                provenance=adapter_provenance,
            ),
        ],
    }


def write_bundle_sums(bundle_root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(bundle_root).as_posix()
        if relative == "integrity/SHA256SUMS":
            continue
        entries[relative] = sha256_file(path)
    sums_path = bundle_root / "integrity" / "SHA256SUMS"
    sums_path.parent.mkdir(parents=True, exist_ok=True)
    sums_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())),
        encoding="utf-8",
    )
    os.chmod(sums_path, 0o444)
    return entries


def write_deterministic_tar(bundle_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(bundle_root.rglob("*")):
                    relative = path.relative_to(bundle_root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o555 if path.is_dir() else 0o444
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--record", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--model-tar", type=Path, required=True)
    result.add_argument("--output-tar", type=Path, required=True)
    result.add_argument("--base-dir", type=Path, required=True)
    result.add_argument("--comparison", type=Path, required=True)
    result.add_argument("--base-predictions", type=Path, required=True)
    result.add_argument("--adapter-predictions", type=Path, required=True)
    result.add_argument("--bundle-root", type=Path, required=True)
    result.add_argument("--bundle-tar", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    record = load_json(args.record)
    require_equal(
        "model tar sha256",
        sha256_file(args.model_tar),
        record["source"]["model_tar_sha256"],
    )
    require_equal(
        "output tar sha256",
        sha256_file(args.output_tar),
        record["source"]["output_tar_sha256"],
    )
    base_files = verify_snapshot(args.base_dir, record)

    with tempfile.TemporaryDirectory(prefix="distillery-serving-bundle-") as temporary:
        temporary_root = Path(temporary)
        model_root = temporary_root / "model"
        output_root = temporary_root / "output"
        safe_extract(args.model_tar, model_root)
        safe_extract(args.output_tar, output_root)
        validate_real_evidence(
            record=record,
            manifest_path=args.manifest,
            model_root=model_root,
            output_root=output_root,
            comparison_path=args.comparison,
            base_predictions_path=args.base_predictions,
            adapter_predictions_path=args.adapter_predictions,
        )

        resolved_bundle = args.bundle_root.resolve()
        if resolved_bundle == Path("/") or len(resolved_bundle.parts) < 3:
            raise ValueError(f"unsafe bundle root: {resolved_bundle}")
        shutil.rmtree(resolved_bundle, ignore_errors=True)
        resolved_bundle.mkdir(parents=True)

        base_checksums = copy_tree_files(
            args.base_dir,
            resolved_bundle / "base",
            list(base_files),
        )
        copy_regular(
            args.base_dir / "snapshot-manifest.json",
            resolved_bundle / "base" / "snapshot-manifest.json",
        )
        base_checksums["snapshot-manifest.json"] = sha256_file(
            resolved_bundle / "base" / "snapshot-manifest.json"
        )
        adapter_files = sorted(
            path.relative_to(model_root).as_posix()
            for path in model_root.rglob("*")
            if path.is_file()
        )
        adapter_checksums = copy_tree_files(
            model_root,
            resolved_bundle / "adapters" / "oracle_sft",
            adapter_files,
        )

        evidence_sources = {
            "training/manifest.json": args.manifest,
            "training/emergency_run.json": output_root / "training" / "emergency_run.json",
            "training/metrics.jsonl": output_root / "training" / "metrics.jsonl",
            "training/load_test.json": output_root / "model" / "load_test.json",
            "training/tokenizer_evidence.json": output_root / "model" / "tokenizer_evidence.json",
            "training/predictions.jsonl": output_root / "evaluation" / "predictions.jsonl",
            "validation/comparison.json": args.comparison,
            "validation/student_base.predictions.jsonl": args.base_predictions,
            "validation/oracle_sft.predictions.jsonl": args.adapter_predictions,
        }
        for relative, source in evidence_sources.items():
            copy_regular(source, resolved_bundle / "evidence" / relative)

        registry = build_registry(
            record=record,
            base_checksums=base_checksums,
            adapter_checksums=adapter_checksums,
        )
        registry_path = resolved_bundle / "serving_registry.json"
        registry_path.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(registry_path, 0o444)
        sums = write_bundle_sums(resolved_bundle)

        write_deterministic_tar(resolved_bundle, args.bundle_tar)
        bundle_tar_sha256 = sha256_file(args.bundle_tar)
        require_equal(
            "serving bundle tar sha256",
            bundle_tar_sha256,
            record["serving"]["bundle_sha256"],
        )
        print(
            json.dumps(
                {
                    "adapter_weights_sha256": record["adapter"]["weights_sha256"],
                    "bundle_files": len(sums),
                    "bundle_root": str(resolved_bundle),
                    "bundle_tar": str(args.bundle_tar),
                    "bundle_tar_sha256": bundle_tar_sha256,
                    "event": "serving_bundle_materialized",
                    "manifest_sha256": record["manifest"]["seal_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
