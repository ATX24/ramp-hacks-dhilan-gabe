#!/usr/bin/env python3
"""Strict schema, atomic-write, and ECR binding helpers for image manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_VERSION = "distillery.training.image.v2"
IMAGE_NAME = "distillery-training"
REPOSITORY_NAME = "distillery-training"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REGION_RE = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-[1-9][0-9]*$")
SAFE_COMPONENT_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is malformed JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_ecr_repository_uri(
    uri: str,
    *,
    account: str,
    region: str,
    repository: str = REPOSITORY_NAME,
) -> str:
    if not re.fullmatch(r"[0-9]{12}", account):
        raise ValueError(f"invalid STS account: {account!r}")
    if not REGION_RE.fullmatch(region):
        raise ValueError(f"invalid AWS region: {region!r}")
    if repository != REPOSITORY_NAME or not SAFE_COMPONENT_RE.fullmatch(repository):
        raise ValueError(f"repository is not allowlisted: {repository!r}")
    expected = f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}"
    if uri != expected:
        raise ValueError(
            "ECR repository URI must exactly match the STS account, selected region, "
            f"and allowlisted repository: expected {expected!r}, found {uri!r}"
        )
    return expected


def validate_manifest_semantics(payload: dict[str, Any]) -> None:
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {payload['schema_version']!r}")
    if payload["image_name"] != IMAGE_NAME:
        raise ValueError("unexpected image_name")
    if payload["repository_name"] != REPOSITORY_NAME:
        raise ValueError("unexpected repository_name")
    if "latest" in payload["tag"].lower():
        raise ValueError("tag must not contain latest")

    source = payload["source"]
    expected_tag = (
        f"pinned-training-{source['commit_sha'][:12]}-{payload['package_lock_sha256'][:12]}"
    )
    if payload["tag"] != expected_tag:
        raise ValueError(f"tag must be derived from source and lock hashes: {expected_tag}")
    if source["commit_bound"]:
        if not source["clean"]:
            raise ValueError("commit_bound source must be clean")
        if source["reviewed_commit_sha"] != source["commit_sha"]:
            raise ValueError("commit_bound source must equal the reviewed commit")

    compatibility = payload["ml_compatibility"]
    if compatibility["status"] == "compatible":
        expected = compatibility["expected"]
        actual = compatibility["actual"]
        expected_values = {
            "torch_version": payload["base"]["torch_version"],
            "torch_cuda_version": payload["base"]["cuda_version"],
            "cudnn_major": payload["base"]["cudnn_major"],
            "bitsandbytes_version": "0.44.1",
        }
        for field, value in expected_values.items():
            if expected[field] != value:
                raise ValueError(f"compatible manifest has unexpected {field}")
        if actual["torch_version"] != expected["torch_version"]:
            raise ValueError("compatible manifest torch lock does not match base")
        if actual["bitsandbytes_version"] != expected["bitsandbytes_version"]:
            raise ValueError("compatible manifest bitsandbytes lock does not match")
        if actual["forbidden_packages"]:
            raise ValueError("compatible manifest contains forbidden packages")
        if actual["unexpected_accelerator_packages"]:
            raise ValueError("compatible manifest contains unexpected accelerator packages")
        if compatibility["reasons"]:
            raise ValueError("compatible manifest must not contain blockers")
    elif not compatibility["reasons"]:
        raise ValueError("blocked manifest must explain its compatibility blockers")

    config_id = payload["local"]["config_id"]
    if config_id is not None and payload["dry_run"]:
        raise ValueError("local config ID cannot be recorded on a dry-run manifest")

    registry = payload["registry"]
    if registry["verified"]:
        expected_digest_uri = f"{registry['repository_uri']}@{registry['image_digest']}"
        if registry["digest_uri"] != expected_digest_uri:
            raise ValueError("registry.digest_uri must equal repository_uri@image_digest exactly")
        if payload["local"]["config_id"] is None:
            raise ValueError("verified registry manifest requires local.config_id")
        if payload["ml_compatibility"]["status"] != "compatible":
            raise ValueError("verified registry manifest requires compatible ML lock")
        if not source["commit_bound"]:
            raise ValueError("verified registry manifest requires commit-bound source")
        if payload["dry_run"]:
            raise ValueError("verified registry manifest cannot be a dry run")
    else:
        registry_values = (
            registry["repository_uri"],
            registry["image_digest"],
            registry["digest_uri"],
            registry["verified_at"],
            registry["scan_status"],
            registry["critical_findings"],
            registry["high_findings"],
        )
        if any(value is not None for value in registry_values):
            raise ValueError("unverified registry fields must all be null")


def validate_manifest(payload: dict[str, Any], schema_path: Path) -> None:
    schema = read_json_object(schema_path)
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(str(component) for component in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValueError(f"manifest schema validation failed: {details}")
    validate_manifest_semantics(payload)


def atomic_write_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    schema_path: Path,
) -> None:
    validate_manifest(payload, schema_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_validated_manifest(path: Path, schema_path: Path) -> dict[str, Any]:
    payload = read_json_object(path)
    validate_manifest(payload, schema_path)
    return payload


def require_hex(value: str, pattern: re.Pattern[str], field: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"{field} has invalid format")
    return value


def command_create(args: argparse.Namespace) -> None:
    compatibility_result = read_json_object(args.compatibility_result)
    compatibility_config = read_json_object(args.compatibility_config)
    source_clean = args.source_clean == "true"
    commit_bound = args.commit_bound == "true"
    reviewed = args.reviewed_commit_sha or None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "image_name": IMAGE_NAME,
        "repository_name": REPOSITORY_NAME,
        "source": {
            "commit_sha": require_hex(args.commit_sha, HEX40_RE, "commit_sha"),
            "reviewed_commit_sha": (
                require_hex(reviewed, HEX40_RE, "reviewed_commit_sha") if reviewed else None
            ),
            "tree_sha256": require_hex(
                args.tree_sha256,
                HEX64_RE,
                "tree_sha256",
            ),
            "clean": source_clean,
            "commit_bound": commit_bound,
        },
        "package_lock_sha256": require_hex(
            args.package_lock_sha256,
            HEX64_RE,
            "package_lock_sha256",
        ),
        "base": {
            "reference": compatibility_config["base_image"],
            "digest": compatibility_config["base_image"].split("@", maxsplit=1)[1],
            "torch_version": compatibility_config["torch_version"],
            "cuda_version": compatibility_config["torch_cuda_version"],
            "cudnn_major": compatibility_config["cudnn_major"],
        },
        "ml_compatibility": {
            "status": compatibility_result["status"],
            "expected": compatibility_result["expected"],
            "actual": compatibility_result["actual"],
            "reasons": compatibility_result["reasons"],
        },
        "tag": args.tag,
        "local": {
            "config_id": None,
        },
        "registry": {
            "repository_uri": None,
            "image_digest": None,
            "digest_uri": None,
            "verified": False,
            "verified_at": None,
            "scan_status": None,
            "critical_findings": None,
            "high_findings": None,
        },
        "dry_run": True,
        "created_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_manifest(args.output, payload, schema_path=args.schema)


def command_set_local(args: argparse.Namespace) -> None:
    payload = load_validated_manifest(args.manifest, args.schema)
    if payload["ml_compatibility"]["status"] != "compatible":
        raise ValueError("cannot record local image while ML compatibility is blocked")
    if not payload["source"]["commit_bound"] or not payload["source"]["clean"]:
        raise ValueError("cannot record local image from uncommitted source")
    if not SHA256_RE.fullmatch(args.config_id):
        raise ValueError("local config ID must be sha256:<64 lowercase hex>")
    payload["local"]["config_id"] = args.config_id
    payload["dry_run"] = False
    atomic_write_manifest(args.manifest, payload, schema_path=args.schema)


def command_set_registry(args: argparse.Namespace) -> None:
    payload = load_validated_manifest(args.manifest, args.schema)
    validate_ecr_repository_uri(
        args.repository_uri,
        account=args.account,
        region=args.region,
        repository=payload["repository_name"],
    )
    if payload["local"]["config_id"] is None:
        raise ValueError("cannot bind registry digest without local config ID")
    if not SHA256_RE.fullmatch(args.image_digest):
        raise ValueError("registry image digest must be sha256:<64 lowercase hex>")
    if args.scan_status != "COMPLETE":
        raise ValueError("registry scan must be COMPLETE")
    if args.critical_findings > args.max_critical:
        raise ValueError(
            f"critical image findings {args.critical_findings} exceed policy {args.max_critical}"
        )
    if args.high_findings > args.max_high:
        raise ValueError(f"high image findings {args.high_findings} exceed policy {args.max_high}")

    payload["registry"] = {
        "repository_uri": args.repository_uri,
        "image_digest": args.image_digest,
        "digest_uri": f"{args.repository_uri}@{args.image_digest}",
        "verified": True,
        "verified_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scan_status": args.scan_status,
        "critical_findings": args.critical_findings,
        "high_findings": args.high_findings,
    }
    atomic_write_manifest(args.manifest, payload, schema_path=args.schema)


def resolve_path(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not component or not isinstance(value, dict) or component not in value:
            raise ValueError(f"manifest path does not exist: {dotted_path!r}")
        value = value[component]
    return value


def command_get(args: argparse.Namespace) -> None:
    payload = load_validated_manifest(args.manifest, args.schema)
    value = resolve_path(payload, args.path)
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))


def command_validate_repository(args: argparse.Namespace) -> None:
    print(
        validate_ecr_repository_uri(
            args.uri,
            account=args.account,
            region=args.region,
            repository=args.repository,
        )
    )


def command_validate(args: argparse.Namespace) -> None:
    load_validated_manifest(args.manifest, args.schema)
    print("valid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--schema", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--compatibility-result", type=Path, required=True)
    create.add_argument("--compatibility-config", type=Path, required=True)
    create.add_argument("--commit-sha", required=True)
    create.add_argument("--reviewed-commit-sha", default="")
    create.add_argument("--tree-sha256", required=True)
    create.add_argument("--source-clean", choices=("true", "false"), required=True)
    create.add_argument("--commit-bound", choices=("true", "false"), required=True)
    create.add_argument("--package-lock-sha256", required=True)
    create.add_argument("--tag", required=True)
    create.set_defaults(handler=command_create)

    set_local = subparsers.add_parser("set-local")
    set_local.add_argument("--schema", type=Path, required=True)
    set_local.add_argument("--manifest", type=Path, required=True)
    set_local.add_argument("--config-id", required=True)
    set_local.set_defaults(handler=command_set_local)

    set_registry = subparsers.add_parser("set-registry")
    set_registry.add_argument("--schema", type=Path, required=True)
    set_registry.add_argument("--manifest", type=Path, required=True)
    set_registry.add_argument("--repository-uri", required=True)
    set_registry.add_argument("--account", required=True)
    set_registry.add_argument("--region", required=True)
    set_registry.add_argument("--image-digest", required=True)
    set_registry.add_argument("--scan-status", required=True)
    set_registry.add_argument("--critical-findings", type=int, required=True)
    set_registry.add_argument("--high-findings", type=int, required=True)
    set_registry.add_argument("--max-critical", type=int, default=0)
    set_registry.add_argument("--max-high", type=int, default=0)
    set_registry.set_defaults(handler=command_set_registry)

    get = subparsers.add_parser("get")
    get.add_argument("--schema", type=Path, required=True)
    get.add_argument("--manifest", type=Path, required=True)
    get.add_argument("--path", required=True)
    get.set_defaults(handler=command_get)

    validate_repository = subparsers.add_parser("validate-repository")
    validate_repository.add_argument("--uri", required=True)
    validate_repository.add_argument("--account", required=True)
    validate_repository.add_argument("--region", required=True)
    validate_repository.add_argument("--repository", required=True)
    validate_repository.set_defaults(handler=command_validate_repository)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--schema", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
