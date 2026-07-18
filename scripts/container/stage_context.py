#!/usr/bin/env python3
"""Create a deterministic, explicit-allowlist Docker build context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_FILES = ("pyproject.toml", "uv.lock", "README.md", "LICENSE")
CONTAINER_FILES = (
    "Dockerfile",
    "container_entrypoint.py",
    "ml-compatibility.json",
    "verify_ml_compatibility.py",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SENSITIVE_PARTS = {".aws", "secrets"}
SENSITIVE_NAMES = {".env", "credentials", "id_rsa", "id_ed25519"}
EXCLUDED_SUFFIXES = {
    ".bin",
    ".key",
    ".onnx",
    ".pem",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_safe_source(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"staging refuses symlink: {path}")
    if not path.is_file():
        raise ValueError(f"required packaging file missing: {path}")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        raise ValueError(f"excluded cache path reached staging: {path}")
    if any(part in SENSITIVE_PARTS for part in path.parts):
        raise ValueError(f"sensitive path reached staging: {path}")
    if path.name in SENSITIVE_NAMES:
        raise ValueError(f"sensitive file reached staging: {path}")
    if path.suffix in EXCLUDED_SUFFIXES:
        raise ValueError(f"excluded binary/weight file reached staging: {path}")


def copy_file(source: Path, destination: Path) -> None:
    ensure_safe_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)
    os.utime(destination, (0, 0))


def source_files(repo: Path) -> list[Path]:
    source_root = repo / "src" / "distillery"
    if not source_root.is_dir():
        raise ValueError(f"package source directory missing: {source_root}")
    symlinks = sorted(path for path in source_root.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError(f"staging refuses package symlink: {symlinks[0]}")
    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.name != ".DS_Store"
    ]
    if not files:
        raise ValueError("src/distillery contains no package files")
    return sorted(files, key=lambda path: path.relative_to(repo).as_posix())


def build_inventory(destination: Path) -> tuple[list[dict[str, Any]], str]:
    inventory_path = destination / "SOURCE_FILES.json"
    files = sorted(
        (path for path in destination.rglob("*") if path.is_file() and path != inventory_path),
        key=lambda path: path.relative_to(destination).as_posix(),
    )
    records = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    canonical = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    payload = {
        "schema_version": "distillery.training.source-files.v1",
        "tree_sha256": tree_sha256,
        "files": records,
    }
    inventory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_path.chmod(0o644)
    os.utime(inventory_path, (0, 0))
    return records, tree_sha256


def normalize_directories(destination: Path) -> None:
    directories = sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    directories.append(destination)
    for directory in directories:
        directory.chmod(0o755)
        os.utime(directory, (0, 0))


def stage_context(repo: Path, destination: Path) -> str:
    repo = repo.resolve()
    destination = destination.resolve()
    if destination == repo or repo in destination.parents:
        raise ValueError("staging destination must be outside the repository")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"staging destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for relative in ROOT_FILES:
        copy_file(repo / relative, destination / relative)

    for source in source_files(repo):
        copy_file(source, destination / source.relative_to(repo))

    container_root = repo / "containers" / "training"
    for relative in CONTAINER_FILES:
        copy_file(
            container_root / relative,
            destination / "containers" / "training" / relative,
        )
    copy_file(container_root / ".dockerignore", destination / ".dockerignore")

    _records, tree_sha256 = build_inventory(destination)
    normalize_directories(destination)
    return tree_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tree_sha256 = stage_context(args.repo, args.destination)
    except (OSError, ValueError) as exc:
        print(f"staging error: {exc}", file=sys.stderr)
        return 2
    print(tree_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
