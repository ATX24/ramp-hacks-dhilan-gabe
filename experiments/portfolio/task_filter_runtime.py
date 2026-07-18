"""Fail-closed task-view resolution expected from the portfolio training image."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from distillery.contracts.hashing import sha256_hex
from distillery.contracts.manifest import SealedRunManifest


def resolve_task_dataset_dir(
    manifest: SealedRunManifest,
    dataset_bundle_root: Path,
) -> Path:
    """Resolve and verify the sealed task-filtered dataset view for one slot."""
    if manifest.tags.get("RunMode") != "portfolio-v2":
        raise ValueError("task-filter runtime only accepts portfolio-v2 manifests")
    raw = manifest.tags.get("PortfolioDatasetView")
    if raw is None:
        raise ValueError("portfolio manifest lacks PortfolioDatasetView")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("PortfolioDatasetView must decode to an object")
    relative = payload.get("relative_prefix")
    if not isinstance(relative, str):
        raise ValueError("portfolio dataset view lacks relative_prefix")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "views"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("portfolio dataset view prefix is unsafe")
    resolved = dataset_bundle_root.joinpath(*path.parts)
    root = dataset_bundle_root.resolve(strict=True)
    resolved = resolved.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("portfolio dataset view escapes the bundle root")
    split_sha256 = payload.get("split_sha256")
    if not isinstance(split_sha256, dict):
        raise ValueError("portfolio dataset view lacks split hashes")
    for split in ("train", "validation"):
        expected = split_sha256.get(split)
        if not isinstance(expected, str):
            raise ValueError(f"portfolio dataset view lacks {split} hash")
        split_path = resolved / f"{split}.jsonl"
        if split_path.is_symlink() or not split_path.is_file():
            raise FileNotFoundError(f"portfolio dataset view lacks {split_path.name}")
        if sha256_hex(split_path.read_bytes()) != expected:
            raise ValueError(f"portfolio dataset view {split} hash mismatch")
    task_filter = payload.get("task_filter")
    if task_filter != json.loads(manifest.tags["PortfolioTaskFilter"]):
        raise ValueError("portfolio dataset view and manifest task filters differ")
    return resolved
